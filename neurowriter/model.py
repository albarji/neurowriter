#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for creating, training and applying generation models.

@author: Álvaro Barbero Jiménez
"""

import copy
import logging
import math
import numpy as np
import os
import pickle as pkl
from transformers import AutoConfig, AutoModelWithLMHead, AdamW, get_linear_schedule_with_warmup
from tempfile import NamedTemporaryFile
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from neurowriter.dataset import Dataset
from neurowriter.tokenizer import build_tokenizer, START, CLS, SEP, END, EOS, MAX_CONTEXT


class Model:
    """Implements a text generation model that can be trained with a given Corpus"""

    def __init__(self, pretrained_model='bert-base-multilingual-cased', dropout=0.1):
        """Initializes a new Model. The model must be trained before text generation is possible"""
        self.model = None
        self.dropout = dropout
        self.pretrained_model = pretrained_model
        # Build tokenizer
        self.tokenizer = build_tokenizer(self.pretrained_model)
        # Prepare GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Model device: {self.device}")
        # Initialize model
        self.model = self._new_model()

    def _new_model(self):
        """Initializes a BERT network model and places it in GPU. Returns the created model"""
        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout
        )
        model = AutoModelWithLMHead.from_pretrained(self.pretrained_model, config=config)
        model.to(self.device)
        # Rearrange embeddings matrix for tokenizer size
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def _new_optimizer(self, lr=5e-5):
        """Creates a new optimizer for a given model

        Returns the created model

        Reference: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py#L80
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)
        return optimizer

    def fit(self, corpus, outputdir, maxepochs=3, lr=5e-5, patience=3, min_lr=1e-7, checkpointepochs=10, gradient_accumulation_steps=1, batch_size=16, trainvalratio=3):
        """Trains a keras model with given parameters
        
        Arguments
            corpus: corpus to use for training
            outputdir: directory in which to save model
            maxepochs: maximum allowed training epochs for each model
            lr: initial learning rate
            patience: number of epochs without improvement for model backtracking and lr reduction
            min_lr: stop training after reaching this learning rate
            checkpointepochs: every checkpointepochs the current model will be saved to disk
            gradient_accumulation_steps: accumulate gradient along n batches. Allows large batch traing with small GPUs
            batch_size: size of training batches
            trainvalratio: ratio between training and validation patterns.
        """
        logging.info(f"Training with batchsize={batch_size}x{gradient_accumulation_steps}")
        logging.info(f"Initial learning rate {lr}")

        # Prepare datasets
        train_dataset, val_dataset = Dataset.build_datasets(corpus, self.tokenizer, trainvalratio=trainvalratio)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Insufficient data for training in the current setting")
        train_loader = train_dataset.loader(batch_size=batch_size)
        val_loader = val_dataset.loader(batch_size=batch_size)

        logging.info(f"Training batches {len(train_loader)}, validation batches {len(val_loader)}")

        # Create optimizer
        optimizer = self._new_optimizer(lr)

        # Decreasing learning rate
        t_total = maxepochs * len(train_loader) / gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = t_total)

        logging.info(f"Training starts")
        global_step = 0
        best_eval_loss = math.inf
        best_lr = lr
        no_improvement = 0
        best_model = self.model
        self.model.zero_grad()
        for epoch in tqdm(range(maxepochs), desc="Training", total=maxepochs):
            train_loss = 0
            self.model.train()
            batch_iterator = tqdm(train_loader, desc=f"Training epoch {epoch}", total=len(train_loader))
            for step, batch in enumerate(batch_iterator):
                model_loss = self._process_batch(batch)
                train_loss += model_loss.item()
                model_loss /= gradient_accumulation_steps

                # Backpropagation
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Model update
                if (step + 1) % gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

            # TODO: change loss estimation, evaluation and checkpointing to operate in terms of global_step instead of epochs
            #  That makes more sense because fine-tuning a BERT model for state-of-the-art tasks requires just about 3 epochs

            train_loss /= len(train_loader)
            # Measure loss in validation set
            eval_loss = self._eval(val_loader)

            # Reports
            current_lr = scheduler.get_lr()[0]
            logging.info(f"lr={current_lr}")
            logging.info(f"train_loss={train_loss}")
            logging.info(f"eval_loss={eval_loss}")

            # Generation sample
            sample = self.generate()
            logging.info(f"Generation sample: {sample}")

            # Save model checkpoint
            if checkpointepochs is not None and epoch % checkpointepochs == 0:
                check_dir = os.path.join(outputdir, 'checkpoint-{}'.format(epoch))
                self.save(check_dir)

            # Check early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                no_improvement = 0
                best_model = copy.deepcopy(self.model)
                best_lr = current_lr
            else:
                no_improvement += 1
            if no_improvement >= patience:
                logging.info(f"No improvement after {patience} epochs, backtracking training and continuing with reduced lr")
                current_lr = best_lr = best_lr/10
                if current_lr < min_lr:
                    logging.info(f"Minimum learning rate {min_lr} reached, stopping training")
                    break
                self.model = best_model
                optimizer = self._new_optimizer(current_lr)
                t_total = (maxepochs - epoch+1) * len(train_loader) / gradient_accumulation_steps
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = t_total)
                no_improvement = 0

        # Save best model
        self.model = best_model
        model_dir = os.path.join(outputdir, 'best')
        self.save(model_dir)

        logging.info(f"Training finished")

        return self

    def _find_lr(self, dataset, gradient_accumulation_steps=1, init_value = -8, final_value=1., beta = 0.98):
        """Finds a good learning rate by following a fastai heuristic.

        References: 
            https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
            https://github.com/davidtvs/pytorch-lr-finderdataset.lenvalbatches
        """
        provmodel = self._new_model(dataset.ntokens)
        optimizer = self._new_optimizer()
        provmodel.train()

        optimgroups = len(optimizer.param_groups)

        batches_iterator = dataset.trainbatches()
        losses = []
        lr_values = np.logspace(init_value, final_value, 50)
        for lr in lr_values:
            for i in range(optimgroups):
                optimizer.param_groups[i]['lr'] = lr
            train_loss = 0
            for _ in range(gradient_accumulation_steps):
                # Gather next batch, looping over the data if necessary
                try:
                    batch = next(batches_iterator)
                except StopIteration:
                    batches_iterator = dataset.trainbatches()
                    batch = next(batches_iterator)

                # Forward pass through network
                model_loss = self._process_batch(provmodel, batch)
                train_loss += model_loss.item()
                model_loss /= gradient_accumulation_steps

                # Backpropagation
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(provmodel.parameters(), 1.0)

            # Model update
            optimizer.step()
            provmodel.zero_grad()

            # Store loss
            losses.append(train_loss)

        # Losses smoothing
        avgloss = 0
        for i in range(len(losses)):
            losses[i] = beta * avgloss + (1-beta) * losses[i]

        # Recommend the lr value corresponding to 1/10 times the smallest loss
        lr_smallest = lr_values[np.argmin(losses)]
        return lr_smallest / 10

    def _eval(self, val_loader):
        """Evaluates the performance of a model in a given validation dataset loader"""
        # Evaluation all data batches
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluation batch", total=len(val_loader)):
                tmp_eval_loss = self._process_batch(batch)
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

        eval_loss /= nb_eval_steps
        return eval_loss

    def _process_batch(self, batch):
        """Processes a batch of data through the model, return the model loss for that batch"""
        # Move tensors GPU
        for key in batch:
            batch[key] = batch[key].to(self.device)
        # Forward pass through network
        outputs = self.model(**batch)
        return outputs[0]

    def generate(self, seed="", maxlength=100, temperature=1, top_k=0, top_p=0.9, appendseed=False):
        """Generates text using this trained model

        Arguments
            - seed: text seed to initialize generator. Default: empty string
            - maxlength: maximum length of generated text.
            - temperature: temperature of modified softmax, can be understood as the level of creativity
            - top_k: keep only top k tokens with highest probability (top-k filtering).
            - top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            - appendseed: whether to append the given seed to the beginning of the returned generated text
        """
        self.model.eval()

        # Prepare some pre-encoded tokens
        ENDidx = self.tokenizer.encode(END, add_special_tokens=False)[0]

        # Prepare seed text
        encoded_context = self.tokenizer.encode(f"{START} {seed}", add_special_tokens=False)
        generated = []

        for _ in range(maxlength):
            inputs = self.tokenizer.encode_plus(encoded_context + [self.tokenizer.mask_token_id], return_tensors="pt")
            with torch.no_grad():
                # Move tensors GPU
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                logits = self.model(**inputs)[0][0][-2]  # Token probabilities for MASK token
                logits = logits / temperature
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                log_probs = F.softmax(filtered_logits, dim=-1)
                predicted_index = torch.multinomial(log_probs, num_samples=1)[0].tolist()
                # Stop if END token generated
                if predicted_index == ENDidx:
                    break
                encoded_context.append(predicted_index)
                generated.append(predicted_index)
            # Clip context if too large
            if len(encoded_context) + 3 > MAX_CONTEXT:  # length + MASK + CLS + SEP
                encoded_context.pop(0)

        generatedtxt = self.tokenizer.decode(generated)
        # Replace EOS with newlines
        generatedtxt = generatedtxt.replace(EOS, "\n")
        # Append seed (if requested)
        if appendseed and len(seed) > 1:
            # Account for a generated text starting with a subword suffix
            if len(generated) >= 2 and generatedtxt[0] == generatedtxt[1] == '#':
                generatedtxt = seed + generatedtxt[2:]
            else:
                generatedtxt = seed + " " + generatedtxt

        return generatedtxt

    def save(self, savefolder):
        """Saves the model into the given folder"""
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        # Save tokenizer
        self.tokenizer.save_pretrained(savefolder)
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  #todo Take care of distributed/parallel training
        model_to_save.save_pretrained(savefolder)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Reference:
            https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
