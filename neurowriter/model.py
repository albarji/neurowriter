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
from pytorch_transformers import BertForSequenceClassification, AdamW, WarmupLinearSchedule
from tempfile import NamedTemporaryFile
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from neurowriter.tokenizer import CLS, SEP, END


class Model:
    """Implements a text generation model that can be trained with a given Corpus"""

    def __init__(self):
        """Initializes a new Model. The model must be trained before text generation is possible"""
        self.model = None
        self.labels = []
        self.contextsize = None
        # Prepare GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _new_model(self, nlabels):
        """Initializes a BERT network model and places it in GPU. Returns the created model"""
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=nlabels)
        model.to(self.device)
        return model

    def _new_optimizer(self, model, lr=5e-5):
        """Creates a new optimizer for a given model

        Returns the created model

        Reference: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py#L80
        """
        no_decay = ['bias', 'LayerNorm.weight']
        # TODO: this parameter grouping is not doing anything, as we are not using weight decay
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        return optimizer

    def fit(self, dataset, outputdir, maxepochs=100, warmup=3, patience=3, checkpointepochs=10, gradient_accumulation_steps=1):
        """Trains a keras model with given parameters
        
        Arguments
            dataset: dataset to use for training
            outputdir: directory in which to save model
            maxepochs: maximum allowed training epochs for each model
            warmup: number of epochs warming up the learning rate
            patience: number of epochs without improvement for early stopping
            checkpointepochs: every checkpointepochs the current model will be saved to disk
            gradient_accumulation_steps: accumulate gradient along n batches. Allows large batch traing with small GPUs
        """
        logging.info(f"Training with batchsize={dataset.batchsize}x{gradient_accumulation_steps}")
        logging.info(f"Training batches {dataset.lentrainbatches}, validation batches {dataset.lenvalbatches}")

        # Check dataset
        if dataset.lentrainbatches == 0 or dataset.lenvalbatches == 0:
            raise ValueError("Insufficient data for training in the current setting")

        # Save dataset info into the model, which will be used later for generation
        self.labels = dataset.uniquetokens
        self.contextsize = dataset.tokensperpattern

        # Estimate a good learning rate with a provisional model
        logging.info(f"Estimating 'optimal' learning rate")
        lropt = self._find_lr(dataset, gradient_accumulation_steps=gradient_accumulation_steps)
        logging.info(f"Found learning rate={lropt}")

        # Create model and optimizer with found learning rate
        self.model = self._new_model(dataset.lenlabels)
        optimizer = self._new_optimizer(self.model, lropt)

        # Apply a 1cycle schedule (https://sgugger.github.io/the-1cycle-policy.html)
        t_total = maxepochs * dataset.lentrainbatches / gradient_accumulation_steps
        #cycle_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup, t_total=t_total)
        plateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=int(patience/3))

        logging.info(f"Training starts")
        global_step = 0
        best_eval_loss = math.inf
        no_improvement = 0
        best_model = None
        self.model.zero_grad()
        for epoch in tqdm(range(maxepochs), desc="Epoch", total=maxepochs):
            train_loss = 0
            self.model.train()
            epoch_iterator = tqdm(dataset.trainbatches(), desc="Batch", total=dataset.lentrainbatches)
            for step, batch in enumerate(epoch_iterator):
                # Forward pass through network
                model_loss = self._process_batch(self.model, batch)
                train_loss += model_loss.item()
                model_loss /= gradient_accumulation_steps

                # Backpropagation
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Model update
                if (step + 1) % gradient_accumulation_steps == 0:
                    #cycle_scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

            # TODO: change loss estimation, evaluation and checkpointing to operate in terms of global_step instead of epochs
            #  That makes more sense because fine-tuning a BERT model for state-of-the-art tasks requires just about 3 epochs

            train_loss /= dataset.lentrainbatches
            # Measure loss in validation set
            eval_loss = self.eval(self.model, dataset)

            plateau_scheduler.step(eval_loss)

            # Reports
            #lr = cycle_scheduler.get_lr()[0]
            logging.info(f"lr={optimizer.param_groups[0]['lr']}")
            logging.info(f"train_loss={train_loss}")
            logging.info(f"eval_loss={eval_loss}")

            # Generation sample
            sample = self.generate(dataset.tokenizer)
            logging.info(f"Generation sample: {sample}")

            # Check early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                no_improvement = 0
                best_model = copy.deepcopy(self.model)
            else:
                no_improvement += 1
            if no_improvement >= patience:
                logging.info(f"No improvement after {patience} epochs, stopping training")
                break

            # Save model checkpoint
            if checkpointepochs is not None and epoch % checkpointepochs == 0:
                check_dir = os.path.join(outputdir, 'checkpoint-{}'.format(epoch))
                self.save(check_dir)

        # Save best model
        self.model = best_model
        model_dir = os.path.join(outputdir, 'best')
        self.save(model_dir)

        return self

    def _find_lr(self, dataset, gradient_accumulation_steps=1, init_value = -8, final_value=1., beta = 0.98):
        """Finds a good learning rate by following a fastai heuristic.

        References: 
            https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
            https://github.com/davidtvs/pytorch-lr-finder
        """
        provmodel = self._new_model(dataset.lenlabels)
        optimizer = self._new_optimizer(provmodel)
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


    def eval(self, model, dataset):
        """Evaluates the performance of a model in a given dataset. The validation part of the dataset is used"""
        # Evaluation all data batches
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataset.valbatches(), desc="Evaluation batch", total=dataset.lenvalbatches):
                tmp_eval_loss = self._process_batch(model, batch)
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

        eval_loss /= nb_eval_steps
        return eval_loss

    def _process_batch(self, model, batch):
        """Processes a batch of data through the model, return the model loss for that batch"""
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}
        ouputs = model(**inputs)
        return ouputs[0]

    def generate(self, tokenizer, seed="", maxlength=100, temperature=1, appendseed=False):
        """Generates text using this trained model

        Arguments
            - tokenizer: tokenizer to use to split text.
            - seed: text seed to initialize generator. Default: empty string
            - maxlength: maximum length of generated text.
            - temperature: temperature of modified softmax, can be understood as the level of creativity
            - mergeseed: whether to append the given seed to the beginning of the returned generated text
        """
        tokenized_context = tokenizer.encodetext(seed)
        generated = []
        self.model.eval()

        # Pretokenize some special symbols
        ENDidx = tokenizer.encodetext(END)[0]

        for _ in range(maxlength):
            tokens, mask, types = tokenizer.encode_bert(tokenized_context)
            inputs = {
                'input_ids':      torch.tensor([tokens]).to(self.device),
                'attention_mask': torch.tensor([mask]).to(self.device),
                'token_type_ids': torch.tensor([types]).to(self.device)
            }
            with torch.no_grad():
                logits = self.model(**inputs)[0]
                logits = logits / temperature
                log_probs = F.softmax(logits, dim=-1)
                predicted_index = torch.multinomial(log_probs, num_samples=1)[0][0].tolist()
                predicted_index = self.labels[predicted_index]
                # Stop if END token generated
                if predicted_index == ENDidx:
                    break
                tokenized_context.append(predicted_index)
                generated.append(predicted_index)
            if len(tokenized_context) > self.contextsize:
                tokenized_context.pop(0)

        generatedtxt = tokenizer.decodeindexes(generated)
        if appendseed:
            # Account for a generated text starting with a subword suffix
            if len(generated) >= 2 and generated[0] == generated[1] == '#':
                generatedtxt = seed + generatedtxt[2:]
            else:
                generatedtxt = seed + " " + generatedtxt

        return generatedtxt

    def save(self, savefolder):
        """Saves the model into the given folder
        
        Saves both the model weights and the assignment between tokenizer indexes 
        and the train dataset metadata
        """
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  #todo Take care of distributed/parallel training
        model_to_save.save_pretrained(savefolder)
        # Save labels
        metadata = (self.labels, self.contextsize)
        with open(os.path.join(savefolder, 'labels.pkl'), 'wb') as f:
            pkl.dump(metadata, f)

    @classmethod
    def load(cls, loadfolder):
        """Loads a model from the given folder"""
        model = Model()

        # Load labels
        with open(os.path.join(loadfolder, 'labels.pkl'), 'rb') as f:
            metadata = pkl.load(f)
        model.labels, model.contextsize = metadata

        model.model = BertForSequenceClassification.from_pretrained(loadfolder, num_labels=len(model.labels))
        model.model.to(model.device)

        return model
