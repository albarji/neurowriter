#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for creating, training and applying generation models.

@author: Álvaro Barbero Jiménez
"""

import copy
import math
import os
import pickle as pkl
from pytorch_transformers import BertForSequenceClassification, AdamW, WarmupLinearSchedule
from tempfile import NamedTemporaryFile
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
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

    def fit(self, dataset, outputdir, maxepochs=1000, patience=10, learningrate=5e-5, checkpointepochs=10):
        """Trains a keras model with given parameters
        
        Arguments
            dataset: dataset to use for training
            outputdir: directory in which to save model
            maxepochs: maximum allowed training epochs for each model
            patience: number of epochs without improvement for early stopping
            learningrate: learning rate to use in the optimizer
        """
        print(f"Training with learningrate={learningrate}")

        # Save dataset info into the model, which will be used later for generation
        self.labels = dataset.uniquetokens
        self.contextsize = dataset.tokensperpattern

        # Build model with input parameters
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', 
                                                                   num_labels=dataset.lenlabels)
        self.model.to(self.device)

        print("Number of training batches:", len(dataset))
        if len(dataset) == 0:
            raise ValueError("Insufficient data for training in the current setting")

        # Prepare optimizer and schedule (linear warmup and decay)
        # Reference: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py#L80
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learningrate, eps=1e-8)
        t_total = maxepochs * len(dataset)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

        # Tensorboard
        tb_writer = SummaryWriter()

        global_step = 0
        tr_loss = 0.0
        best_eval_loss = math.inf
        no_improvement = 0
        best_model = None
        self.model.zero_grad()
        ntrainbatches = len(list(dataset.trainbatches()))
        for epoch in tqdm(range(maxepochs), desc="Epoch", total=maxepochs):
            train_loss = 0
            epoch_iterator = tqdm(dataset.trainbatches(), desc="Batch", total=ntrainbatches)
            for batch in epoch_iterator:
                # Forward pass through network
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels':         batch[3]}
                ouputs = self.model(**inputs)
                model_loss = ouputs[0]
                train_loss += model_loss.mean().item()

                # Backpropagation
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                self.model.zero_grad()
                global_step += 1

            # Measure loss in validation set
            eval_loss = self.eval(dataset)

            lr = scheduler.get_lr()[0]
            tb_writer.add_scalar('lr', lr, global_step)
            print(f"lr={lr}")
            train_loss = train_loss / ntrainbatches
            print(f"train_loss={train_loss}")
            tb_writer.add_scalar('train_loss', train_loss, global_step)
            print(f"eval_loss={eval_loss}")
            tb_writer.add_scalar('eval_loss', eval_loss, global_step)

            # Check early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                no_improvement = 0
                best_model = copy.deepcopy(self.model)
            else:
                no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement after {patience} epochs, stopping training")
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

    def eval(self, dataset):
        """Evaluates the performance of the model in a given dataset. The validation part of the dataset is used"""
        # Evaluation all data batches
        eval_loss = 0.0
        nb_eval_steps = 0
        for batch in tqdm(dataset.valbatches(), desc="Evaluation batch"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels':         batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss = outputs[0]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        return eval_loss

    def generate(self, tokenizer, seed="", maxlength=100, temperature=1):
        """Generates text using this trained model

        Arguments
            - tokenizer: tokenizer to use to split text.
            - seed: text seed to initialize generator. Default: empty string
            - maxlength: maximum length of generated text.
            - temperature: temperature of modified softmax, can be understood as the level of creativity
        """
        tokenized_context = tokenizer.encodetext(seed)
        generated = []

        # Pretokenize some special symbols
        specialidx = {s: tokenizer.encodetext(s)[0] for s in [CLS, SEP, END]}

        for _ in range(maxlength):
            indexed_tokens = [specialidx[CLS]] + tokenized_context + [specialidx[SEP]]
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            with torch.no_grad():
                logits = self.model(tokens_tensor)[0]
                logits = logits / temperature
                log_probs = F.softmax(logits, dim=-1)
                predicted_index = torch.multinomial(log_probs, num_samples=1)[0][0].tolist()
                predicted_index = self.labels[predicted_index]
                # Stop if END token generated
                if predicted_index == specialidx[END]:
                    return tokenizer.decodeindexes(generated)
                tokenized_context.append(predicted_index)
                generated.append(predicted_index)
            if len(tokenized_context) > self.contextsize:
                tokenized_context.pop(0)
        return tokenizer.decodeindexes(generated)

    def save(self, savefolder):
        """Saves the model into the given folder
        
        Saves both the model weights and the assignment between tokenizer indexes 
        and the train dataset metadata
        """
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
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
