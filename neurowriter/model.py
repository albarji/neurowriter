#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for creating, training and applying generation models.

@author: Álvaro Barbero Jiménez
"""

import logging
import numpy as np
import os
from transformers import AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments
import torch
import torch.nn.functional as F

from neurowriter.dataset import Dataset, TextGenerationCollator
from neurowriter.tokenizer import build_tokenizer, START, CLS, SEP, END, EOS, MAX_CONTEXT


class Model:
    """Implements a text generation model that can be trained with a given Corpus"""

    def __init__(self, pretrained_model='bert-base-multilingual-cased', special_tokens=[], dropout=0.1):
        """Initializes a new Model. The model must be trained before text generation is possible"""
        self.model = None
        self.dropout = dropout
        self.pretrained_model = pretrained_model
        # Build tokenizer
        self.tokenizer = build_tokenizer(self.pretrained_model, special_tokens=special_tokens)
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
        model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model, config=config)
        model.to(self.device)
        # Rearrange embeddings matrix for tokenizer size
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def fit(self, corpus, outputdir, max_steps=1000, checkpointsteps=100, lr=5e-5, gradient_accumulation_steps=1, batch_size=16, trainvalratio=3):
        """Trains a keras model with given parameters
        from transformers import TrainingArguments
            max_steps: maximum allowed training steps for model training
            checkpointsteps: after how many steps a checkpoint of the model will be written down to disk
            lr: initial learning rate
            gradient_accumulation_steps: accumulate gradient along n batches. Allows large batch traing with small GPUs
            batch_size: size of training batches
            trainvalratio: ratio between training and validation patterns.
        """
        logging.info(f"Training with batchsize={batch_size}x{gradient_accumulation_steps}")
        logging.info(f"Initial learning rate {lr}")

        # Prepare datasets
        logging.info(f"Building training datasets...")
        train_dataset, val_dataset = Dataset.build_datasets(corpus, self.tokenizer, trainvalratio=trainvalratio)
        if len(train_dataset) == 0:
            raise ValueError("Insufficient data for training in the current setting")

        logging.info(f"Training patterns {len(train_dataset)}")
        if val_dataset is not None:
            logging.info(f"Validation patterns {len(val_dataset)}")

        # Prepare data collator
        collator = TextGenerationCollator(self.tokenizer, self.device)

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=outputdir,  # Folder in which to save the trained model
            overwrite_output_dir=True,  # Whether to overwrite previous models found in the output folder
            per_gpu_train_batch_size=batch_size,  # batch size during training
            per_gpu_eval_batch_size=128,  # batch size during evaluation (prediction)
            max_steps=max_steps,  # Model training steps
            logging_steps=100,  # After how many training steps (batches) a log message showing progress will be printed
            save_steps=checkpointsteps,  # After how many training steps (batches) the model will be checkpointed to disk
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Prepare Trainer object
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(outputdir)

        # Train!
        trainer.train()

        # Save model
        self.model.save_pretrained(outputdirPero en general, digamos que mi sensación es que la cosa empieza bien)

        logging.info(f"Training finished")

        return self

    def generate(self, seed="", maxlength=512, temperature=1, top_k=0, top_p=0.9, appendseed=False):
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
