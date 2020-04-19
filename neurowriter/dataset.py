#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module managing dataset creation for training the language generation model

@author: Álvaro Barbero Jiménez
"""

from itertools import cycle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from neurowriter.tokenizer import START, END, MAX_CONTEXT


class Dataset(TorchDataset):
    """Class managing dataset creation for training the language generation model"""

    def __init__(self, corpus, tokenizer, indices):
        """Creates a new Dataset out of a given Corpus and Tokenizer.
        
        Arguments:
            - corpus: corpus to build this dataset from
            - tokenizer: tokenizer to process the corpus
            - tokensperpattern: maximum number of previous tokens to be used to predict the current token
            - indices: iterable of indices for the patterns from the corpus that will be used in this dataset
        """
        self.tokenizer = tokenizer
        self.tokenized_corpus = self._tokenize_corpus(corpus)
        self._build_indices(indices)

    def _tokenize_corpus(self, corpus):
        """Pre-tokenizes a corpus for faster pattern generation
        
        Also adds the special BEGIN and END tokens
        """
        return [
            self.tokenizer.encode(f"{START} {doc} {END}", add_special_tokens=False)
            for doc in corpus
        ]

    def _build_indices(self, indices):
        """Build fast document->token indices to allow fast retrieval of patterns"""
        i = 0
        self.indices_docs = []
        self.indices_tokens = []
        for doc_index, doc in enumerate(self.tokenized_corpus):
            for token_index in range(len(doc) - 1):
                if i in indices:
                    self.indices_docs.append(doc_index)
                    self.indices_tokens.append(token_index)
                i += 1

    def __len__(self):
        """Number of patterns in the dataset"""
        return len(self.indices_docs)

    def __getitem__(self, index):
        """Returns a dataset pattern (X, y) by index"""
        doc = self.tokenized_corpus[self.indices_docs[index]]
        token_index = self.indices_tokens[index]
        X = doc[:token_index+1] + [self.tokenizer.mask_token_id] # Add [MASK] as last token to every text
        y = doc[token_index+1]
        return X, y

    def _collateX(self, encoded_docs):
        """Collates an iterable of encoded documents into a batch ready for model processing"""
        # Find length of longest doc, reserving 2 spaces for [CLS] and [SEP] tokens
        max_length = max([len(encoded_doc) for encoded_doc in encoded_docs]) + 2
        max_length = min(max_length, MAX_CONTEXT)
        # Prepare tuples (encoded_doc, None), which are required by batch_encode_plus
        tuples = [(encoded_doc, None) for encoded_doc in encoded_docs]
        # Batch encoding
        tensors = self.tokenizer.batch_encode_plus(tuples, max_length=max_length, pad_to_max_length=True, return_tensors="pt")
        return tensors

    def _collateXY(self, encoded_docs):
        """Collates an iterable of texts into a batch inputs and outputs for the language generation model
        
        The last token in each text is used as the target.
        """
        batch = self._collateX([x for x, _ in encoded_docs])
        # To make use of the Language Model loss function we need to mark the correct output token for each input token 
        # Special value -100 ignores that token in the loss calculation
        batch["masked_lm_labels"] = torch.tensor([-100]).repeat(batch["input_ids"].shape)
        # We just activate masked tokens for the loss funcion
        for i in range(len(batch["input_ids"])):
            mask_index = torch.nonzero(batch["input_ids"][i] == self.tokenizer.mask_token_id)[0][0]
            _, batch["masked_lm_labels"][i][mask_index] = encoded_docs[i]
        return batch

    def loader(self, batch_size=8, include_targets=True):
        """Returns a Torch DataLoader for this Dataset"""
        if include_targets:
            collate_fn = self._collateXY
        else:
            collate_fn = self._collateX

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    @classmethod
    def build_datasets(cls, corpus, tokenizer, trainvalratio=3):
        """Builds training and validation datasets from a given corpus
        Tokenizer
        Arguments:Tokenizer
            - corpus: corpus to build this dataset fromTokenizer
            - tokenizer: tokenizer to process the corpusTokenizer
            - trainvalratio: ratio between training and validation patterns.
                trainvalratio=3 means 3 training patterns for each validation pattern.
                If None or 0, the whole dataset is used both for train and validation

        Returns two Dataset objects, one for the training and another for the validation set
        """
        # Prepare train/val masks patterns
        if trainvalratio is not None and trainvalratio > 0:
            train_pattern = [1] * trainvalratio + [0]
            val_pattern = [0] * trainvalratio + [1]
        else:
            train_pattern = val_pattern = [1]
        
        # Measure total dataset length
        # From each document we will predict all tokens + [END] token
        length = sum([len(doc) + 1 for doc in corpus])

        # Generate full training/validation masks
        train_indices = [i for i, mask in zip(range(length), cycle(train_pattern)) if mask]
        val_indices = [i for i, mask in zip(range(length), cycle(val_pattern)) if mask]

        # Create Datasets for masked data
        train_dataset = Dataset(corpus, tokenizer, train_indices)
        val_dataset = Dataset(corpus, tokenizer, val_indices)
        return train_dataset, val_dataset
