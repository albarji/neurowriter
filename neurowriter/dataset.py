#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module managing dataset creation for training the language generation model

@author: Álvaro Barbero Jiménez
"""

from itertools import chain
import torch
import logging

from neurowriter.genutils import batchedgenerator, infinitegenerator, maskedgenerator
from neurowriter.tokenizer import END

MAX_CONTEXT_BERT = 510  # BERT accepts 512, but we need to account for [CLS] and [SEP] added symbols

class Dataset():
    """Class managing dataset creation for training the language generation model"""

    def __init__(self, corpus, tokenizer, tokensperpattern, batchsize=8, trainvalratio=3):
        """Creates a new Dataset out of a given Corpus and Tokenizer.
        
        Arguments:
            - corpus: corpus to build this dataset from
            - tokenizer: tokenizer to process the corpus
            - tokensperpattern: number of previous tokens to be used to predict the current token
            - batchsize: size of the batches in which to group the paterns
            - trainvalratio: ratio between training and validation patterns.
                trainvalratio=3 means 3 training patterns for each validation pattern.
                If None or 0, the whole dataset is used both for train and validation
        """
        if tokensperpattern < 1:
            raise ValueError(f"tokensperpattern must be >= 1, received value was {tokensperpattern}")
        if tokensperpattern > MAX_CONTEXT_BERT:  
            logging.warning(f"Context too large, limiting to {MAX_CONTEXT_BERT}")
            self.tokensperpattern = MAX_CONTEXT_BERT
        else:
            self.tokensperpattern = tokensperpattern

        self.tokenizer = tokenizer
        self.batchsize = batchsize
        self.trainvalratio = trainvalratio
        # Tokenize whole corpus, store tokenized form
        self.tokenizedcorpus = [self.tokenizer.encodetext(doc) for doc in corpus]
        # Store unique tokens in this corpus
        # TODO: replace low frequency tokens by UNK
        self.uniquetokens = sorted(list(set(chain(*self.tokenizedcorpus, tokenizer.encodetext(END)))))
        # Prepare train/val masks
        if trainvalratio is not None and trainvalratio > 0:
            self.trainmask = [1] * trainvalratio + [0]
            self.valmask = [0] * trainvalratio + [1]
        else:
            self.trainmask = self.valmask = [1]
        # Measure dataset length (in training batches)
        self.len = len(list(self.trainbatches()))

    @batchedgenerator
    @maskedgenerator
    def _tokenizedpatterngenerator(self):
        """Generator of all patterns in the dataset, decorated to accept batches and masks"""
        for tokens in self.tokenizedcorpus:
            # Add document end token
            extended_tokens = tokens + self.tokenizer.encodetext(END)
            for i in range(len(extended_tokens)):
                # Encode context in BERT style
                padding = max(self.tokensperpattern - i, 0)
                real = self.tokensperpattern - padding
                x, mask, types = self.tokenizer.encode_bert(extended_tokens[i-real:i], padding)
                # Encode target token
                yindex = self._idx_to_label(extended_tokens[i])
                yield x, mask, types, yindex

    def _patterngenerator(self, mask=None):
        """Generator of encoded patterns as pytorch batches
        
        Arguments
            - mask: binary vector with 0 at the positions to ignore for data generation
        """
        # Prepare generator to produce infinite masked batches
        gen = self._tokenizedpatterngenerator(batchsize=self.batchsize, mask=mask)
        for batch in gen:
            tokens, mask, types, y = zip(*batch)
            yield (
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(types, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            )

    def trainbatches(self):
        """Generator of training batches"""
        return self._patterngenerator(mask=self.trainmask)

    def valbatches(self):
        """Generator of validation batches"""
        return self._patterngenerator(mask=self.valmask)

    def __len__(self):
        return self.len

    def _idx_to_label(self, tokenidx):
        return self.uniquetokens.index(tokenidx)

    @property
    def lenlabels(self):
        return len(self.uniquetokens)
