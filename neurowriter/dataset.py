#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module managing dataset creation for training the language generation model

@author: Álvaro Barbero Jiménez
"""

from itertools import chain
import torch

from neurowriter.genutils import batchedgenerator, infinitegenerator, maskedgenerator
from neurowriter.tokenizer import CLS, SEP, NULL, UNK, SPECIAL_TOKENS

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
        """
        if tokensperpattern < 1:
            raise ValueError(f"tokensperpattern must be >= 1, received value was {tokensperpattern}")
        self.tokenizer = tokenizer
        self.tokensperpattern = tokensperpattern
        self.batchsize = batchsize
        self.trainvalratio = trainvalratio
        # Tokenize whole corpus, store tokenized form
        import pdb; pdb.set_trace()
        self.tokenizedcorpus = [self.tokenizer.encodetext(doc) for doc in corpus]
        # TODO: replace low frequency tokens by UNK
        # Store indexes of special tokens
        self.special = {sp: self.tokenizer.encodetext(sp)[0] for sp in SPECIAL_TOKENS}
        # Store unique tokens in this corpus
        self.uniquetokens = sorted(list(set(chain(*self.tokenizedcorpus, [self.special[UNK]]))))
        # Prepare train/val masks
        self.trainmask = [1] * trainvalratio + [0]
        self.valmask = [0] * trainvalratio + [1]
        # Measure dataset length (in training batches)
        self.len = len(list(self.trainbatches()))

    @batchedgenerator
    @maskedgenerator
    def _tokenizedpatterngenerator(self):
        """Generator of all patterns in the dataset, decorated to accept batches and masks"""
        for tokens in self.tokenizedcorpus:
            for i in range(len(tokens)):
                # BERT encoding for single sentences 
                # (https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py#L426)
                #  tokens:   [NULL] [NULL] ... [CLS] the dog is hairy . [SEP]
                #  mask:     0      0           1     1   1   1  1    1   1
                #  type_ids: 0      0      ...  0     0   0   0  0    0   0
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                # Add padding and [CLS] [SEP] symbols
                padded = max(self.tokensperpattern - i, 0)
                real = self.tokensperpattern - padded

                x = (
                    [self.special["[NULL]"]] * padded + 
                    [self.special["[CLS]"]] + tokens[i-real:i] + [self.special["[SEP]"]]
                )
                mask = [0] * padded + [1] * (real + 2)  # +2 to account for CLS and SEP
                types = [0] * len(x)
                yindex = self._idx_to_label(self.tokenizer.encodetokens([tokens[i]])[0])
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
