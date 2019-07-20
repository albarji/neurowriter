#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module managing dataset creation for training the language generation model

@author: Álvaro Barbero Jiménez
"""

from itertools import chain
import torch

from neurowriter.genutils import batchedgenerator, infinitegenerator, maskedgenerator
from neurowriter.tokenizer import CLS, SEP, NULL

class Dataset():
    """Class managing dataset creation for training the language generation model"""

    def __init__(self, corpus, tokenizer, tokensperpattern):
        """Creates a new Dataset out of a given Corpus and Tokenizer.
        
        The number of tokens to use to form each training pattern must also be provided.
        """
        if tokensperpattern < 1:
            raise ValueError(f"tokensperpattern must be >= 1, received value was {tokensperpattern}")
        self.tokenizer = tokenizer
        self.tokensperpattern = tokensperpattern
        # Tokenize whole corpus, store tokenized form
        self.tokenizedcorpus = [self.tokenizer.tokenize(doc) for doc in corpus]
        # Store unique tokens in this corpus
        self.uniquetokens = sorted(list(set(chain(*self.tokenizedcorpus))))
        # Store indexes of special tokens
        self.special = {
            "[NULL]": self.tokenizer.tokenize([NULL])[0],
            "[CLS]": self.tokenizer.tokenize([CLS])[0],
            "[SEP]": self.tokenizer.tokenize([SEP])[0],
        }

    def patterngenerator(self, batchsize, mask=None):
        """Infinite generator of encoded patterns as pytorch batches
        
        Arguments
            - batchsize: size of batches to produce
            - mask: binary vector with 0 at the positions to ignore for data generation
        """
        # Prepare generator to produce infinite masked batches
        gen = self._tokenizedpatterngenerator(self.tokenizedcorpus, self.tokensperpattern, infinite=True, 
                                              batchsize=batchsize, mask=mask)
        for tokens, mask, types, y in gen:
            yield (
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(types, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            )

    # Mask the patterns, then batch them, then repeat the cycle endlessly
    @infinitegenerator
    @batchedgenerator
    @maskedgenerator
    def _tokenizedpatterngenerator(self, tokenizedcorpus):
        for tokens in tokenizedcorpus:
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
                    self.special["[NULL]"] * padded + 
                    self.special["[CLS]"] + tokens[i-real:i] + self.special["[SEP]"]
                )
                mask = [0] * padded + [1] * (real + 2)  # +2 to account for CLS and SEP
                types = [0] * len(x)
                yindex = self.tokenizer.encodetokens([tokens[i]])[0]
                yield x, mask, types, yindex
