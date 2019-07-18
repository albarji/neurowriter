#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module managing dataset creation for training the language generation model

@author: Álvaro Barbero Jiménez
"""

from itertools import chain

from neurowriter.genutils import batchedpatternsgenerator, infinitegenerator, maskedgenerator
from neurowriter.tokenizer import CLS, SEP, NULL

class Dataset():
    """Class managing dataset creation for training the language generation model"""

    def __init__(self, corpus, tokenizer):
        """Creates a new Dataset out of a given Corpus and Tokenizer"""
        self.tokenizer = tokenizer
        # Tokenizer whole corpus, store tokenized form
        self.tokenizedcorpus = [self.tokenizer.tokenize(doc) for doc in corpus]
        # Store unique tokens in this corpus
        self.uniquetokens = sorted(list(set(chain(*self.tokenizedcorpus))))

    def patterngenerator(self, tokensperpattern, batchsize):
        """Infinite generator of encoded patterns as pytorch batches
        
        Arguments
            - tokensperpattern: how many tokens to include in every pattern
        """
        # Pre-tokenized all corpus documents, for efficiency
        for pattern in self._tokenizedpatterngenerator(self.tokenizedcorpus, tokensperpattern, infinite=True, 
                                                       batchsize=batchsize):
            yield pattern

    # Mask the patterns, then batch them, then repeat the cycle endlessly
    @infinitegenerator
    @batchedpatternsgenerator
    @maskedgenerator
    def _tokenizedpatterngenerator(self, tokenizedcorpus, tokensperpattern):
        for doctokens in tokenizedcorpus:
            # Append padding
            tokens = [NULL] * (tokensperpattern-1) + doctokens
            for i in range(tokensperpattern, len(tokens)):
                # BERT encoding for single sentences 
                # (https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py#L426)
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                x = self.tokenizer.encodetokens([CLS] + tokens[i-tokensperpattern:i] + [SEP])
                types = [0] * len(x)
                yindex = self.tokenizer.encodetokens([tokens[i]])[0]
                yield x, types, yindex
                # TODO: somehow put everything togetether into a pytorch torch, either here or in patterngenerator
