
# coding: utf-8

# Module for token encoding/decoding operations in the language generation model
#
# @author Álvaro Barbero Jiménez

import numpy as np
from itertools import chain
import pickle as pkl
from collections import OrderedDict

from neurowriter.genutils import batchedpatternsgenerator, infinitegenerator, maskedgenerator
from neurowriter.tokenizer import get_tokenizer, CLS, SEP, NULL, SPECIAL_TOKENS


class Encoder:
    # Tokenizer used to process text
    tokenizer = None
    
    def __init__(self, tokenizer):
        """Creates an encoder from tokens to numbers and viceversa

        The encoder is built to represent all tokens present in the
        given base tokenizer, plus some special tokens for padding
        and sequence start/end. The special tokens are always codified
        as the first numbers, so that meaning is the same throughout
        different corpus.

        Arguments
            tokenizer: tokenize object used to split the corpus into tokens.
        """
        self.tokenizer = tokenizer

    def encodetext(self, text, addstart=False, fixlength=None):
        """Transforms a single text to tensor representation
        
        An special CLS character is added at the beginning to mark the start,
        if requested.
    
        If the fixlength parameter is provided, NULL characters are added
        at the beginning until such length is met.    
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        return self.encodetokens(tokens, addstart, fixlength)
    
    def encodetokens(self, tokens, addstart=False, fixlength=None):
        """Transforms a list of tokens to tensor representation
        
        An special CLS token is added at the beginning to mark the start,
        if requested.
    
        If the fixlength parameter is provided, NULL token are added
        at the beginning until such length is met.    
        """
        tokenslen = len(tokens) + (1 if addstart else 0)
        capacity = tokenslen if fixlength is None else fixlength

        # Initialize with null padding
        x = self.tokenizer.convert_tokens_to_ids([NULL] * capacity)

        # Add start symbol if requested
        if addstart:
            x[-tokenslen] = self.tokenizer.convert_tokens_to_ids([CLS])[0]

        # Add standard tokens
        x[-len(tokens):] = self.tokenizer.convert_tokens_to_ids(tokens)
            
        return x

    def decodeindexes(self, idx):
        """Transforms a list of indexes representing a text into text form

        Special characters are ignored"""
        special_idx = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        filtered = [x for x in idx if x not in special_idx]
        tokens = self.tokenizer.decode(filtered, clean_up_tokenization_spaces=True)
        return tokens

    def patterngenerator(self, corpus, tokensperpattern, **kwargs):
        """Infinite generator of encoded patterns.
        
        Arguments
            - corpus: iterable of strings making up the corpus
            - tokensperpattern: how many tokens to include in every pattern
            - **kwargs: any other arguments are passed on to decodetext
        """
        # Pre-tokenized all corpus documents, for efficiency
        tokenizedcorpus = [self.tokenizer.tokenize(doc) for doc in corpus]
        for pattern in self._tokenizedpatterngenerator(tokenizedcorpus, tokensperpattern, **kwargs):
            yield pattern

    # Mask the patterns, then batch them, then repeat the cycle endlessly
    @infinitegenerator
    @batchedpatternsgenerator
    @maskedgenerator
    def _tokenizedpatterngenerator(self, tokenizedcorpus, tokensperpattern, **kwargs):
        for tokens in tokenizedcorpus:
            # Append padding
            tokens = [NULL] * (tokensperpattern-1) + [CLS] + tokens + [SEP]
            for i in range(tokensperpattern, len(tokens)):
                x = self.encodetokens(tokens[i-tokensperpattern:i], **kwargs)
                yindex = self.encodetokens([tokens[i]], **kwargs)[0]
                y = np.zeros(self.nchars)
                y[yindex] = 1.0
                yield x, y
