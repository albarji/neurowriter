#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:49:48 2017

Functions for tokenizing texts.

All tokenizers are defined as classes that must feature a fit and transform
method.

@author: Álvaro Barbero Jiménez
"""

from collections import OrderedDict
import re
from pytorch_transformers import BertTokenizer

# BERT sample start token
CLS = "[CLS]"
# BERT sentence separator token
SEP = "[SEP]"
# Null value in sequence special token
NULL = "[NULL]"
# Unknown token
UNK = "[UNK]"
# Dictionary of all special tokens
SPECIAL_TOKENS = [NULL, CLS, SEP, UNK]


class Tokenizer():
    """Wrapper over BertTokenizer that deals with special symbols for language generation"""
    def __init__(self):
        """Creates a new Tokenizer"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        add_tokens(self.tokenizer, SPECIAL_TOKENS)

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


def add_tokens(tokenizer, tokens):
    """Add new unsplittable tokens to an existing tokenizer, by making use of registered 'unused' tokens
    
    The changes are applied recursively to any tokenizers inside this one: basic and wordpiece tokenizers.
    """
    # Record new tokens in the tokenizer vocabulary (if it exists)
    if hasattr(tokenizer, "vocab"):
        # Find unused token indexes in vocabulary, enough to add all new tokens
        unused_pattern = re.compile("\[unused[0-9]+\]")
        unused_keys = [k for k, v in tokenizer.vocab.items() if unused_pattern.match(k)]
        if len(unused_keys) < len(tokens):
            raise ValueError("Not enought unused tokens left in vocabulary, cannot add new tokens")

        # Map new tokens to unused keys
        new_map = {k: token for k, token in zip(unused_keys, tokens)}

        # Create vocabulary again, replacing unused tokens with new tokens
        tokenizer.vocab = OrderedDict([
            (new_map[k], v) if k in new_map else (k, v) for k, v in tokenizer.vocab.items()
        ])
        
    # Add the new tokens to the never_split list (if it exists)
    if hasattr(tokenizer, 'never_split'):
        tokenizer.never_split = tokenizer.never_split + tokens
        
    # Do the same with recursive tokenizers (if they exist)
    subtokenizers = ["wordpiece_tokenizer", "basic_tokenizer"]
    for subtokenizer in subtokenizers:
        if hasattr(tokenizer, subtokenizer):
            add_tokens(getattr(tokenizer, subtokenizer), tokens)
