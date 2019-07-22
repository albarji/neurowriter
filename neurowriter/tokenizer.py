#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:49:48 2017

Functions for tokenizing texts.

All tokenizers are defined as classes that must feature a fit and transform
method.

@author: Álvaro Barbero Jiménez
"""

from pytorch_transformers import BertTokenizer

# BERT sample start token
CLS = "[CLS]"
# BERT sentence separator token
SEP = "[SEP]"
# Padding value in sequence special token
PAD = "[PAD]"
# Unknown token
UNK = "[UNK]"
# End of document token
END = "[END]"
# Dictionary of all special tokens
SPECIAL_TOKENS = [PAD, CLS, SEP, UNK, END]


class Tokenizer():
    """Wrapper over BertTokenizer that deals with special symbols for language generation"""

    def __init__(self):
        """Creates a new Tokenizer"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.tokenizer.add_tokens(SPECIAL_TOKENS)

    def encodetext(self, text):
        """Transforms a single text to indexes representation"""
        return self.tokenizer.encode(text)

    def decodeindexes(self, idx):
        """Transforms a list of indexes representing a text into text form

        Special characters are ignored"""
        return self.tokenizer.decode(idx, clean_up_tokenization_spaces=True)
