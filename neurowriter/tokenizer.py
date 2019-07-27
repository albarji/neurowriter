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
        self.tokenizer.add_tokens([END])

    def encodetext(self, text):
        """Transforms a single text to indexes representation"""
        return self.tokenizer.encode(text)

    def decodeindexes(self, idx):
        """Transforms a list of indexes representing a text into text form

        Special characters are ignored"""
        return self.tokenizer.decode(idx, clean_up_tokenization_spaces=True)

    def encode_bert(self, tokens, padding=0):
        """Encodes a sequence of tokens as the set of index sequences expected by BERT

        BERT encoding for single sentences 
        (https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py#L426)
            tokens:   [PAD] [PAD]   ... [CLS] the dog is hairy . [SEP]
            mask:     0      0           1     1   1   1  1    1   1
            type_ids: 0      0      ...  0     0   0   0  0    0   0
        The mask has 1 for real tokens and 0 for padding tokens. Only real
        tokens are attended to.

        Arguments:
            - padding: how much padding to add at the beginning of the encoded sequence
        """
        x = (
            [self.tokenizer.vocab[PAD]] * padding + 
            [self.tokenizer.vocab[CLS]] + tokens + [self.tokenizer.vocab[SEP]]
        )
        mask = [0] * padding + [1] * (len(tokens) + 2)  # +2 to account for CLS and SEP
        types = [0] * len(x)
        return x, mask, types

    @property
    def vocab(self):
        return self.tokenizer.vocab
