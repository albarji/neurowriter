#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:49:48 2017

Functions for creating a tokenizer.

@author: Álvaro Barbero Jiménez
"""

from transformers import AutoTokenizer

# BERT sample start token
CLS = "[CLS]"
# BERT sentence separator token
SEP = "[SEP]"
# Padding value in sequence special token
PAD = "[PAD]"
# Unknown token
UNK = "[UNK]"
# Start of document token
START = "[START]"
# End of document token
END = "[END]"
# End of sentence token (line break)
EOS = "[EOS]"
# Dictionary of all special tokens
SPECIAL_TOKENS = [PAD, CLS, SEP, UNK, START, END, EOS]
# Maximum number of tokens in Transformer models
MAX_CONTEXT = 512


def build_tokenizer(pretrained_model='bert-base-multilingual-cased'):
    """Creates a tokenizer based on a Transformers model, and adding special symbols for structured text generation"""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
    tokenizer.add_tokens([START, END])
    tokenizer.add_special_tokens({'eos_token': EOS})
    return tokenizer
