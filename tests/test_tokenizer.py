#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:27:15 2017

Tests for the tokenizer module

@author: Álvaro Barbero Jiménez
"""

import random
import string
import time

from neurowriter.tokenizer import Tokenizer, CLS, SEP, PAD


def test_reversible_encoding():
    """Encoding a text and decoding it produces the same result"""
    texts = [
        "Glory to mankind",
        "Endless forms most beautiful [END]",
        "abcdedg 1251957151"
    ]

    tokenizer = Tokenizer()

    for text in texts:
        coded = tokenizer.encodetext(text)
        decoded = tokenizer.decodeindexes(coded)

        print("Original text: %s" % text)
        print("Encoded text: " + str(coded))
        print("Decoded text: %s" % decoded)
        assert text == decoded


def test_bert_encoding():
    """Encoding a text in BERT format returns appropriate indexes"""
    tokenizer = Tokenizer()

    CLSidx = tokenizer.vocab[CLS]
    SEPidx = tokenizer.vocab[SEP]
    PADidx = tokenizer.vocab[PAD]
    inputs = [
        {"tokens": [1, 2, 3], "padding": 0},
        {"tokens": [4, 5, 6, 7], "padding": 1},
        {"tokens": [8], "padding": 3},
    ]
    expected = [
        [
            [CLSidx, 1, 2, 3, SEPidx], 
            [1, 1, 1, 1, 1], 
            [0, 0, 0, 0, 0]
        ],
        [
            [PADidx, CLSidx, 4, 5, 6, 7, SEPidx], 
            [0, 1, 1, 1, 1, 1, 1], 
            [0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [PADidx, PADidx, PADidx, CLSidx, 8, SEPidx], 
            [0, 0, 0, 1, 1, 1], 
            [0, 0, 0, 0, 0, 0]
        ]
    ]

    for inp, exp in zip(inputs, expected):
        print(f"Inputs: {inp}")
        print(f"Expected: {exp}")
        real = tokenizer.encode_bert(**inp)
        print(f"Real: {real}")
        for e, r in zip(exp, real):
            assert e == r
    