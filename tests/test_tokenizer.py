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

from neurowriter.tokenizer import build_tokenizer, EOS, START, END


def test_reversible_encoding():
    """Encoding a text and decoding it produces the same result"""
    texts = [
        "Glory to mankind",
        "Endless forms most beautiful [END]",
        "abcdedg 1251957151"
    ]

    tokenizer = build_tokenizer()

    for text in texts:
        coded = tokenizer.encode(text)
        decoded = tokenizer.decode(coded, skip_special_tokens=True)

        print("Original text: %s" % text)
        print("Encoded text: " + str(coded))
        print("Decoded text: %s" % decoded)
        assert text == decoded


def test_added_tokens():
    """Encoding a sentence with special added tokesn keeps them as a single token"""
    tokenizer = build_tokenizer()
    special_tokens = [EOS, START, END]
    for special_token in special_tokens:
        encoded = tokenizer.encode(special_token, add_special_tokens=False)
        assert len(encoded) == 1
