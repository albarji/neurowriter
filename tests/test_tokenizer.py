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

from neurowriter.tokenizer import Tokenizer


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
