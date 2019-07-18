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
        "Endless forms most beautiful",
        "abcdedg 1251957151"
    ]
    options = [
        {"addstart":False, "fixlength":None},
        {"addstart":True, "fixlength":None},
        {"addstart":False, "fixlength":16},
        {"addstart":True, "fixlength":16}
    ]

    tokenizer = Tokenizer()

    for text in texts:
        for option in options:
            coded = tokenizer.encodetext(text, **option)
            decoded = tokenizer.decodeindexes(coded)

            print("Original text: %s" % text)
            print("Encoded text: " + str(coded))
            print("Decoded text: %s" % decoded)
            assert text == decoded
