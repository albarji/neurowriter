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

from neurowriter.tokenizer import SubwordTokenizer

   
def test_SubwordTokenizerExact():
    """The subword tokenizer obtains the exact expected symbols for toy data"""
    
    corpus = "aaabdaaabac"
    expected = {'a','b','c','d','aa','ab','aaab'}
    
    tok = SubwordTokenizer(numsymbols=1024, minfreq=2)
    
    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    print("Found but not expected", tok.symbols - expected)
    assert(expected == tok.symbols)

def test_SubwordTokenizerAtLeast():
    """The subword tokenizer obtains at least a set of expected symbols"""
    
    corpus = "a green dog inside a green house"
    expected = {
                'a', ' ', 'g', 'r', 'e', 'n', 'o', 'i', 's', 'd', 'h', 'u',
                'a green '
            }
    
    tok = SubwordTokenizer(numsymbols=1024, minfreq=2)
    
    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    assert(len(expected - tok.symbols) == 0)
    
def test_SubwordTokenizerTransform():
    """The subword tokenizer correctly transforms a toy example"""
    train = "aaabdaaabac"    
    tok = SubwordTokenizer(numsymbols=1024, minfreq=2)
    tok.fit(train)
    
    test = "aaabababaabcdabaa"
    expected = ["aaab", "ab", "ab", "aa", "b", "c", "d", "ab", "aa"]
    
    obtained = tok.transform(test)
    print("Expected", expected)
    print("Obtained", obtained)
    assert(obtained == expected)
    
def test_SubwordTokenizerTimes():
    """Performs som runtime tests on the subword tokenizer"""
    n = 10000
    symbols = 5000
    print("Measuring SubwordTokenizer times for input lenght %s, symbols %s"
           % (n, symbols))
    data = ''.join(random.choice(string.ascii_letters) for _ in range(n))
    start = time.time()
    tok = SubwordTokenizer(numsymbols=symbols, minfreq=2)
    tok.fit(data)
    end = time.time()
    print("Fit time:", end-start)

    start = time.time()
    tok.transform(data)
    end = time.time()
    print("Transform time:", end-start)
