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

from neurowriter.tokenizer import WordTokenizer, BPETokenizer, SubwordsListTokenizer


def test_WordTokenizerExact():
    """The word tokenizer obtains the exact expected symbols for toy data"""
    
    corpus = ["a green big dog inside a green big house"]
    expected = {' ', 'a', 'd', 'e', 'g', 'h', 'i', 'n', 'o', 'r', 's', 'u', 
                'b', "green", "big"}
    
    tok = WordTokenizer(numsymbols=1024, minfreq=2)
    
    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    print("Found but not expected", tok.symbols - expected)
    assert(expected == tok.symbols)


def test_WordTokenizerTransform():
    """The word tokenizer correctly transforms a toy example"""
    train = ["a green big dog and a cat inside a green big house"]
    tok = WordTokenizer(numsymbols=1024, minfreq=2)
    tok.fit(train)
    
    test = "a green cat inside a big green house"
    expected = ["a", " ", "green", " ", "c", "a", "t", " ", "i", "n", "s",
                "i", "d", "e", " ", "a", " ", "big", " ", "green", " ", "h", 
                "o", "u", "s", "e"]
    
    obtained = tok.transform(test)
    print("Expected", expected)
    print("Obtained", obtained)
    assert(obtained == expected)


def test_BPETokenizerExact():
    """The subword tokenizer obtains the exact expected symbols for toy data"""
    
    corpus = ["aaabdaaabac"]
    expected = {'a', 'b', 'c', 'd', 'aaab'}
    
    tok = BPETokenizer(numsymbols=1024, minfreq=2)
    
    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    print("Found but not expected", tok.symbols - expected)
    assert(expected == tok.symbols)


def test_BPETokenizerAtLeast():
    """The subword tokenizer obtains at least a set of expected symbols"""
    
    corpus = ["a green dog inside a green house"]
    expected = {
                'a', ' ', 'g', 'r', 'e', 'n', 'o', 'i', 's', 'd', 'h', 'u',
                'a green '
            }
    
    tok = BPETokenizer(numsymbols=1024, minfreq=2, crosswords=True)
    
    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    assert(len(expected - tok.symbols) == 0)


def test_BPETokenizerAtLeast_nocrossword():
    """The subword tokenizer obtains at least a set of expected symbols, avoiding word crossings"""

    corpus = ["a green dog inside a green house"]
    expected = {
        'a', ' ', 'g', 'r', 'e', 'n', 'o', 'i', 's', 'd', 'h', 'u',
        'green'
    }

    tok = BPETokenizer(numsymbols=1024, minfreq=2, crosswords=False)

    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    assert (len(expected - tok.symbols) == 0)


def test_BPETokenizerTransform():
    """The subword tokenizer correctly transforms a toy example"""
    train = ["aaababdaaabcab"]
    tok = BPETokenizer(numsymbols=1024, minfreq=2)
    tok.fit(train)
    
    test = "aaabababaabcdabaa"
    expected = ["aaab", "ab", "ab", "a", "ab", "c", "d", "ab", "a", "a"]
    
    obtained = tok.transform(test)
    print("Expected", expected)
    print("Obtained", obtained)
    assert(obtained == expected)


def test_BPETokenizerTimes():
    """Performs some runtime tests on the subword tokenizer"""
    n = 10000
    symbols = 5000
    print("Measuring SubwordTokenizer times for input lenght %s, symbols %s"
           % (n, symbols))
    data = [''.join(random.choice(string.ascii_letters) for _ in range(n))]
    start = time.time()
    tok = BPETokenizer(numsymbols=symbols, minfreq=2)
    tok.fit(data)
    end = time.time()
    print("Fit time:", end-start)

    start = time.time()
    tok.transform(data[0])
    end = time.time()
    print("Transform time:", end-start)


def test_SubwordsListTokenizerExact():
    """The subwords list tokenizer obtains the exact expected symbols for toy data"""

    corpus = ["a green big dog inside a green big house"]
    subwords = ["gre", "en", "bi", "og", "insi", "de", "hou", "se"]
    expected = {"gre", "en", "bi", "g", "d", "og", "insi", "de", "a", "hou", "se",
                "r", "e", "n", "b", "i", "g", "s", "d", "h", "o", "u", " "}

    tok = SubwordsListTokenizer(subwords=subwords)

    tok.fit(corpus)
    print("Expected", expected)
    print("Obtained", tok.symbols)
    print("Expected but not found", expected - tok.symbols)
    print("Found but not expected", tok.symbols - expected)
    assert (expected == tok.symbols)


def test_SubwordsListTransform():
    """The subword list tokenizer correctly transforms a toy example"""
    train = ["aaababdaaabcab"]
    tok = SubwordsListTokenizer(subwords=["aa", "ab", "bd", "aaab", "ca", "b"])
    tok.fit(train)

    test = "aaabababaabcdabaa"
    expected = ["aaab", "ab", "ab", "aa", "b", "c", "d", "ab", "aa"]

    obtained = tok.transform(test)
    print("Expected", expected)
    print("Obtained", obtained)
    assert (obtained == expected)
