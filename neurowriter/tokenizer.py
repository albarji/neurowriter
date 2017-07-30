#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:49:48 2017

Functions for tokenizing texts.

All tokenizers are defined as classes that must feature a fit and transform
method, as well as a property "intertoken" that states the string to be
included between tokens when recovering the original text back (e.g. blank).

@author: Álvaro Barbero Jiménez
"""

from nltk.tokenize import word_tokenize

def tokenizerbyname(tokenizername):
    """Returns a tokenizer class by name"""
    tokenizers = {
        "char" : CharTokenizer,
        "word" : WordTokenizer
    }
    if tokenizername not in tokenizers:
        raise ValueError("Unknown tokenizer %s" % tokenizername)
    return tokenizers[tokenizername]

class CharTokenizer():
    """Tokenizer that splits a text into its basic characters"""
    
    def fit(self, text):
        # No training necessary
        pass
    
    def transform(self, text):
        return list(text)
    
    intertoken = ""

class WordTokenizer():
    """Tokenizer that splits text in words
    
    The default nltk tokenizer for english is used.
    """
    
    def fit(self, text):
        # No training necessary
        pass
    
    def transform(self, text):
        return word_tokenize(text)
    
    intertoken = " "

