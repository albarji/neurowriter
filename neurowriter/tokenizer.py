#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:49:48 2017

Functions for tokenizing texts.

All tokenizers are defined as classes that must feature a fit and transform
method.

@author: Álvaro Barbero Jiménez
"""

import re
import collections
from itertools import chain

from neurowriter.linkedlist import LinkedList


def tokenizerbyname(tokenizername):
    """Returns a tokenizer class by name"""
    tokenizers = {
        "char": CharTokenizer,
        "word": WordTokenizer,
        "subword": SubwordTokenizer,
    }
    if tokenizername not in tokenizers:
        raise ValueError("Unknown tokenizer %s" % tokenizername)
    return tokenizers[tokenizername]


class CharTokenizer:
    """Tokenizer that splits a text into its basic characters"""

    @staticmethod
    def fit(corpus):
        # No training necessary
        pass

    @staticmethod
    def transform(text):
        return list(text)


class WordTokenizer:
    """Tokenizer that splits text in words
    
    Punctuation and whitespace symbols are kept as individual
    tokens, so the input text can be rebuild by just concatenating
    all tokens.
    
    Words that have a low number of occurrences in the text are broken
    down to individual characters, to reduce the overall number of tokens.
    """
    
    def __init__(self, numsymbols=4096, minfreq=2):
        self.numsymbols = numsymbols
        self.minfreq = minfreq
        self.symbols = None
        # Precompile parsing expression
        self.parser = re.compile('(\W)')
    
    def fit(self, corpus):
        # First add all basic characters to the dictionary
        self.symbols = set(chain(*[doc for doc in corpus]))
        # Split input in words, get unique tokens and counts
        tokens = collections.Counter(
            chain(*[self.parser.split(doc) for doc in corpus])
        )
        # Filter out unfrequent symbols
        freqsymbols = [(symbol, freq) for symbol, freq in tokens.items() 
                        if freq >= self.minfreq]
        # Sort remaining symbols by frequency
        srt = sorted(freqsymbols, key=lambda x: x[1], reverse=True)
        # Remove already included characters
        remain = [symbol for symbol, freq in srt if symbol not in self.symbols]
        # Fill the dictionary with symbols until max allowed
        freespace = self.numsymbols - len(self.symbols)
        self.symbols.update(remain[0:freespace])
    
    def transform(self, text):
        # Break input in words
        tokens = self.parser.split(text)
        # For every word not in the recognized symbols list, break into chars
        # If a character has never been seen before, it is ignored
        result = []
        for token in tokens:
            if token in self.symbols:
                result.append(token)
            else:
                for char in token:
                    if char in self.symbols:
                        result.append(char)
        return result


class SubwordTokenizer:
    """Tokenizer that splits text in descriptive subword parts

    Subword parts are trained for each corpus, building from single
    characters and using a Byte Pair Encoding (BPE) method.

    References:
        - https://en.wikipedia.org/wiki/Byte_pair_encoding
        - https://github.com/rsennrich/subword-nmt
        - https://arxiv.org/abs/1508.07909
    """

    def __init__(self, numsymbols=4096, minfreq=5):
        self.numsymbols = numsymbols
        self.minfreq = minfreq
        self.detector = None

    @staticmethod
    def initfreqs(corpus):
        """Initializes the symbol statistics with char pairs

        The input must be a list of docs, each a LinkedList of symbols
        """
        stats = collections.defaultdict(int)
        for doc in corpus:
            for node in doc.iternodes():
                if node.nxt is not None:
                    stats[node.value, node.nxt.value] += 1
        return stats

    def mergesymbols(self, corpus, symbols, freqs, leftsymbol, rightsymbol):
        """Merges two symbols in the encoding

        Arguments:
            - corpus: current list of docs, each a LinkedList of symbols
            - symbols: current set of symbols
            - freqs: current symbol pairs statistics
            - leftsymbol, rightsymbol: symbols to merge

        Returns:
            - new corpus with merged symbols
            - new list of symbols
            - updated symbol pair statistics
        """
        # Add new symbol to set
        newsymbol = leftsymbol + rightsymbol
        self.symbols.add(newsymbol)

        # Go over each doc in corpus, find occurrences of the given pair and merge
        for doc in corpus:
            for node in doc.iternodes():
                if node.value == leftsymbol and node.nxt is not None and node.nxt.value == rightsymbol:
                    node.mergewithnext()
                    # Update frequencies with previous symbol
                    if node.prev is not None:
                        prevsymbol = node.prev.value
                        freqs[prevsymbol, newsymbol] += 1
                        freqs[prevsymbol, leftsymbol] -= 1
                    # Update frequencies with next symbol
                    if node.nxt is not None:
                        nextsymbol = node.nxt.value
                        freqs[newsymbol, nextsymbol] += 1
                        freqs[rightsymbol, nextsymbol] -= 1

        # Delete statistics of merged symbols
        del freqs[(leftsymbol, rightsymbol)]

        return corpus, freqs, symbols

    def compile(self):
        """Compiles the parsing expression for more efficiency"""
        # Sort symbols by length, so larger symbols have precedence
        srt = sorted(self.symbols, key=lambda x: len(x), reverse=True)
        # Escape special symbols
        srt = [re.escape(token) for token in srt]
        # Detect any symbol, with precedence for larger ones
        self.detector = re.compile('|'.join(srt))

    def bestmatch(self, string):
        """Find the best matching symbol at the beggining of a string"""
        if self.detector is None:
            raise ValueError("Tokenizer has not been fitted")
        match = self.detector.match(string)
        if match is not None:
            return match.group()
        else:
            return None

    def fit(self, corpus):
        # Cast corpus to list of linked-lists of symbols
        corpus = [LinkedList(doc) for doc in corpus]
        # Initialize symbols with chars
        self.symbols = set(chain(*[doc for doc in corpus]))
        # Compute char pairs frequencies
        freqs = self.initfreqs(corpus)
        # Merge steps
        for i in range(self.numsymbols - len(self.symbols)):
            # Find most frequent pair
            leftsymbol, rightsymbol = max(freqs, key=freqs.get)
            # If too infrequent, stop procedure
            if freqs[(leftsymbol, rightsymbol)] < self.minfreq:
                break
            # Merge symbols
            corpus, freqs, self.symbols = self.mergesymbols(
                corpus,
                self.symbols,
                freqs,
                leftsymbol,
                rightsymbol
            )
        # Compile tokenizer for found symbols
        self.compile()

    def transform(self, text):
        transformed = []
        i = 0
        while i < len(text):
            symbol = self.bestmatch(text[i:])
            transformed.append(symbol)
            i += len(symbol)
        return transformed

