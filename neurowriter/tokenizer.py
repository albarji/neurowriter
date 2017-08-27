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

import re, collections
from itertools import chain

from neurowriter.symbols import NULL

def tokenizerbyname(tokenizername):
    """Returns a tokenizer class by name"""
    tokenizers = {
        "char" : CharTokenizer,
        "word" : WordTokenizer,
        "subword" : SubwordTokenizer,
    }
    if tokenizername not in tokenizers:
        raise ValueError("Unknown tokenizer %s" % tokenizername)
    return tokenizers[tokenizername]

class CharTokenizer():
    """Tokenizer that splits a text into its basic characters"""
    
    def fit(self, corpus):
        # No training necessary
        pass
    
    def transform(self, text):
        return list(text)

class WordTokenizer():
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

class SubwordTokenizer():
    """Tokenizer that splits text in descriptive subword parts
    
    Subword parts are trained for each corpus, building from single
    characters and using a Byte Pair Encoding (BPE) method.
    
    References:
        - https://en.wikipedia.org/wiki/Byte_pair_encoding
        - https://github.com/rsennrich/subword-nmt
        - https://arxiv.org/abs/1508.07909
    """
    
    def __init__(self, numsymbols=4096, minfreq=2):
        self.numsymbols = numsymbols
        self.minfreq = minfreq
        self.detector = None
    
    def initfreqs(self, corpus):
        """Initializes the symbol statistics with char pairs"""
        stats = collections.defaultdict(int)
        for doc in corpus:
            for a, b in zip(doc[:-1], doc[1:]):
                stats[a,b] += 1
        return stats
    
    def mergesymbols(self, corpus, symbols, freqs, leftsymbol, rightsymbol):
        """Merges two symbols in the encoding
        
        Arguments:
            - corpus: current list of docs, each a list of symbols
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
        newcorpus = []
        for doc in corpus:
            padded = doc + [NULL]
            newdoc = []
            locations = []
            i = 0
            while i < len(doc):
                if padded[i] == leftsymbol and padded[i+1] == rightsymbol:
                    locations.append(len(newdoc))
                    newdoc.append(newsymbol)
                    i += 2 # Skip already processed next symbol
                else:
                    newdoc.append(padded[i])
                    i += 1
            newcorpus.append(newdoc)

            for i in locations:
                # Update frequencies with previous symbol
                if i > 0:
                    prevsymbol = newdoc[i-1]
                    freqs[prevsymbol, newsymbol] += 1
                    freqs[prevsymbol, leftsymbol] -= 1
                # Update frequencies with next symbol
                if i < len(newdoc)-1:
                    nextsymbol = newdoc[i+1]
                    freqs[newsymbol, nextsymbol] += 1
                    freqs[rightsymbol, nextsymbol] -= 1
                         
        # Delete statistics of merged symbols
        del freqs[(leftsymbol, rightsymbol)]
                         
        return newcorpus, freqs, symbols
        
    def compile(self):
        """Compiles the parsing expression for more efficiency"""
        # Sort symbols by length, so larger symbols have precedence
        srt = sorted(self.symbols, key=lambda x: len(x), reverse=True)
        # Escape special symbols
        srt = [re.escape(token) for token in srt]
        # Detect any symbol, with precedence for larger ones
        self.detector = re.compile('|'.join(srt))
        
            
    def bestmatch(self, string, symbols):
        """Find the best matching symbol at the beggining of a string"""
        if self.detector is None:
            raise ValueError("Tokenizer has not been fitted")
        match = self.detector.match(string)
        if match is not None:
            return match.group()
        else:
            return None
    
    def fit(self, corpus):
        # Cast corpus to list of lists of symbols
        corpus = [list(doc) for doc in corpus]
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
            symbol = self.bestmatch(text[i:], self.symbols)
            transformed.append(symbol)
            i += len(symbol)
        return transformed
