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
import re, collections

# Special token to mark tokenizer borders
BORDER = "<TK_BORDER>"

#from subwordnmt import learn_bpe

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
    
    intertoken = ""

class WordTokenizer():
    """Tokenizer that splits text in words
    
    The default nltk tokenizer for english is used.
    """
    
    def fit(self, corpus):
        # No training necessary
        pass
    
    def transform(self, text):
        return word_tokenize(text)
    
    intertoken = " "

class SubwordTokenizer():
    """Tokenizer that splits text in descriptive subword parts
    
    Subword parts are trained for each corpus, building from single
    characters and using a Byte Pair Encoding (BPE) method.
    
    References:
        - https://en.wikipedia.org/wiki/Byte_pair_encoding
        - https://github.com/rsennrich/subword-nmt
        - https://arxiv.org/abs/1508.07909
    """
    
    def __init__(self, numsymbols=1024, minfreq=2):
        self.numsymbols = numsymbols
        self.minfreq = minfreq
        self.detector = None
    
    def initfreqs(self, corpus):
        """Initializes the symbol statistics with char pairs"""
        stats = collections.defaultdict(int)
        for a, b in zip(corpus[:-1], corpus[1:]):
            stats[a,b] += 1
        return stats
    
    def mergesymbols(self, corpus, symbols, freqs, leftsymbol, rightsymbol):
        """Merges two symbols in the encoding
        
        Arguments:
            - corpus: current list of symbols representing the corpus
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
        
        # Go over the corpus, find occurrences of the given pair and merge
        padded = corpus + [BORDER]
        newcorpus = []
        i = 0
        while i < len(corpus):
            if padded[i] == leftsymbol and padded[i+1] == rightsymbol:
                newcorpus.append(newsymbol)
                i += 2 # Skip already processed next symbol
            else:
                newcorpus.append(padded[i])
                i += 1
            
        # Find positions where the new combined symbol appears, update freqs
        for i, s in enumerate(newcorpus):
            if s == newsymbol:
                # Update frequencies with previous symbol
                if i > 0:
                    prevsymbol = newcorpus[i-1]
                    freqs[prevsymbol, newsymbol] += 1
                    freqs[prevsymbol, leftsymbol] -= 1
                # Update frequencies with next symbol
                if i < len(newcorpus)-1:
                    nextsymbol = newcorpus[i+1]
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
        # Cast corpus to list of symbols
        corpus = list(corpus)
        # Initialize symbols with chars
        self.symbols = set(corpus)
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

    def transform(self, corpus):
        transformed = []
        i = 0
        while i < len(corpus):
            symbol = self.bestmatch(corpus[i:], self.symbols)
            transformed.append(symbol)
            i += len(symbol)
        return transformed
        
    intertoken = ""