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
from tempfile import TemporaryFile
import re, collections
import operator

#from subwordnmt import learn_bpe

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
    
#    def fit(self, corpus):
#        # Invoke BPE algorithm
#        infile = TemporaryFile(mode="w")
#        infile.write(corpus)
#        outfile = TemporaryFile()
#        learn_bpe.main(infile, outfile, self.num_symbols)
#        infile.close()
#        #TODO
#        
#            
#    def parsebperesult(outfile):
#        #TODO
#        for line in outfile:
#            # Skip comments
#            if line[0] == "#":
#                continue
#            # 
#            matched = re.match("([^ ]+) (.+?)(?=abc)", line)
#
#    def get_stats(vocab):
#        pairs = collections.defaultdict(int)
#        for word, freq in vocab.items():
#            symbols = word.split()
#            for i in range(len(symbols)-1):
#                pairs[symbols[i],symbols[i+1]] += freq
#        return pairs
#    
#    def merge_vocab(pair, v_in):
#        v_out = {}
#        bigram = re.escape(' '.join(pair))
#        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#        for word in v_in:
#            w_out = p.sub(''.join(pair), word)
#            v_out[w_out] = v_in[word]
#        return v_out
#    
#    def samplebpe():
#        vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
#                'n e w e s t </w>':6, 'w i d e s t </w>':3}
#        num_merges = 10
#        for i in range (num_merges):
#            pairs = get_stats(vocab)
#            best = max(pairs, key=pairs.get)
#            vocab = merge_vocab(best, vocab)
#            print(best)
    
#    def initsymbols(self, corpus):
#        """Initializes the symbol representations using single chars"""
#        return collections.Counter(corpus)
    
    def initfreqs(self, corpus):
        """Initializes the symbol statistics with char pairs"""
        stats = collections.defaultdict(int)
        for a, b in zip(corpus[:-1], corpus[1:]):
            stats[a,b] += 1
        return stats
    
    def updatefreqs(self, freqs, symbols, leftsymbol, rightsymbol, corpus):
        # Find positions where the new combined symbol appears
        newsymbol = leftsymbol + rightsymbol
        p = re.compile(newsymbol)
        for match in p.finditer(corpus):
            nextpos = match.end()
            # Now find the following symbol
            nextsymbol = self.bestmatch(corpus[nextpos:], symbols)
            freqs[newsymbol, nextsymbol] += 1
            freqs[rightsymbol, nextsymbol] -= 1
            # TODO: update also with the PREVIOUS symbol
            
    def bestmatch(self, string, symbols):
        """Find the best matching symbol at the beggining of a string"""
        # Sort symbols by length, so larger symbols have precedence
        srt = sorted(symbols, key=lambda x: len(x), reverse=True)
        # Detect any symbol, with precedence for larger ones
        detector = re.compile('|'.join(srt))
        match = detector.match(string)
        if match is not None:
            return match.group()
        else:
            return None
    
    def fit(self, corpus):
        #import pdb; pdb.set_trace()  #TODO
        # Initialize symbols with chars
        self.symbols = set(corpus)
        # Compute char pairs frequencies
        freqs = self.initfreqs(corpus)
        # Merge steps
        for i in range(self.numsymbols - len(self.symbols)):
            # Find most frequent pair
            leftsymbol, rightsymbol = max(freqs, key=freqs.get)
            newsymbol = leftsymbol + rightsymbol
            # If too infrequent, stop procedure
            if freqs[(leftsymbol, rightsymbol)] < self.minfreq:
                break
            # Add to dictionary of symbols
            self.symbols.add(newsymbol)
            # Update pairs frequencies with the new symbol
            self.updatefreqs(freqs, self.symbols, leftsymbol, rightsymbol, corpus)
            # Remove processed pair from frequencies
            del freqs[(leftsymbol, rightsymbol)]
        #import pdb; pdb.set_trace()  #TODO
