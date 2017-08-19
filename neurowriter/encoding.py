
# coding: utf-8

# Module for character encoding/decoding operations
#
# @author Álvaro Barbero Jiménez

import numpy as np
from itertools import chain
import pickle as pkl

from neurowriter.genutils import batchedpatternsgenerator, infinitegenerator
from neurowriter.genutils import maskedgenerator
from neurowriter.tokenizer import CharTokenizer

# Start sequence special character
START = "<START>"
# End sequence special character
END = "<END>"
# Null value in sequence special character
NULL = "<NULL>"
# Dictionary of all special characters
SPCHARS = [NULL, START, END]

class Encoder():
    # Dictionary of chars to numeric indices
    char2index = None    
    # Dictionary of numeric indices to chars
    index2char = None
    # Tokenizer used to process text
    tokenizer = None
    
    def __init__(self, corpus=None, tokenizer=None):
        """Creates an encoder from tokens to numbers and viceversa

        The encoder is built to represent all tokens present in the
        given corpus of texts, plus some special tokens for padding
        and sequence start/end. The special tokens are always codified
        as the first numbers, so that meaning is the same throughout
        different corpus.

        Arguments
            corpus: iterable of strings (corpus documents)
            tokenizer: tokenize object used to split the corpus into tokens.
        """
        if corpus is not None:
            # Train tokenizer on corpus
            self.tokenizer = tokenizer
            if self.tokenizer is None:
                self.tokenizer = CharTokenizer()
            self.tokenizer.fit(corpus.whole())
            # Get unique tokens from data
            tokens = set([token for token in self.tokenizer.transform(corpus.whole())])
            print('Total tokens:', len(tokens) + len(SPCHARS))
            self.char2index = dict((c, i) for i, c in enumerate(chain(SPCHARS ,tokens)))
            self.index2char = dict((i, c) for i, c in enumerate(chain(SPCHARS ,tokens)))

    def encodetext(self, text, addstart=False, fixlength=None):
        """Transforms a single text to tensor representation
        
        An special START character is added at the beginning to mark the start,
        if requested.
    
        If the fixlength parameter is provided, NULL characters are added
        at the beginning until such length is met.    
        """
        # Tokenize text
        tokens = self.tokenizer.transform(text)
        return self.encodetokens(tokens, addstart, fixlength)
    
    def encodetokens(self, tokens, addstart=False, fixlength=None):
        """Transforms a list of tokens to tensor representation
        
        An special START token is added at the beginning to mark the start,
        if requested.
    
        If the fixlength parameter is provided, NULL token are added
        at the beginning until such length is met.    
        """
        ln = len(tokens) + (1 if addstart else 0)
        x = np.zeros(ln, dtype=int)
        if addstart:
            x[0] = self.char2index[START]
            offset = 1
        else:
            offset = 0
        for t, token in enumerate(tokens):
            if token in self.char2index:
                x[offset + t] = self.char2index[token]
            else:
                print("WARNING: token", token, "not recognized")
    
        # Null padding, if requested            
        if fixlength is not None:
            xfix = np.zeros(fixlength, dtype=int)
            for i in range(min(ln,fixlength)):
                xfix[-i] = x[-i]
            for i in range(ln, fixlength):
                xfix[-i] = self.char2index[NULL]
            x = xfix
            
        return x
        
    def decodetext(self, X):
        """Transforms a matrix representing a text into text form
        
        Special characters are ignored"""
        text = ""
        for elem in X:
            char = self.index2char[np.argmax(elem)]
            if char not in SPCHARS:
                text += char + self.tokenizer.intertoken
                
        return text
    
    # Mask the patterns, then batch them, then repeat the cycle endlessly
    @infinitegenerator
    @batchedpatternsgenerator
    @maskedgenerator
    def patterngenerator(self, corpus, tokensperpattern, **kwargs):
        """Infinite generator of encoded patterns.
        
        Arguments
            - corpus: iterable of strings making up the corpus
            - tokensperpattern: how many tokens to include in every pattern
            - **kwargs: any other arguments are passed on to decodetext
        """
        #TODO: some tokenizers are very slow, repeating this for every
        # call to patterngeneration is not a good idea.
        # It actually seems that generating patterns is a bottleneck,
        # as there are large lapses of time where the GPUs are not being used!!
        for doc in corpus:
            # Tokenize document
            tokens = self.tokenizer.transform(doc)
            # Append padding
            tokens = [NULL] * (tokensperpattern-1) + [START] + tokens + [END]
            for i in range(tokensperpattern,len(tokens)):
                x = self.encodetokens(tokens[i-tokensperpattern:i], **kwargs)
                yindex = self.encodetokens([tokens[i]], **kwargs)[0]
                y = np.zeros(self.nchars)
                y[yindex] = 1.0
                yield x, y

    def save(self, filename):
        """Saves the encoding to a file"""
        with open(filename, "wb") as f:
            pkl.dump(self, f)
                       
    @property                
    def nchars(self):
        return len(self.char2index)

def loadencoding(filename):
    with open(filename, "rb") as f:
        encoder = pkl.load(f)
    return encoder
