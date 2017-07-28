
# coding: utf-8

# Module for character encoding/decoding operations
#
# @author Álvaro Barbero Jiménez

import numpy as np
import json
from itertools import chain

from neurowriter.genutils import batchedgenerator, infinitegenerator

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
    
    def __init__(self, corpus=None):
        """Creates an encoder from char to numbers and viceversa

        The encoder is built to represent all characters present in the
        given corpus of texts, plus some special characters for padding
        and sequence start/end. The special characters are always codified
        as the first numbers, so that meaning is the same throughout
        different corpus.

        Arguments
            corpus: iterable of strings
        """
        if corpus is not None:
            chars = set([letter for text in corpus for letter in text])
            print('Total chars:', len(chars) + len(SPCHARS))
            self.char2index = dict((c, i) for i, c in enumerate(chain(SPCHARS ,chars)))
            self.index2char = dict((i, c) for i, c in enumerate(chain(SPCHARS ,chars)))

    def encodetext(self, text, addstart=False, fixlength=None):
        """Transforms a single text to tensor representation
        
        An special START character is added at the beginning to mark the start,
        if requested.
    
        If the fixlength parameter is provided, NULL characters are added
        at the beginning until such length is met.    
        """
        ln = len(text) + (1 if addstart else 0)
        X = np.zeros((ln, len(self.char2index)))
        if addstart:
            X[0, self.char2index[START]] = 1
            offset = 1
        else:
            offset = 0
        for t, char in enumerate(text):
            if char in self.char2index:
                X[offset + t, self.char2index[char]] = 1
            else:
                print("WARNING: character", char, "not recognized")
    
        # Null padding, if requested            
        if fixlength is not None:
            Xfix = np.zeros((fixlength, len(self.char2index)))
            for i in range(min(ln,fixlength)):
                Xfix[-i] = X[-i]
            for i in range(ln, fixlength):
                Xfix[-i, self.char2index[NULL]] = 1
            X = Xfix
            
        return X
        
    def decodetext(self, X):
        """Transforms a matrix representing a text into text form
        
        START and END characters are ignored"""
        text = ""
        for elem in X:
            char = self.index2char[np.argmax(elem)]
            if char not in SPCHARS:
                text += char
        return text
    
    @batchedgenerator
    @infinitegenerator
    def patterngenerator(self, corpus, tokensperpattern, start=0, end=None, **kwargs):
        """Infinite generator of encoded patterns.
        
        Arguments
            - corpus: list of tokens making up the corpus
            - tokensperpattern: how many tokens to include in every pattern
            - start: first corpus token to use in pattern generation
            - end: last corpus token to use in pattern generation
            - **kwargs: any other arguments are passed on to decodetext
        """
        end = len(corpus) if end is None else end
        for i in range(start,end-tokensperpattern):
            x = self.encodetext(corpus[i:i+tokensperpattern], **kwargs)
            y = self.encodetext(corpus[i+tokensperpattern], **kwargs).squeeze()
            yield x, y

    def save(self, filename):
        """Saves the encoding to a file"""
        with open(filename, "w") as f:
            json.dump({"char2index" : self.char2index, 
                       "index2char" : self.index2char}, 
                       f)
                       
    @property                
    def nchars(self):
        return len(self.char2index)

def loadencoding(filename):
    encoder = Encoder()
    with open(filename, "r") as f:
        js = json.load(f)
        encoder.char2index = js["char2index"]
        encoder.index2char = {int(idx) : js["index2char"][idx] for idx in js["index2char"]}
    return encoder
