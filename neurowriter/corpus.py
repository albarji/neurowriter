#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:33:20 2017

Module for loading training corpus in different formats.

@author: Álvaro Barbero Jiménez
"""

import pandas as pd

class CorpusMixin():
    DOCSPLITTER = "<DOC>"
    
    def whole(self):
        """Returns an iterable for the whole corpus as a list of chars.
        
        Different documents are separated through an special character
        """
        for doc in self:
            for char in doc:
                yield char
            yield self.DOCSPLITTER

class SingleTxtCorpus(CorpusMixin):
    """Corpus made of a txt file with a single document"""
    
    def load(self, corpusfile):
        with open(corpusfile) as f:
            self.corpus = f.read()
            
    def __iter__(self):
        yield self.corpus
        
    def __getitem__(self, key):
        if key == 0:
            return self.corpus
        else:
            raise IndexError()
            
    def __len__(self):
        return 1

class MultiLineCorpus(CorpusMixin):
    """Corpus made of a txt file, one document per line"""
    
    def load(self, corpusfile):
        with open(corpusfile) as f:
            # Store lines removing final \n
            self.corpus = [line[:-1] for line in f.readlines()]
            
    def __iter__(self):
        for line in self.corpus:
            yield line

    def __getitem__(self, key):
        return self.corpus[key]

    def __len__(self):
        return len(self.corpus)

class StringsCorpus(CorpusMixin):
    """Corpus made from an iterable of strings"""
    
    def load(self, strings):
        self.corpus = strings
            
    def __iter__(self):
        for line in self.corpus:
            yield line

    def __getitem__(self, key):
        return self.corpus[key]

    def __len__(self):
        return len(self.corpus)
    
class CsvCorpus(CorpusMixin):
    """Corpus loaded from a CSV with additional conditioning data
    
    The CSV is assumed to represent one document per row, with the first
    column of the file containing the document text. Additional columns
    in the file are taken as conditioning variables.
    """
    
    def load(self, corpusfile):
        self.corpus = pd.read_csv(corpusfile)
        
    def __iter__(self):
        for line in self.corpus[self.corpus.columns[0]]:
            yield line
            
    def __getitem__(self, key):
        return self.corpus[self.corpus.columns[0]][key]
    
    def __len__(self):
        return len(self.corpus)
