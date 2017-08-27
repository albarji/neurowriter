#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:33:20 2017

Module for loading training corpus in different formats.

@author: Álvaro Barbero Jiménez
"""

import pandas as pd
import json
from copy import copy


class CorpusMixin:
    """Base documents corpus class"""
     
    def iterconditioners(self):
        for _ in range(len(self)):
            yield None


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
    
    def iterconditioners(self):
        for _, row in self.corpus.iterrows():
            yield row[self.corpus.columns[1:]]


class JsonCorpus(CorpusMixin):
    """Corpus loaded from a JSON with additional conditioning data
    
    The JSON be a list of objects, each object containing at least a key
    "text" with the document texts. Additional keys per object are taken
    as conditioning variables.
    """
    
    def load(self, corpusfile):
        # Load data
        with open(corpusfile) as f:
            corpus = json.load(f)
        # Separate texts from conditioners, for better management
        self.corpus = [
            {
                "text" : doc["text"],
                "conditioners" : {
                    key : doc[key]
                    for key in doc if key != "text"
                }
            }
            for doc in corpus
        ]
        
    def __iter__(self):
        for doc in self.corpus:
            yield doc["text"]
            
    def __getitem__(self, key):
        return self.corpus[key]["text"]
    
    def __len__(self):
        return len(self.corpus)
    
    def iterconditioners(self):
        for doc in self.corpus:
            yield doc["conditioners"]
