#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:33:20 2017

Module for loading training corpus in different formats.

@author: Álvaro Barbero Jiménez
"""

import pandas as pd
import json


class Corpus:
    """Class representing a corpus of documents.

    Behaves like an iterable of documents, each document being an iterable of text tokens.
    Provides methods for loading/saving the corpus in different formats.
    """
    def __init__(self, docs=None, conds=None):
        """Initializes the corpus with an iterable of documents, and optional conditioners

        Arguments:
            docs: iterable of documents, each document an iterable of tokens
            conds: iterable of conditioners, each conditioner a dictionary of conditioning variables
        """
        if docs is not None:
            if conds is None:
                self.corpus = [{"text": doc, "conditioners": None} for doc in docs]
            else:
                if len(docs) != len(conds):
                    raise ValueError("Lengths of docs iterable and conds iterable do not match")
                self.corpus = [{"text": doc, "conditioners": cond} for doc, cond in zip(docs, conds)]
        else:
            self.corpus = []

    def __iter__(self):
        """Iterate over the documents in the corpus"""
        for doc in self.corpus:
            yield doc["text"]

    def iterconditioners(self):
        """Iterate over conditioners in the corpus"""
        for doc in self.corpus:
            yield doc["conditioners"]

    def __getitem__(self, key):
        """Get the n-th document or a slice of documents in the corpus"""
        if isinstance(key, slice):
            return [self.corpus[idx]["text"]
                    for idx in range(key.start or 0, key.stop or len(self)-1, key.step or 1)]
        else:
            return self.corpus[key]["text"]

    def __len__(self):
        """Number of documents in the corpus"""
        return len(self.corpus)

    @classmethod
    def load_singletxt(cls, corpusfile):
        """Reads a corpus made of a single document, stored as a text file"""
        with open(corpusfile) as f:
            corpus = cls([f.read()])
        return corpus

    @classmethod
    def load_multilinetxt(cls, corpusfile):
        """Reads a corpus made of a text file, one document per line

        Final \n at each line are discarded from each document.
        """
        with open(corpusfile) as f:
            # Store lines removing final \n
            corpus = [line[:-1] for line in f.readlines()]
        return cls(corpus)

    @classmethod
    def load_csv(cls, corpusfile):
        """Reads a corpus from a CSV with additional conditioning data

        The CSV is assumed to represent one document per row, with the first
        column of the file containing the document text. Additional columns
        in the file are taken as conditioning variables.
        """
        df = pd.read_csv(corpusfile)
        txts = [line for line in df[df.columns[0]]]
        conds = [
            { key: row[key] for key in df.columns[1:] }
            for _, row in df.iterrows()
        ]
        return cls(txts, conds)

    @classmethod
    def load_json(cls, corpusfile):
        """Reads a corpus from a JSON with additional conditioning data

        The JSON be a list of objects, each object containing at least a key
        "text" with the document as an iterable of tokens. Additional keys per object are taken
        as conditioning variables.
        """
        # Load data
        with open(corpusfile) as f:
            corpus = json.load(f)
        # Separate texts from conditioners, for better management
        txts = [doc["text"] for doc in corpus]
        conds = [{key: doc[key] for key in doc if key != "text"} for doc in corpus]
        return cls(txts, conds)

    def save_json(self, corpusfile):
        data = []
        for doc in self.corpus:
            if doc["conditioners"] is not None:
                js = {key: doc["conditioners"][key] for key in doc["conditioners"]}
            else:
                js = {}
            js["text"] = doc["text"]
            data.append(js)
        with open(corpusfile, "w") as f:
            json.dump(data, f)

"""Dictionary of corpus loading functions indexed by a string"""
FORMATTERSBYNAME = {
    'singletxt': Corpus.load_singletxt,
    'multilinetxt': Corpus.load_multilinetxt,
    'csv': Corpus.load_csv,
    'json': Corpus.load_json
}