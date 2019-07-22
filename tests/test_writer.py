#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:35:18 2017

Tests for the Writer module

@author: Álvaro Barbero Jiménez
"""

import numpy as np

from neurowriter.writer import Writer
from neurowriter.corpus import Corpus


class MockModel():
    """Mock model for beam search tests"""
    def predict(self, X, **kwargs):
        return [[0.5, 0.3, 0.2]]


def _test_writer_beamsearch():
    """Beam search works as expected"""
    mockmodel = MockModel()
    corpus = Corpus(["abc"])
    encoder = Encoder(corpus=corpus, tokenizer=CharTokenizer())
    writer = Writer(mockmodel, encoder, creativity=0, beamsize=3, batchsize=3)
    seed = np.array([0, 0])
    
    expected = [0, 0, 0]
    obtained = writer.beamsearch(seed)
    print("Expected", expected)
    print("Obtained", obtained)
    assert obtained == expected


def _test_writer_beamsearch_long():
    """Beam search works as expected for long batch sizes"""
    mockmodel = MockModel()
    corpus = Corpus(["abc"])
    encoder = Encoder(corpus=corpus, tokenizer=CharTokenizer())
    batchsize = 10
    writer = Writer(mockmodel, encoder, creativity=0, beamsize=3,
                    batchsize=batchsize)
    seed = np.array([0, 0])
    expected = [0] * batchsize
    obtained = writer.beamsearch(seed)
    print("Expected", expected)
    print("Obtained", obtained)
    assert obtained == expected
