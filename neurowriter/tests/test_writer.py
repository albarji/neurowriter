#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:35:18 2017

Tests for the Writer module

@author: Álvaro Barbero Jiménez
"""

import numpy as np

from neurowriter.writer import Writer
from neurowriter.encoding import Encoder

def test_writer_beamsearch():
    """Beam search works as expected"""
    
    class MockModel():
        """Mock model for beam search tests"""        
        def predict(self, X, **kwargs):
            return [[0.5, 0.3, 0.2]]
        
    mockmodel = MockModel()
    encoder = Encoder(corpus="abc", tokenizer="char")
    writer = Writer(mockmodel, encoder, creativity=0, beamsize=3, batchsize=3)
    seed = np.array([0,0])
    
    expected = [0, 0, 0]
    obtained = writer.beamsearch(seed)
    print("Expected", expected)
    print("Obtained", obtained)
    assert(obtained == expected)
