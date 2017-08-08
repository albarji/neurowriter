#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:55:20 2017

Tests for the models creation module.

@author: Álvaro Barbero Jiménez
"""

import os
import tensorflow as tf
import numpy as np

from neurowriter.models import tensorslice

# Minimice tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
tf.logging.set_verbosity(tf.logging.WARN)

def test_tensorslice_normal():
    """Tensor slicing is performed correctly for data > slices"""
    data = np.array([
        [[1,1,1], [1,1,1], [1,1,1]],
        [[2,2,2], [2,2,2], [2,2,2]],
        [[3,3,3], [3,3,3], [3,3,3]],
        [[4,4,4], [4,4,4], [4,4,4]],
        [[5,5,5], [5,5,5], [5,5,5]],
        [[6,6,6], [6,6,6], [6,6,6]]
    ])
    datatensor = tf.constant(data)
    
    tests = [
        (1, [data]),
        (2, [data[0:3], data[3:6]]),
        (3, [data[0:2], data[2:4], data[4:6]]),
        (6, [data[0:1], data[1:2], data[2:3],
             data[3:4], data[4:5], data[5:6]])
    ]
    
    # We need a tensor flow session to run the graph
    sess = tf.Session()
    
    for nslices, expected in tests:
        obtained = [sess.run(tensorslice(datatensor, i, nslices)) for i in range(nslices)]
        print("nslices:", nslices)
        print("Expected:", expected)
        print("Obtained:", obtained)
        assert(np.allclose(obtained, expected))
    
def test_tensorslice_small():
    """Tensor slicing is performed correctly for data < slices"""
    data = np.array([
        [[1,1,1], [1,1,1], [1,1,1]],
        [[2,2,2], [2,2,2], [2,2,2]],
        [[3,3,3], [3,3,3], [3,3,3]]
    ])
    datatensor = tf.constant(data)

    nulldata = np.zeros([0,3,3])
    tests = [
        (4, [nulldata, data[0:1], data[1:2], data[2:3]]),
        (5, [nulldata, data[0:1], nulldata, data[1:2], data[2:3]]),
        (10, [nulldata, nulldata, nulldata, data[0:1], nulldata,
              nulldata, data[1:2], nulldata, nulldata,data[2:3]])
    ]
    
    # We need a tensor flow session to run the graph
    sess = tf.Session()
    
    for nslices, expected in tests:
        obtained = [sess.run(tensorslice(datatensor, i, nslices)) for i in range(nslices)]
        print("nslices:", nslices)
        print("Expected:", expected)
        print("Obtained:", obtained)
        for x, y in zip(obtained, expected):
            assert(x.shape == y.shape)
            assert(np.allclose(x, y))
