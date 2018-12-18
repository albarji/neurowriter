#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:55:20 2017

Tests for the models creation module.

@author: Álvaro Barbero Jiménez
"""

import tensorflow as tf
import numpy as np
from keras import backend

from neurowriter.models import get_available_gpus, tensorslice
from neurowriter.models import CNNLSTMModel, LSTMModel, StackedLSTMModel, WavenetModel, DilatedConvModel


def test_getgpus():
    """The list of GPU devices can be recovered"""
    gpus = get_available_gpus()
    assert isinstance(gpus, list)


def test_tensorslice_normal():
    """Tensor slicing is performed correctly for data > slices"""
    data = np.array([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
        [[6, 6, 6], [6, 6, 6], [6, 6, 6]]
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
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    ])
    datatensor = tf.constant(data)

    nulldata = np.zeros([0, 3, 3])
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


def model_build_checks(modelclass, paramsets):
    """Performs a series on check on a model class

    Arguments
        modelclass: class of the model to test
        paramets: iterable of sets of parameters to try in constructor
    """
    for paramset in paramsets:
        model = modelclass.create(**paramset)
        assert hasattr(model, "compile")
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        assert hasattr(model, "fit_generator")
        assert hasattr(model, "summary")
        model.summary()
        del model
        backend.clear_session()


def test_train_dilatedconv():
    """Dilated convolution models can be built correctly"""
    paramsets = [
        {"inputtokens": 128, "vocabsize": 1000},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "densedrop": 0.1},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 1000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 1, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 32, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.1, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.5, "denselayers": 1,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.5, "denselayers": 2,
         "denseunits": 64, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.5, "denselayers": 2,
         "denseunits": 128, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.5, "denselayers": 2,
         "denseunits": 128, "embedding": 64},
        {"inputtokens": 256, "vocabsize": 2000, "convlayers": 2, "kernels": 64, "convdrop": 0.5, "denselayers": 0,
         "denseunits": 0, "embedding": 64}
    ]
    model_build_checks(DilatedConvModel, paramsets)


def test_train_wavenet():
    """Wavenet models can be built correctly"""
    paramsets = [
        {"inputtokens": 128, "vocabsize": 1000},
        {"inputtokens": 128, "vocabsize": 1000, "kernels": 64},
        {"inputtokens": 128, "vocabsize": 1000, "kernels": 64, "wavenetblocks": 1},
        {"inputtokens": 128, "vocabsize": 1000, "kernels": 64, "wavenetblocks": 1, "dropout": 0},
        {"inputtokens": 128, "vocabsize": 1000, "kernels": 64, "wavenetblocks": 1, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 1000, "kernels": 64, "wavenetblocks": 1, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "kernels": 64, "wavenetblocks": 1, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "kernels": 256, "wavenetblocks": 1, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "kernels": 256, "wavenetblocks": 3, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "kernels": 256, "wavenetblocks": 3, "dropout": 0.5, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "kernels": 256, "wavenetblocks": 3, "dropout": 0.5, "embedding": 512},
    ]
    model_build_checks(WavenetModel, paramsets)


def test_train_lstm():
    """LSTM models can be built correctly"""
    paramsets = [
        {"inputtokens": 128, "vocabsize": 1000},
        {"inputtokens": 128, "vocabsize": 1000, "units": 16},
        {"inputtokens": 128, "vocabsize": 1000, "units": 16, "dropout": 0},
        {"inputtokens": 128, "vocabsize": 1000, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 1000, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "units": 128, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "units": 128, "dropout": 0.5, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "units": 128, "dropout": 0.5, "embedding": 256},
    ]
    model_build_checks(LSTMModel, paramsets)


def test_train_stackedlstm():
    """StackedLSTM models can be built correctly"""
    paramsets = [
        {"inputtokens": 128, "vocabsize": 1000},
        {"inputtokens": 128, "vocabsize": 1000, "layers": 1},
        {"inputtokens": 128, "vocabsize": 1000, "layers": 1, "units": 16},
        {"inputtokens": 128, "vocabsize": 1000, "layers": 1, "units": 16, "dropout": 0},
        {"inputtokens": 128, "vocabsize": 1000, "layers": 1, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 1000, "layers": 1, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "layers": 1, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "layers": 3, "units": 16, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "layers": 3, "units": 128, "dropout": 0, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "layers": 3, "units": 128, "dropout": 0.5, "embedding": 32},
        {"inputtokens": 256, "vocabsize": 2000, "layers": 3, "units": 128, "dropout": 0.5, "embedding": 256},
    ]
    model_build_checks(StackedLSTMModel, paramsets)


def test_train_cnnlstm():
    """CNN-LSTM models can built correctly"""
    paramsets = [
        {"inputtokens": 128, "vocabsize": 1000},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 0, "lstmunits": 64, "lstmdropout": 0, "embedding": 256,
         "embdropout": 0},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 1, "kernels": 64, "kernelsize": 3,
         "convdropout": 0, "lstmunits": 64, "lstmdropout": 0, "embedding": 256, "embdropout": 0},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 2, "kernels": 256, "kernelsize": 5,
         "convdropout": 0.5, "lstmunits": 128, "lstmdropout": 0.1, "embedding": 512, "embdropout": 0.5},
        {"inputtokens": 128, "vocabsize": 1000, "convlayers": 3, "kernels": 1024, "kernelsize": 15,
         "convdropout": 0.9, "lstmunits": 512, "lstmdropout": 0.9, "embedding": 1024, "embdropout": 0.5},
    ]
    model_build_checks(CNNLSTMModel, paramsets)
