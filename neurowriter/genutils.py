
# coding: utf-8

# Utilities for data generation
#
# @author Álvaro Barbero Jiménez

from itertools import islice
import numpy as np

def splitevery(iterable, n):
    """Returns blocks of elements from an iterator"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))
    
def batchedgenerator(generatorfunction):
    """Decorator that makes a pattern generator produce patterns in batches

    A "batchsize" parameter is added to the generator, that if specified
    groups the data in batches of such size.    
    """
    def modgenerator(*args, **kwargs):
        if "batchsize" in kwargs:
            batchsize = kwargs["batchsize"]
            del kwargs["batchsize"]
        else:
            batchsize = 1
        for batch in splitevery(generatorfunction(*args, **kwargs), batchsize):
            Xb, yb = zip(*batch)
            yield np.stack(Xb), np.stack(yb)
    return modgenerator
    
def infinitegenerator(generatorfunction):
    """Decorator that makes a generator replay indefinitely
    
    An "infinite" parameter is added to the generator, that if set to True
    makes the generator loop indifenitely.    
    """
    def infgenerator(*args, **kwargs):
        if "infinite" in kwargs:
            infinite = kwargs["infinite"]
            del kwargs["infinite"]
        else:
            infinite = False
        if infinite == True:
            while True:
                for elem in generatorfunction(*args, **kwargs):
                    yield elem
        else:
            for elem in generatorfunction(*args, **kwargs):
                yield elem            
    return infgenerator

def addtensordimension(bidifunction):
    """Decorator for function returning 2D objects, adds singleton dimension"""
    def reshapedfunction(*args, **kwargs):
        output = bidifunction(*args, **kwargs)
        return np.reshape(output, output.shape.append(1))
    return reshapedfunction
        
    
def generatorshape(generator):
    """Consumes a generator and returns the shape of its full X, Y tensors"""
    Xlen = 0
    Ylen = 0
    for X, Y in generator:
        Xlen += len(X)
        Ylen += len(Y)
    return (Xlen,) + X.shape[1:], (Ylen,) + Y.shape[1:]
    
def generatorlengths(generator):
    """Consumes a generator and returns a list of lengths of its X, Y patterns"""
    return [(len(X), len(Y)) for X,Y in generator]
