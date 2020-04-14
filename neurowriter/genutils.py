
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


def maskedgenerator(generatorfunction):
    """Decorator that adds outputs masking to a generator.
    
    A "mask" parameter is added to the generator function, which expects
    a list of boolean variables. The mask is iterated in parallel to the
    generator, blocking from the output those items with a False value
    in the mask. If the mask is depleted it is re-cycled.
    """
    def mskgenerator(*args, **kwargs):
        if "mask" in kwargs:
            mask = kwargs["mask"]
            del kwargs["mask"]
        else:
            mask = [True]
        for i, item in enumerate(generatorfunction(*args, **kwargs)):
            if mask[i % len(mask)]:
                yield item
                
    return mskgenerator


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
