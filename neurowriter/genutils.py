
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
    
    It is expected that the generator returns instances of data patterns,
    as tuples of numpy arrays (X,y)
    """
    def modgenerator(*args, **kwargs):
        if "batchsize" in kwargs:
            batchsize = kwargs["batchsize"]
            del kwargs["batchsize"]
        else:
            batchsize = 1
        for batch in splitevery(generatorfunction(*args, **kwargs), batchsize):
            yield batch
    return modgenerator

def batchedpatternsgenerator(generatorfunction):
    """Decorator that assumes patterns (X,y) and stacks them in batches
    
    This can be thought of a specialized version of the batchedgenerator
    that assumes the base generator returns instances of data patterns,
    as tuples of numpy arrays (X,y). When grouping them in batches the
    numpy arrays are stacked so that each returned batch has a pattern 
    per row.
    
    A "batchsize" parameter is added to the generator, that if specified
    groups the data in batches of such size.
    """
    def modgenerator(*args, **kwargs):
        for batch in batchedgenerator(generatorfunction)(*args, **kwargs):
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
