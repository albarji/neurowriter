#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:10:25 2017

Tests for the general utilities module

@author: Álvaro Barbero Jiménez
"""

import numpy as np
from itertools import islice

from neurowriter.genutils import batchedgenerator, batchedpatternsgenerator
from neurowriter.genutils import infinitegenerator, maskedgenerator
        
def test_batchedgenerator():
    """Batched generator works as expected"""
    
    @batchedgenerator
    def genfun():
        for i in range(10):
            yield i
            
    tests = [
        (1, [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
        (2, [[0,1],[2,3],[4,5],[6,7],[8,9]]),
        (3, [[0,1,2],[3,4,5],[6,7,8],[9]]),   
        (4, [[0,1,2,3],[4,5,6,7],[8,9]]),
        (5, [[0,1,2,3,4],[5,6,7,8,9]]),
        (6, [[0,1,2,3,4,5],[6,7,8,9]]),
        (7, [[0,1,2,3,4,5,6],[7,8,9]]),
        (8, [[0,1,2,3,4,5,6,7],[8,9]]),
        (9, [[0,1,2,3,4,5,6,7,8],[9]]),
        (10, [[0,1,2,3,4,5,6,7,8,9]]),
        (11, [[0,1,2,3,4,5,6,7,8,9]])
    ]
    
    for batchsize, expected in tests:
        obtained = list(genfun(batchsize=batchsize))
        print("Batch size", batchsize)
        print("Expected", expected)
        print("Obtained", obtained)
        assert(expected == obtained) 
        
def test_batchedpatternsgenerator():
    """Batched patterns generator works as expected"""
    
    @batchedpatternsgenerator
    def genfun():
        for i in range(10):
            yield np.array([i]), np.array([i])
            
    tests = [
        (1, [(np.array([[0]]),np.array([[0]])),
             (np.array([[1]]),np.array([[1]])),
             (np.array([[2]]),np.array([[2]])),
             (np.array([[3]]),np.array([[3]])),
             (np.array([[4]]),np.array([[4]])),
             (np.array([[5]]),np.array([[5]])),
             (np.array([[6]]),np.array([[6]])),
             (np.array([[7]]),np.array([[7]])),
             (np.array([[8]]),np.array([[8]])),
             (np.array([[9]]),np.array([[9]]))]),    
        (2, [(np.array([[0],[1]]),np.array([[0],[1]])),
             (np.array([[2],[3]]),np.array([[2],[3]])),
             (np.array([[4],[5]]),np.array([[4],[5]])),
             (np.array([[6],[7]]),np.array([[6],[7]])),
             (np.array([[8],[9]]),np.array([[8],[9]]))]),
        (3, [(np.array([[0],[1],[2]]),np.array([[0],[1],[2]])),
             (np.array([[3],[4],[5]]),np.array([[3],[4],[5]])),
             (np.array([[6],[7],[8]]),np.array([[6],[7],[8]])),
             (np.array([[9]]),np.array([[9]]))]),
        (4, [(np.array([[0],[1],[2],[3]]),np.array([[0],[1],[2],[3]])),
             (np.array([[4],[5],[6],[7]]),np.array([[4],[5],[6],[7]])),
             (np.array([[8],[9]]),np.array([[8],[9]]))]),
        (5, [(np.array([[0],[1],[2],[3],[4]]),np.array([[0],[1],[2],[3],[4]])),
             (np.array([[5],[6],[7],[8],[9]]),np.array([[5],[6],[7],[8],[9]]))]),
        (6, [(np.array([[0],[1],[2],[3],[4],[5]]),
              np.array([[0],[1],[2],[3],[4],[5]])),
             (np.array([[6],[7],[8],[9]]), np.array([[6],[7],[8],[9]]))]),
        (7, [(np.array([[0],[1],[2],[3],[4],[5],[6]]),
              np.array([[0],[1],[2],[3],[4],[5],[6]])),
             (np.array([[7],[8],[9]]), np.array([[7],[8],[9]]))]),
        (8, [(np.array([[0],[1],[2],[3],[4],[5],[6],[7]]),
              np.array([[0],[1],[2],[3],[4],[5],[6],[7]])),
             (np.array([[8],[9]]), np.array([[8],[9]]))]),
        (9, [(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]]),
              np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]])),
             (np.array([[9]]), np.array([[9]]))]),
        (10, [(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
              np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]))]),
        (11, [(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
              np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]))])
    ]
    
    for batchsize, expected in tests:
        obtained = list(genfun(batchsize=batchsize))
        print("Batch size", batchsize)
        print("Expected", expected)
        print("Obtained", obtained)
        for b1, b2 in zip(expected, obtained):
            assert(np.allclose(b1,b2))

def test_infinitegenerator():
    """Infinite generator works as expected"""
    @infinitegenerator
    def genfun():
        for i in range(10):
            yield i
            
    tests = [
        (1, [0]),
        (3, [0,1,2]),
        (5, [0,1,2,3,4]),
        (10, [0,1,2,3,4,5,6,7,8,9]),
        (12, [0,1,2,3,4,5,6,7,8,9,
              0,1]),
        (15, [0,1,2,3,4,5,6,7,8,9,
              0,1,2,3,4]),
        (20, [0,1,2,3,4,5,6,7,8,9,
              0,1,2,3,4,5,6,7,8,9]),
        (23, [0,1,2,3,4,5,6,7,8,9,
              0,1,2,3,4,5,6,7,8,9,
              0,1,2])
    ]
    
    for nelems, expected in tests:
        obtained = list(islice(genfun(infinite=True), nelems))
        print("Elements taken", nelems)
        print("Expected", expected)
        print("Obtained", obtained)
        assert(expected == obtained)     

def test_maskedgenerator():
    """Masked generator works as expected"""
    
    @maskedgenerator
    def genfun():
        for i in range(10):
            yield i
    
    tests = [
        ([True], [0,1,2,3,4,5,6,7,8,9]),
        ([False], []),
        ([True, False], [0,2,4,6,8]),
        ([False, True], [1,3,5,7,9]),
        ([True, False, True], [0,2,3,5,6,8,9]),
        ([True, False, False, False], [0,4,8])
    ]
    
    for mask, expected in tests:
        obtained = list(genfun(mask=mask))
        print("Mask", mask)
        print("Expected", expected)
        print("Obtained", obtained)
        assert(expected == obtained)
        
def test_infbatchmaskgenerator():
    """Combination of infinite, batched and masked generator"""
    
    @infinitegenerator
    @batchedgenerator
    @maskedgenerator
    def genfun():
        for i in range(10):
            yield i
    
    tests = [
        (1, [True], 10, [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
        (1, [True, False], 10, [[0],[2],[4],[6],[8],[0],[2],[4],[6],[8]]),
        (2, [True], 10, [[0,1],[2,3],[4,5],[6,7],[8,9],
                         [0,1],[2,3],[4,5],[6,7],[8,9]]),
        (2, [False, True], 10, [[1,3],[5,7],[9],[1,3],[5,7],[9],
                                [1,3],[5,7],[9],[1,3]]),
        (3, [False, True, False], 10, [[1,4,7],[1,4,7],[1,4,7],[1,4,7],[1,4,7],
                                      [1,4,7],[1,4,7],[1,4,7],[1,4,7],[1,4,7]])
    ]
    
    for batchsize, mask, nelems, expected in tests:
        generator = genfun(infinite=True,batchsize=batchsize,mask=mask)
        obtained = list(islice(generator, nelems))
        print("Mask", mask)
        print("Batch size", batchsize)
        print("Elements taken", nelems)
        print("Expected", expected)
        print("Obtained", obtained)
        assert(expected == obtained)
    