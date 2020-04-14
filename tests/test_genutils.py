#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:10:25 2017

Tests for the general utilities module

@author: Álvaro Barbero Jiménez
"""

import numpy as np
from itertools import islice

from neurowriter.genutils import maskedgenerator
        

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
