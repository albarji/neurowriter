#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:37:46 2017

Tests for the corpus loading module.

@author: Álvaro Barbero Jiménez
"""

from neurowriter.corpus import (SingleTxtCorpus, MultiLineCorpus, CsvCorpus,
                                JsonCorpus)

DATAFOLDER = "neurowriter/tests/data/"

def test_SingleTxtCorpus():
    """Loading a single text corpus works as expected"""
    expected = ("This is is a single document corpus.\n"
                + "All the lines from this file belong to the same document.\n"
                + "And now for something different!\n"
                + "\n"
                + "PINEAPPLES!!!\n"
                )
    datafile = DATAFOLDER + "singledoc.txt"
    corpus = SingleTxtCorpus()
    corpus.load(datafile)
    
    # Test iterator
    for doc in corpus:
        print("Expected", expected)
        print("Obtained", doc)
        assert doc == expected
        
    # Test length
    assert len(corpus) == 1
        
    # Test direct access
    assert corpus[0] == expected

def test_MultiLineCorpus():
    """Loading a multiline text corpus works as expected"""
    expected = [
        "This is a multidocument.",
        "The file stores one document per line.",
        "So there are three documents here."
    ]
    datafile = DATAFOLDER + "multiline.txt"
    corpus =  MultiLineCorpus()
    corpus.load(datafile)
    
    # Test iterator
    for doc, exp in zip(corpus, expected):
        print("Expected", exp)
        print("Obtained", doc)
        assert doc == exp

    # Test length
    assert len(corpus) == 3

    # Test direct access
    for i in range(len(corpus)):
        assert corpus[i] == expected[i]

def test_CsvCorpus():
    """Loading a CSV corpus works as expected"""
    
    expected = [
        "Na Boca da Noite",
        "The Other Side of the Wind",
        "Prata Palomares"
    ]
    datafile = DATAFOLDER + "csvcorpus.csv"
    corpus =  CsvCorpus()
    corpus.load(datafile)
    
    # Test iterator
    for doc, exp in zip(corpus, expected):
        print("Expected", exp)
        print("Obtained", doc)
        assert doc == exp

    # Test length
    assert len(corpus) == 3

    # Test direct access
    for i in range(len(corpus)):
        assert corpus[i] == expected[i]
    
    # Test conditioners access    
    expected = [
        "['Drama']",
        "['Drama']",
        "['Thriller']"
    ]
    for cond, exp in zip(corpus.iterconditioners(), expected):
        print("Expected", exp)
        print("Obtained", cond["genres"])
        assert cond["genres"] == exp

def test_JsonCorpus():
    """Loading a JSON corpus works as expected"""
    
    expected = [
        "Na Boca da Noite",
        "The Other Side of the Wind",
        "Prata Palomares"
    ]
    datafile = DATAFOLDER + "jsoncorpus.json"
    corpus = JsonCorpus()
    corpus.load(datafile)
    
    # Test iterator
    for doc, exp in zip(corpus, expected):
        print("Expected", exp)
        print("Obtained", doc)
        assert doc == exp

    # Test length
    assert len(corpus) == 3

    # Test direct access
    for i in range(len(corpus)):
        assert corpus[i] == expected[i]
    
    # Test conditioners access    
    expected = [
        ['Drama'],
        ['Drama'],
        ['Thriller']
    ]
    for cond, exp in zip(corpus.iterconditioners(), expected):
        print("Expected", exp)
        print("Obtained", cond["genres"])
        assert cond["genres"] == exp
                   