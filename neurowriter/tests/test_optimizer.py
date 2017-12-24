#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the optimizer module.

@author: Álvaro Barbero Jiménez
"""

from neurowriter.optimizer import chekpointappend, checkpointload, hypertrain
from neurowriter.models import SmallWavenet
from neurowriter.corpus import Corpus
from neurowriter.encoding import Encoder
from neurowriter.tokenizer import CharTokenizer
from tempfile import NamedTemporaryFile, mkdtemp
from shutil import copyfile, rmtree

DATAFOLDER = "neurowriter/tests/data/"


def test_checkpoint():
    """Hyperoptimizer checkpoint loading and saving works as expected"""
    params = (
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    )
    losses = (1.2341, 2.6871, 0.5152)

    with NamedTemporaryFile("r") as tempfile:
        tmpname = tempfile.name
        # Write parameters
        for paramlist, loss in zip(params, losses):
            chekpointappend(tmpname, paramlist, loss)
        # Load parameters
        x0, y0 = zip(*checkpointload(tmpname))
        print("Expected", params, losses)
        print("Loaded", x0, y0)

        assert(x0 == params)
        assert(y0 == losses)


def test_hypertrain_run():
    """A small hypertraining procedure can be run"""
    modelclass = SmallWavenet
    corpus = Corpus(["This is a very small corpus for testing the hypertrain procedure.", "Hope it works!!!"])
    encoder = Encoder(corpus, CharTokenizer)

    with NamedTemporaryFile("r") as tempfile:
        tempdir = mkdtemp()
        tmpname = tempfile.name
        model = hypertrain(modelclass, encoder, corpus, tempdir, n_calls=15, verbose=2, valmask=[False, True],
                           patience=1, maxepochs=10, checkpointfile=tmpname)
        rmtree(tempdir)
        assert model is not None


def test_hypertrain_loadcheckpoints():
    """A previously generated checkpoints file can be used to continue the hyperoptimization"""
    modelclass = SmallWavenet
    corpus = Corpus(["This is a very small corpus for testing the hypertrain procedure.", "Hope it works!!!"])
    encoder = Encoder(corpus, CharTokenizer)
    checkpointsfile = DATAFOLDER + "checkpoints"

    with NamedTemporaryFile("r") as tempfile:
        tempdir = mkdtemp()
        tmpname = tempfile.name
        copyfile(checkpointsfile, tmpname)
        model = hypertrain(modelclass, encoder, corpus, tempdir, n_calls=15, verbose=2, valmask=[False, True],
                           patience=1, maxepochs=10, checkpointfile=tmpname)
        rmtree(tempdir)
        assert model is not None
