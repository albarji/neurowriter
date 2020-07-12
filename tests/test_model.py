
"""
Tests for the model module

@author: Álvaro Barbero Jiménez
"""
from functools import partial
from tempfile import mkdtemp
from unittest import skip
from unittest.mock import Mock


from neurowriter.model import Model
import torch


CORPUS = [
    "glory to mankind",
    "endless forms most beautiful",
    "abcdedg 1251957151"
]


def test_fit():
    """Test fitting a model to a toy corpus"""
    model = Model("distilbert-base-uncased")
    torch.manual_seed(0)
    outputdir = mkdtemp()
    model.fit(CORPUS, outputdir, maxepochs=1, batch_size=4)


def test_generate():
    """Tests generating some text with a base model"""
    model = Model("distilbert-base-uncased")
    torch.manual_seed(0)
    generated = model.generate(seed="hello", maxlength=5, appendseed=True)

    assert len(model.tokenizer.tokenize(generated)) >= 1+5  # Seed + generated + possible untokenization errors
    assert model.tokenizer.tokenize(generated)[0] == "hello"
