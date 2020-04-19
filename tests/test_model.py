
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


@skip("temporarily disabled")
def test_gradient_accumulation():
    """Test gradient accumulation is equivalent to larger batches"""
    model = Model("distilbert-base-uncased")
    outputdir = mkdtemp() # TODO: needed?

    # Train model with batch size 4
    model.generate = lambda x: ""
    model.save = lambda x: None
    torch.manual_seed(0)
    model.fit(CORPUS, outputdir, maxepochs=1, batch_size=4, gradient_accumulation_steps=1)

    # Train model with batch size 2, gradient accumulation 2
    model2 = Model("distilbert-base-uncased")
    model2.generate = lambda x: ""
    model2.save = lambda x: None
    torch.manual_seed(0)
    model2.fit(CORPUS, outputdir, maxepochs=1, batch_size=2, gradient_accumulation_steps=2)

    # Weights should be equal
    for np1, np2 in zip(model.model.named_parameters(), model2.model.named_parameters()):
        name1, weights1 = np1
        name2, weights2 = np2
        assert torch.all(torch.eq(weights1, weights2))

    # Currently the models are only equal if they are trained exactly with the same parameters
    # When gradient accumulation is introduced, even the loss estimation prior to a model update shows differences


def test_generate():
    """Tests generating some text with a base model"""
    model = Model("distilbert-base-uncased")
    torch.manual_seed(0)
    generated = model.generate(seed="hello", maxlength=5, appendseed=True)

    assert len(model.tokenizer.tokenize(generated)) == 1+5  # Seed + generated
    assert model.tokenizer.tokenize(generated)[0] == "hello"
