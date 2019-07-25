#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dataset module

@author: Álvaro Barbero Jiménez
"""

from itertools import chain
import torch
from unittest.mock import MagicMock

from neurowriter.dataset import Dataset


CORPUS = [
    "Glory to mankind",
    "Endless forms most beautiful",
    "abcdedg 1251957151"
]

class MockTokenizer():
    """Simple tokenizer to use in tests"""
    def __init__(self):
        # List of tokens
        self.encoding = sorted(list(set(
            chain(*[tokens.split(" ") for tokens in CORPUS], ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[END]"])
        )))
        # Map from tokens to indices
        self.encoding = {
            token: idx
            for idx, token in enumerate(self.encoding)
        }
    
    def encodetext(self, text):
        tokens = text.split(" ")
        return [self.encoding[token] for token in tokens]

    def encode_bert(self, tokens, padding=0):
        x = [self.encoding["[PAD]"]] * padding
        x += [self.encoding["[CLS]"]] + tokens + [self.encoding["[SEP]"]]
        mask = [0] * padding + [1] * (len(tokens) + 2)
        types = [0] * len(x)
        return x, mask, types

    @property
    def vocab(self):
        return self.encoding

def test_patterns():
    """Test a Dataset produces a correct set of patterns"""
    tokenizer = MockTokenizer()
    dataset = Dataset(CORPUS, tokenizer, tokensperpattern=1, batchsize=1, trainvalratio=1)

    # Test train patterns
    traindata = list(dataset.trainbatches())
    expected = [
        (
            torch.tensor([tokenizer.encodetext("[PAD] [CLS] [SEP]")]),
            torch.tensor([[0, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("Glory")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] to [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("mankind")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[PAD] [CLS] [SEP]")]),
            torch.tensor([[0, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("Endless")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] forms [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("most")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] beautiful [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("[END]")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] abcdedg [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("1251957151")[0])])
        )
    ]

    for i in range(len(traindata)):
        print(f"Expected {expected[i]}")
        print(f"Obtained {traindata[i]}")
        assert all(torch.all(t.eq(e)) for t, e in zip(traindata[i], expected[i]))

    # Test validation patterns
    valdata = list(dataset.valbatches())
    expected = [
        (
            torch.tensor([tokenizer.encodetext("[CLS] Glory [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("to")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] mankind [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("[END]")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] Endless [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("forms")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] most [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("beautiful")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[PAD] [CLS] [SEP]")]),
            torch.tensor([[0, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("abcdedg")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] 1251957151 [SEP]")]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("[END]")[0])])
        )
    ]

    for i in range(len(valdata)):
        print(f"Expected {expected[i]}")
        print(f"Obtained {valdata[i]}")
        assert all(torch.all(t.eq(e)) for t, e in zip(valdata[i], expected[i]))


def test_patterns_noval():
    """Test a Dataset produces a correct set of patterns when no validation ratio is provided"""
    tokenizer = MockTokenizer()

    options = [
        {"tokensperpattern": 1, "batchsize": 1},
        {"tokensperpattern": 2, "batchsize": 1},
        {"tokensperpattern": 1, "batchsize": 2},
        {"tokensperpattern": 2, "batchsize": 2}
    ]

    for opt in options:
        dataset = Dataset(CORPUS, tokenizer, trainvalratio=0, **opt)

        traindata = list(dataset.trainbatches())
        valdata = list(dataset.valbatches())
        assert len(traindata) == len(valdata)

        for i in range(len(traindata)):
            print(f"Train data {traindata[i]}")
            print(f"Validation data {valdata[i]}")
            assert all(torch.all(t.eq(e)) for t, e in zip(traindata[i], valdata[i]))
