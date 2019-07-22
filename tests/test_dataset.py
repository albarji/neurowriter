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
        self.encoding = sorted(list(set(
            chain(*[tokens.split(" ") for tokens in CORPUS], ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[END]"])
        )))
    
    def encodetext(self, text):
        tokens = text.split(" ")
        return [self.encoding.index(token) for token in tokens]


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
