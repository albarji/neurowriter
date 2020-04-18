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
from neurowriter.tokenizer import build_tokenizer, CLS, SEP, START, END


CORPUS = [
    "Glory to mankind",
    "Endless forms most beautiful",
    "abcdedg 1251957151"
]
TOKENIZER = None
TOKENIZED_CORPUS = None
MASK = None


def setup():
    # Pre-tokenize corpus for test
    global TOKENIZER, TOKENIZED_CORPUS, MASK
    TOKENIZER = build_tokenizer()
    MASK = TOKENIZER.mask_token
    TOKENIZED_CORPUS = [[START] + TOKENIZER.tokenize(txt, add_special_tokens=False) + [END] for txt in CORPUS]


def _assert_equal_batches(expected_batch, real_batch):
    """Test two batches of patterns are equal"""
    print(f"Expected {expected_batch}")
    print(f"Obtained {real_batch}")
    # Compare X
    for key in real_batch[0]:
        assert torch.all(expected_batch[0][key] == real_batch[0][key])
    # Compare y
    assert torch.all(expected_batch[1] == real_batch[1])


def test_patterns():
    """Test a Dataset produces a correct set of patterns"""
    # Build data loaders
    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=1)

    # Expected patterns for all corpus
    expected = [
        (
            TOKENIZER.batch_encode_plus([(tokenized_doc[:i] + [MASK], None)], return_tensors="pt"),
            torch.tensor([TOKENIZER.encode(tokenized_doc[i:i+1], add_special_tokens=False)])
        )
        for tokenized_doc in TOKENIZED_CORPUS
        for i in range(1, len(tokenized_doc))
    ]

    # Test train patterns
    loader = train_dataset.loader(batch_size=1)
    for real_batch, expected_batch in zip(loader, expected[::2]):
        _assert_equal_batches(expected_batch, real_batch)

    # Test validation patterns
    loader = val_dataset.loader(batch_size=1)
    for real_batch, expected_batch in zip(loader, expected[1::2]):
        _assert_equal_batches(expected_batch, real_batch)


def test_patterns_noval():
    """Test a Dataset produces a correct set of patterns when no validation ratio is provided"""
    # Build data loaders
    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=0)

    batch_sizes = [1, 2]

    for batch_size in batch_sizes:
        traindata = list(train_dataset.loader(batch_size=batch_size))
        valdata = list(val_dataset.loader(batch_size=batch_size))
        assert len(traindata) == len(valdata)

        for i in range(len(traindata)):
            _assert_equal_batches(traindata[i], valdata[i])


def test_len():
    """Test the dataset returns correct lengths for different batch sizes"""
    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=1)
    assert len(train_dataset) == 10
    assert len(val_dataset) == 10
    assert len(list(train_dataset.loader(batch_size=1))) == 10
    assert len(list(val_dataset.loader(batch_size=1))) == 10
    assert len(list(train_dataset.loader(batch_size=2))) == 5
    assert len(list(val_dataset.loader(batch_size=2))) == 5
    assert len(list(train_dataset.loader(batch_size=3))) == 4
    assert len(list(val_dataset.loader(batch_size=3))) == 4

    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=3)
    assert len(train_dataset) == 15
    assert len(val_dataset) == 5
    assert len(list(train_dataset.loader(batch_size=1))) == 15
    assert len(list(val_dataset.loader(batch_size=1))) == 5
    assert len(list(train_dataset.loader(batch_size=2))) == 8
    assert len(list(val_dataset.loader(batch_size=2))) == 3
    assert len(list(train_dataset.loader(batch_size=3))) == 5
    assert len(list(val_dataset.loader(batch_size=3))) == 2

    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=4)
    assert len(train_dataset) == 16
    assert len(val_dataset) == 4
    assert len(list(train_dataset.loader(batch_size=1))) == 16
    assert len(list(val_dataset.loader(batch_size=1))) == 4
    assert len(list(train_dataset.loader(batch_size=2))) == 8
    assert len(list(val_dataset.loader(batch_size=2))) == 2
    assert len(list(train_dataset.loader(batch_size=3))) == 6
    assert len(list(val_dataset.loader(batch_size=3))) == 2
