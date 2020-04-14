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
from neurowriter.tokenizer import build_tokenizer, START, END


CORPUS = [
    "Glory to mankind",
    "Endless forms most beautiful",
    "abcdedg 1251957151"
]


def test_patterns():
    """Test a Dataset produces a correct set of patterns"""
    tokenizer = build_tokenizer()
    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, tokenizer, tokensperpattern=4, trainvalratio=1)

    # Test train patterns
    expected = [
        (
            tokenizer.batch_encode_plus(f"{START}", return_tensors="pt"),
            tokenizer.batch_encode_plus("Glory", return_tensors="pt")
        ),
        (
            tokenizer.batch_encode_plus("to", return_tensors="pt"),
            tokenizer.batch_encode_plus("mankind", return_tensors="pt")
        ),
        (
            tokenizer.batch_encode_plus(f"{START}", return_tensors="pt"),
            tokenizer.batch_encode_plus("Endless", return_tensors="pt")
        ),
        (
            tokenizer.batch_encode_plus("forms", return_tensors="pt"),
            tokenizer.batch_encode_plus("most", return_tensors="pt")
        ),
        (
            tokenizer.batch_encode_plus("beautiful", return_tensors="pt"),
            tokenizer.batch_encode_plus(f"{END}", return_tensors="pt")
        ),
        (
            tokenizer.batch_encode_plus("abcdedg", return_tensors="pt"),
            tokenizer.batch_encode_plus("1251957151", return_tensors="pt")
        )
    ]

    loader = train_dataset.loader(batch_size=1)
    for real, exp in zip(loader, expected):
        print(f"Expected {exp}")
        print(f"Obtained {real}")
        assert all(torch.all(t.eq(e)) for t, e in zip(real, exp))

    # Test validation patterns
    valdata = list(dataset.valbatches())
    expected = [
        (
            torch.tensor([tokenizer.encodetext("[CLS] Glory [SEP]")]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("to")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] mankind [SEP]")]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("[END]")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] Endless [SEP]")]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("forms")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] most [SEP]")]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("beautiful")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[PAD] [CLS] [SEP]")]),
            torch.tensor([dataset._idx_to_label(tokenizer.encodetext("abcdedg")[0])])
        ),
        (
            torch.tensor([tokenizer.encodetext("[CLS] 1251957151 [SEP]")]),
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


def test_len():
    """Test the dataset returns correct lengths"""
    tokenizer = MockTokenizer()

    dataset = Dataset(CORPUS, tokenizer, tokensperpattern=1, batchsize=1, trainvalratio=3)
    assert dataset.lenpatterns == 12
    assert dataset.lentrainbatches == 9
    assert dataset.lenvalbatches == 3
    assert dataset.lentrainbatches == len(list(dataset.trainbatches()))
    assert dataset.lenvalbatches == len(list(dataset.valbatches()))

    dataset = Dataset(CORPUS, tokenizer, tokensperpattern=1, batchsize=2, trainvalratio=3)
    assert dataset.lenpatterns == 12
    assert dataset.lentrainbatches == 5
    assert dataset.lenvalbatches == 2
    assert dataset.lentrainbatches == len(list(dataset.trainbatches()))
    assert dataset.lenvalbatches == len(list(dataset.valbatches()))

    dataset = Dataset(CORPUS, tokenizer, tokensperpattern=1, batchsize=5, trainvalratio=1)
    assert dataset.lenpatterns == 12
    assert dataset.lentrainbatches == 2
    assert dataset.lenvalbatches == 2
    assert dataset.lentrainbatches == len(list(dataset.trainbatches()))
    assert dataset.lenvalbatches == len(list(dataset.valbatches()))
