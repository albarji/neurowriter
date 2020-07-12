#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dataset module

@author: Álvaro Barbero Jiménez
"""

from itertools import chain
import torch
from unittest.mock import MagicMock


from neurowriter.dataset import Dataset, TextGenerationCollator
from neurowriter.tokenizer import build_tokenizer, CLS, SEP, START, END


CORPUS = [
    "glory to mankind",
    "endless forms most beautiful",
    "abcdedg 1251957151"
]
TOKENIZER = None
EXPECTED_DATASET = None


def setup():
    # Pre-tokenize corpus for test
    global TOKENIZER, EXPECTED_DATASET
    TOKENIZER = build_tokenizer(pretrained_model='distilbert-base-uncased')
    encoded_corpus = [TOKENIZER.encode(f"{START} {txt} {END}", add_special_tokens=False) for txt in CORPUS]
    # Prepare expected dataset
    EXPECTED_DATASET = [
        (doc[0:i+1] + [TOKENIZER.mask_token_id], doc[i+1]) 
        for doc in encoded_corpus 
        for i in range(0, len(doc)-1)
    ]


def test_dataset_patterns():
    """Test a Dataset produces a correct set of patterns"""
    # Build datasets
    train_dataset, val_dataset = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=1)

    # Split into expected train and test patterns
    train_expected_dataset = EXPECTED_DATASET[::2]
    test_expected_dataset = EXPECTED_DATASET[1::2]

    # Test train patterns
    for real_pattern, expected_pattern in zip(train_dataset, train_expected_dataset):
        assert real_pattern == expected_pattern

    # Test validation patterns
    for real_pattern, expected_pattern in zip(val_dataset, test_expected_dataset):
        assert real_pattern == expected_pattern


def test_collator_batches():
    """Test a TextGenerationCollator produces a correct set of batches"""
    # Build dataset
    train_dataset, _ = Dataset.build_datasets(CORPUS, TOKENIZER, trainvalratio=0)
    collator = TextGenerationCollator(TOKENIZER, torch.device("cpu"))

    # Try different batch sizes for collating
    for batch_size in [1, 2, 5, 10]:
        # Build batch using collator
        patterns = [train_dataset[i] for i in range(batch_size)]
        batch = collator(patterns)
        # Batch contains expected fields for distilbert
        assert "input_ids" in batch
        assert "masked_lm_labels" in batch

        # Tensors shapes are as expected
        maxlen = max([len(EXPECTED_DATASET[i][0]) + 2 for i in range(batch_size)])  # [CLS] + pattern encoding + [CLS]
        for key in batch:
            assert batch[key].shape == (batch_size, maxlen)

        # All input_ids start with CLS
        assert all(batch["input_ids"][:, 0] == TOKENIZER.cls_token_id)
        # All input_ids end with SEP followed by zero or more PAD
        for pattern in batch["input_ids"]:
            sep_location = (pattern == TOKENIZER.sep_token_id).nonzero()
            assert len(sep_location) == 1
            assert all(pattern[sep_location.item()+1:] == TOKENIZER.pad_token_id)
        # Codified inputs in the batch are as expected
        for pattern, expected_pattern in zip(batch["input_ids"], EXPECTED_DATASET):
            expected_input = expected_pattern[0]
            assert all(pattern[1:1+len(expected_input)] == torch.tensor(expected_input))
        # Codified outputs in the batch are as expected
        for pattern, labels, expected_pattern in zip(batch["input_ids"], batch["masked_lm_labels"], EXPECTED_DATASET):
            expected_input, expected_output = expected_pattern
            expected_labels = [-100] * len(labels)
            mask_location = (pattern == TOKENIZER.mask_token_id).nonzero()
            assert len(mask_location) == 1
            expected_labels[mask_location.item()] = expected_output
            assert all(labels == torch.tensor(expected_labels))
