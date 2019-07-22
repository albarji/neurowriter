#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for optimizing model desing.

@author: Álvaro Barbero Jiménez
"""

import os
import pickle as pkl
from pytorch_transformers import BertForSequenceClassification, AdamW, WarmupLinearSchedule
from tempfile import NamedTemporaryFile
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm


def train(dataset, outputdir, maxepochs=1000,patience=10, learningrate=5e-5, checkpointepochs=10):
    """Trains a keras model with given parameters
    
    Arguments
        dataset: dataset to use for training
        outputdir: directory in which to save model
        maxepochs: maximum allowed training epochs for each model
        patience: number of epochs without improvement for early stopping
        learningrate: learning rate to use in the optimizer
    """
    print(f"Training with learningrate={learningrate}")

    # Prepare GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with input parameters
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=dataset.lenlabels)
    model.to(device)

    print("Number of training batches:", len(dataset))
    if len(dataset) == 0:
        raise ValueError("Insufficient data for training in the current setting")

    # Prepare optimizer and schedule (linear warmup and decay)
    # Reference: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py#L80
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learningrate, eps=1e-8)
    t_total = maxepochs * len(dataset)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

    # Tensorboard
    tb_writer = SummaryWriter()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    ntrainbatches = len(list(dataset.trainbatches()))
    for epoch in tqdm(range(maxepochs), desc="Epoch", total=maxepochs):
        train_loss = 0
        epoch_iterator = tqdm(dataset.trainbatches(), desc="Batch", total=ntrainbatches)
        for batch in epoch_iterator:
            # Forward pass through network
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
            ouputs = model(**inputs)
            model_loss = ouputs[0]
            train_loss += model_loss.mean().item()

            # Backpropagation
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1

        # Measure loss in validation set
        eval_loss = eval(model, dataset)

        lr = scheduler.get_lr()[0]
        tb_writer.add_scalar('lr', lr, global_step)
        print(f"lr={lr}")
        train_loss = train_loss / ntrainbatches
        print(f"train_loss={train_loss}")
        tb_writer.add_scalar('train_loss', train_loss, global_step)
        print(f"eval_loss={eval_loss}")
        tb_writer.add_scalar('eval_loss', eval_loss, global_step)
        logging_loss = tr_loss


        # Save model checkpoint
        if epoch % checkpointepochs == 0:
            check_dir = os.path.join(outputdir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(check_dir)

    return global_step, tr_loss / global_step


def eval(model, dataset):
    """Evaluates the performance of a model in a given dataset. The validation part of the dataset is used"""
    # Prepare GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation all data batches
    eval_loss = 0.0
    nb_eval_steps = 0
    for batch in tqdm(dataset.valbatches(), desc="Evaluation batch"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss = outputs[0]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss
