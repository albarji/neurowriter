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


def train(dataset, outputdir, inputtokens=128, maxepochs=1000, valmask=None, patience=10, batchsize=256,
               learningrate=5e-5):
    """Trains a keras model with given parameters
    
    Arguments
        dataset: dataset to use for training
        outputdir: directory in which to save model
        inputtokens: number of input tokens the model will receive at a time
        maxepochs: maximum allowed training epochs for each model
        valmask: boolean mask marking patterns to use for validation.
            Input None to use all the data for training AND validation.
        patience: number of epochs without improvement for early stopping
        batchsize: number of patterns per training batch
        learningrate: learning rate to use in the optimizer
    """
    print(f"Training with inputtokens={inputtokens}, batchsize={batchsize}, learningrate={learningrate}")

    # Prepare GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with input parameters
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(dataset.labels))
    model.to(device)

    # Build model with input parameters
    #model = modelclass.create(inputtokens, encoder.nchars, *modelparams)
    # Prepare optimizer
    #optimizer = optimizerclass(lr=learningrate)
    # Compile model with optimizer
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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
    for epoch in range(maxepochs):
        epoch_iterator = tqdm(dataset.trainbatches(), desc="Iteration")
        for batch in epoch_iterator:
            # Forward pass through network
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            tr_loss += loss.item()
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1

        # TODO: use valgenerator to measure performance and make early stopping
        #results = evaluate(args, model, tokenizer)
        #for key, value in results.items():
        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
        logging_loss = tr_loss

        # Save model checkpoint
        check_dir = os.path.join(outputdir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(check_dir)

    return global_step, tr_loss / global_step
