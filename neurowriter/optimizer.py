#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for optimizing model desing.

@author: Álvaro Barbero Jiménez
"""

import numpy as np
from keras.callbacks import EarlyStopping
from skopt import gbrt_minimize
from skopt.plots import plot_convergence

def trainmodel(modelkind, inputtokens, encoder, corpus, maxepochs = 1000, 
               val = 0.25, patience = 10, batchsize = 128, modelparams=[]):
    """Trains a keras model with given parameters
    
    Arguments
        modelkind: function creating the generative model
        inputtokens: number of input tokens the model will receive at a time
        encoder: encoder object used to transform from tokens to number
        corpus: corpus to use for training
        maxepochs: maximum allowed training epochs for each model
        val: size of the validation set
        patience: number of epochs without improvement for early stopping
        batchsize: number of patterns per training batch
        modelparams: list of parameters to be passed to the modelkind function
    """        
    # Build model with input parameters
    model = modelkind(inputtokens, encoder, *modelparams)
    # Prepare callbacks
    callbacks = [
        #ModelCheckpoint(filepath=modelname,save_best_only=True),
        EarlyStopping(patience=patience)
    ]
    # Prepare data generators
    cutpoint = int(np.ceil(len(corpus) * (1-0.25)))
    traingenerator = encoder.patterngenerator(
        corpus, 
        tokensperpattern=inputtokens, 
        end=cutpoint, 
        batchsize=batchsize, 
        infinite=True
    )
    valgenerator = encoder.patterngenerator(
        corpus, 
        tokensperpattern=inputtokens, 
        start=cutpoint,
        batchsize=batchsize, 
        infinite=True
    )
    # Train model
    train_history = model.fit_generator(
        traingenerator,
        steps_per_epoch=int((1-val)*(len(corpus)-inputtokens+1)/batchsize),
        validation_data=valgenerator,
        validation_steps=int(val*(len(corpus)-inputtokens+1)/batchsize),
        epochs=maxepochs,
        verbose=0,
        callbacks=callbacks
    )
    # Return model and train history
    return model, train_history

def createobjective(modelkind, inputtokens, encoder, corpus, verbose=True,
                    savemodel=None):
    """Creates an objective function for the hyperoptimizer
    
    Arguments
        modelkind: function creating the generative model
        inputtokens: number of input tokens the model will receive at a time
        encoder: encoder object used to transform from tokens to number
        corpus: corpus to use for training
        maxepochs: maximum allowed training epochs for each model
        patience: number of epochs without improvement for early stopping
        batchsize: number of patterns per training batch
        val: size of the validation set
        verbose: whether to print info on the evaluations of this objective
        savemodel: name of file where to save model after each training
        
    Returns an objective function that given model parameters traings a full
    network over a corpus, and returns the validation loss over such corpus.
    """
    def valloss(params):
        """Trains a keras model with given parameters and returns val loss"""
        model, train_history = trainmodel(
            modelkind, 
            inputtokens, 
            encoder, 
            corpus, 
            modelparams=params
        )
        # Return validation loss
        bestloss = min(train_history.history['val_loss'])
        if verbose:
            print("Params:", params, ", loss: ", bestloss)
        if savemodel is not None:
            model.save(savemodel)
        return bestloss
    
    # Return model train and validation loss producing function
    return valloss

def findbestparams(modelclass, inputtokens, encoder, corpus, 
                   n_calls=100, savemodel=None):
    """Find the best parameters for a given model architecture and param grid
    
    Returns
        - list with the best parameters found for the model
        - OptimizeResult object with info on the optimization procedure
    """
    fobj = createobjective(modelclass.create, inputtokens, encoder, corpus,
                           savemodel=savemodel)
    optres = gbrt_minimize(fobj, modelclass.paramgrid, n_calls=n_calls, 
                           random_state=0)
    bestparams = optres.x
    return bestparams, optres

def hypertrain(modelclass, inputtokens, encoder, corpus, 
               n_calls=100, verbose=True, savemodel=None):
    """Performs hypertraining of a certain model architecture
    
    Returns 
        - The trained model with the best parameters
        - A train history object
    """
    # Hyperoptimization to find the best neural network parameters
    bestparams, optres = findbestparams(modelclass, inputtokens, encoder, 
                                        corpus, n_calls, savemodel)
    if verbose:
        print("Best parameters are", bestparams)
        plot_convergence(optres);
    # Train again a new network with the best parameters
    return trainmodel(modelclass.create, inputtokens, encoder, corpus, 
                      modelparams=bestparams)
