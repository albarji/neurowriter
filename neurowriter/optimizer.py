#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for optimizing model desing.

@author: Álvaro Barbero Jiménez
"""

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from skopt import gbrt_minimize
from skopt.plots import plot_convergence
from keras import backend
import numpy as np


# Optimizer parameters
OPTPARAMS = {
    "batchsize": [8, 16, 32, 64, 128, 256],
    "optimizer": [Adam, RMSprop, Nadam],
    "learningrate": [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
}


def trainmodel(modelclass, inputtokens, encoder, corpus, maxepochs=1000, valmask=None, patience=20, batchsize=256,
               optimizerclass=Adam, learningrate=None, verbose=1, modelparams=[]):
    """Trains a keras model with given parameters
    
    Arguments
        modelclass: class defining the generative model
        inputtokens: number of input tokens the model will receive at a time
        encoder: encoder object used to transform from tokens to number
        corpus: corpus to use for training
        maxepochs: maximum allowed training epochs for each model
        valmask: boolean mask marking patterns to use for validation.
            Input None to use all the data for training AND validation.
        patience: number of epochs without improvement for early stopping
        batchsize: number of patterns per training batch
        optimizerclass: keras class of the optimizer to use
        learningrate: learning rate to use in the optimizer
        verbose: verbosity level (0 to 2)
        modelparams: list of parameters to be passed to the modelkind function
    """
    if verbose >= 1:
        print("Training with batchsize=%d, optimizer=%s, learningrate=%f, modelparams=%s" %
              (batchsize, str(optimizerclass), learningrate, str(modelparams)))
    # Build model with input parameters
    model = modelclass.create(inputtokens, encoder, *modelparams)
    # Prepare optimizer
    optimizer = optimizerclass(lr=learningrate)
    # Compile model with optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Prepare callbacks
    callbacks = [
        EarlyStopping(patience=patience)
    ]
    # Precompute some data size measurements
    ntokens = len(encoder.tokenizer.transform(corpus))
    npatterns = ntokens-inputtokens+1
    if valmask is not None:
        trainmask = [not x for x in valmask]
    else:
        valmask= [True]
        trainmask = [True]
    val_ratio = len([x for x in valmask if x]) / len(valmask)
    train_ratio = len([x for x in trainmask if x]) / len(trainmask)
    val_steps = np.ceil(npatterns * val_ratio / batchsize)
    train_steps = np.ceil(npatterns * train_ratio / batchsize)
    if train_steps == 0 or val_steps == 0:
        raise ValueError("Insufficient data for training in the current setting")
    # Prepare data generators
    traingenerator = encoder.patterngenerator(
        corpus, 
        tokensperpattern=inputtokens, 
        mask=trainmask, 
        batchsize=batchsize, 
        infinite=True
    )
    valgenerator = encoder.patterngenerator(
        corpus, 
        tokensperpattern=inputtokens, 
        mask=valmask,
        batchsize=batchsize, 
        infinite=True
    )
    # Train model
    train_history = model.fit_generator(
        traingenerator,
        steps_per_epoch=train_steps,
        validation_data=valgenerator,
        validation_steps=val_steps,
        epochs=maxepochs,
        verbose=2 if verbose == 2 else 0,
        callbacks=callbacks
    )
    # Trim model to make it more efficent for predictions
    model = modelclass.trim(model)
    # Return model and train history
    return model, train_history


def trainwrapper(modelclass, inputtokens, encoder, corpus, params, **kwargs):
    """Wrapper around the trainmodel function that unpacks model and optimizer parameters"""
    paramsdict = splitparams(params)
    return trainmodel(
        modelclass,
        inputtokens,
        encoder,
        corpus,
        batchsize=paramsdict["batchsize"],
        optimizerclass=paramsdict["optimizer"],
        learningrate=paramsdict["learningrate"],
        modelparams=paramsdict["modelparams"],
        **kwargs
    )


def createobjective(modelclass, inputtokens, encoder, corpus, verbose=1,
                    savemodel=None):
    """Creates an objective function for the hyperoptimizer
    
    Arguments
        modelclass: class defining the generative model
        inputtokens: number of input tokens the model will receive at a time
        encoder: encoder object used to transform from tokens to number
        corpus: corpus to use for training
        verbose: whether to print info on the evaluations of this objective
        savemodel: name of file where to save model after each training
        
    Returns an objective function that given model parameters traings a full
    network over a corpus, and returns the validation loss over such corpus.
    """
    def valloss(params):
        """Trains a keras model with given parameters and returns val loss"""
        try:
            model, train_history = trainwrapper(
                modelclass, 
                inputtokens, 
                encoder, 
                corpus,
                params=params,
                verbose=verbose
            )
            # Extract validation loss
            bestloss = min(train_history.history['val_loss'])
            if verbose:
                print("Params:", params, ", loss: ", bestloss)
            if savemodel is not None:
                model.save(savemodel)
            # Clear model and tensorflow session to free up space
            del model
            backend.clear_session()
        except Exception as e:
            print("Error while training", e)
            bestloss = 1000000
            if verbose:
                print("Params:", params, ", TRAINING FAILURE")            
        
        return bestloss
    
    # Return model train and validation loss producing function
    return valloss


def findbestparams(modelclass, inputtokens, encoder, corpus, 
                   n_calls=100, savemodel=None, verbose=1):
    """Find the best parameters for a given model architecture and param grid
    
    Returns
        - list with the best parameters found for the model
        - OptimizeResult object with info on the optimization procedure
    """
    fobj = createobjective(modelclass, inputtokens, encoder, corpus,
                           savemodel=savemodel, verbose=verbose)
    grid = addoptimizerparams(modelclass.paramgrid)
    optres = gbrt_minimize(fobj, grid, n_calls=n_calls,
                           random_state=0)
    bestparams = optres.x
    return bestparams, optres


def addoptimizerparams(paramgrid):
    """Adds optimizer parameters to a given model parameter grid"""
    newparamgrid = [OPTPARAMS[optparam] for optparam in sorted(OPTPARAMS.keys())]
    newparamgrid.extend(paramgrid)
    return newparamgrid


def splitparams(params):
    """Given a parameters vector, splits parameters by groups.

    Returns a dictionary with each optimizar parameter, and an additional entry
    'modelparams' that contains the vector of model parameters.
    """
    # Optimizer parameters
    paramsdict = {
        optparam: params[i]
        for i, optparam in enumerate(sorted(OPTPARAMS.keys()))
    }
    # Model parameters
    paramsdict["modelparams"] = params[len(OPTPARAMS):]
    return paramsdict


def hypertrain(modelclass, inputtokens, encoder, corpus, 
               n_calls=100, verbose=1, savemodel=None):
    """Performs hypertraining of a certain model architecture
    
    Returns 
        - The trained model with the best parameters
        - A train history object
    """
    # Hyperoptimization to find the best neural network parameters
    bestparams, optres = findbestparams(modelclass, inputtokens, encoder, 
                                        corpus, n_calls, savemodel, 
                                        verbose=verbose)
    if verbose >= 1:
        print("Best parameters are", bestparams)
        plot_convergence(optres)
    # Train again a new network with the best parameters
    return trainwrapper(modelclass, inputtokens, encoder, corpus, params=bestparams, verbose=verbose)
