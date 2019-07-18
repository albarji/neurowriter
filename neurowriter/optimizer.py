#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:26:32 2017

Module for optimizing model desing.

@author: Álvaro Barbero Jiménez
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.models import load_model
from skopt import gbrt_minimize
from skopt.plots import plot_convergence
from keras import backend
from tempfile import NamedTemporaryFile
import pickle as pkl
from pytorch_transformers import BertForSequenceClassification, AdamW, WarmupLinearSchedule

# Loss to account for failed hyperoptmimization trials
FAILEDTRIALLOSS = 1000
# Number of random trials at the start of the hyperoptimization
RANDOMTRIALS = 10

# Optimizer parameters
OPTPARAMS = {
    "batchsize": [32, 64, 128, 256, 512],
    "learningrate": [2e-3, 1e-3, 5e-4, 2e-4, 1e-4],
    "inputtokens": [4, 8, 16, 32, 64, 128],
}


def trainmodel(modelclass, inputtokens, corpus, maxepochs=1000, valmask=None, patience=10, batchsize=256,
               learningrate=None, verbose=1, modelparams=[]):
    """Trains a keras model with given parameters
    
    Arguments
        modelclass: class defining the generative model
        inputtokens: number of input tokens the model will receive at a time
        corpus: corpus to use for training
        maxepochs: maximum allowed training epochs for each model
        valmask: boolean mask marking patterns to use for validation.
            Input None to use all the data for training AND validation.
        patience: number of epochs without improvement for early stopping
        batchsize: number of patterns per training batch
        learningrate: learning rate to use in the optimizer
        verbose: verbosity level (0 to 2)
        modelparams: list of parameters to be passed to the modelkind function
    """
    if verbose >= 1:
        print("Training with inputtokens=%d, batchsize=%d, optimizer=%s, learningrate=%f, modelparams=%s" %
              (inputtokens, batchsize, str(optimizerclass), learningrate, str(modelparams)))

    # Build model with input parameters
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

    # Build model with input parameters
    #model = modelclass.create(inputtokens, encoder.nchars, *modelparams)
    # Prepare optimizer
    #optimizer = optimizerclass(lr=learningrate)
    # Compile model with optimizer
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Prepare masks
    if valmask is not None:
        trainmask = [not x for x in valmask]
    else:
        valmask = [True]
        trainmask = [True]

    # Precompute some data size measurements
    ntrainbatches = len(list(encoder.patterngenerator(corpus, tokensperpattern=inputtokens, mask=trainmask,
                                                      batchsize=batchsize)))
    nvalbatches = len(list(encoder.patterngenerator(corpus, tokensperpattern=inputtokens, mask=valmask,
                                                    batchsize=batchsize)))
    if verbose >= 2:
        print("Number of training batches:", ntrainbatches)
        print("Number of validation batches:", nvalbatches)
    if ntrainbatches == 0 or nvalbatches == 0:
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

    # Prepare optimizer and schedule (linear warmup and decay)
    # Reference: https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py#L80
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Model training
    with NamedTemporaryFile() as modelfile:
        # Prepare callbacks
        callbacks = [
            EarlyStopping(patience=patience),
            ModelCheckpoint(modelfile.name, save_best_only=True, save_weights_only=True)
        ]
        # Train model
        train_history = model.fit_generator(
            traingenerator,
            steps_per_epoch=ntrainbatches,
            validation_data=valgenerator,
            validation_steps=nvalbatches,
            epochs=maxepochs,
            verbose=2 if verbose == 2 else 0,
            callbacks=callbacks
        )
        # Recover best model seen during training
        model.load_weights(modelfile.name)

    # Trim model to make it more efficent for predictions
    model = modelclass.trim(model)
    # Return model and train history
    return model, train_history


def trainwrapper(modelclass, encoder, corpus, params, **kwargs):
    """Wrapper around the trainmodel function that unpacks model and optimizer parameters"""
    paramsdict = splitparams(params)
    return trainmodel(
        modelclass,
        paramsdict["inputtokens"],
        encoder,
        corpus,
        batchsize=paramsdict["batchsize"],
        optimizerclass=optimizerbyname(paramsdict["optimizer"]),
        learningrate=paramsdict["learningrate"],
        modelparams=paramsdict["modelparams"],
        **kwargs
    )


def createobjective(modelclass, encoder, corpus, verbose=1, valmask=None, patience=20, maxepochs=1000,
                    modelsfolder=None, checkpointfile=None):
    """Creates an objective function for the hyperoptimizer
    
    Arguments
        modelclass: class defining the generative model
        encoder: encoder object used to transform from tokens to number
        corpus: corpus to use for training
        verbose: whether to print info on the evaluations of this objective
        patience: number of epochs to wait without validation improvement
        maxepochs: maximum allowed training epochs for each model
        modelsfolder: folder in which to save all tested models
        checkpointfile: file in which to save hyperoptimizer trials
        
    Returns an objective function that given model parameters traings a full
    network over a corpus, and returns the validation loss over such corpus.
    """
    def valloss(params):
        """Trains a keras model with given parameters and returns val loss"""
        try:
            model, train_history = trainwrapper(
                modelclass,
                encoder, 
                corpus,
                params=params,
                verbose=verbose,
                valmask=valmask,
                patience=patience,
                maxepochs=maxepochs
            )
            # Extract validation loss
            bestloss = min(train_history.history['val_loss'])
            if verbose:
                print("Params:", params, ", loss: ", bestloss)
            # Save model with loss value
            if modelsfolder is not None:
                model.save(modelsfolder + "/" + loss2modelname(bestloss))
            # Clear model and tensorflow session to free up space
            del model
            backend.clear_session()
        except Exception as e:
            print("Error while training", e)
            bestloss = FAILEDTRIALLOSS
            if verbose:
                print("Params:", params, ", TRAINING FAILURE")
        # Save hyperoptimizer checkpoint
        if checkpointfile is not None:
            chekpointappend(checkpointfile, params, bestloss)
        
        return bestloss
    
    # Return model train and validation loss producing function
    return valloss


def loss2modelname(loss):
    """Given a loss value, returns a string with a corresponding name for model saving"""
    return "model_loss%f" % loss


def findbestparams(modelclass, encoder, corpus, modelsfolder, n_calls=100, verbose=1, valmask=None, patience=20,
                   maxepochs=1000, checkpointfile=None):
    """Find the best parameters for a given model architecture and param grid
    
    Returns
        - list with the best parameters found for the model
        - validation loss for such model
        - best model found
        - OptimizeResult object with info on the optimization procedure
    """
    if verbose >= 1:
        print("Will save all tested models under %s" % modelsfolder)
    # Load checkpoint (if any)
    x0 = None
    y0 = None
    previoustrials = 0
    if checkpointfile:
        prevtrials = checkpointload(checkpointfile)
        if len(prevtrials) > 0:
            params, losses = zip(*prevtrials)
            x0 = list(params)
            y0 = list(losses)
            previoustrials = len(x0)
    if verbose >= 1:
        print("Checkpoint x0", x0)
        print("Checkpoint y0", y0)
    # Prepare and run optimizer
    fobj = createobjective(modelclass, encoder, corpus, verbose=verbose, valmask=valmask, patience=patience,
                           maxepochs=maxepochs, modelsfolder=modelsfolder, checkpointfile=checkpointfile)
    grid = addoptimizerparams(modelclass.paramgrid)
    trials = max(n_calls - previoustrials, 1)  # Run remaining trials, minimum 10
    randomtrials = max(RANDOMTRIALS - previoustrials, 0)
    optres = gbrt_minimize(fobj, grid, n_calls=trials, n_random_starts=randomtrials, random_state=0, x0=x0, y0=y0)
    # Recover best parameters, best loss, best model
    bestparams = optres.x
    bestloss = optres.fun
    bestmodel = load_model(modelsfolder + "/" + loss2modelname(bestloss))

    return bestparams, bestloss, bestmodel, optres


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


def chekpointappend(checkpointfile, params, loss):
    """Appends an hyperoptimization result to a checkpoint file

    If the checkpoint file does not exist, creates it anew.
    """
    previous = checkpointload(checkpointfile)
    previous.append((params, loss))
    with open(checkpointfile, "wb") as f:
        pkl.dump(previous, f)


def checkpointload(checkpointfile):
    """Loads an hyperoptimizer checkpoint from file

    Returns a list of tuples (params, loss) referring to previous hyperoptimization trials
    """
    try:
        with open(checkpointfile, "rb") as f:
            return pkl.load(f)
    except (FileNotFoundError, EOFError):
        return []


def hypertrain(modelclass, encoder, corpus, modelsfolder, n_calls=100, verbose=1, valmask=None, patience=20,
               maxepochs=1000, checkpointfile=None):
    """Performs hypertraining of a certain model architecture

    Arguments
        - modelsclass: class of the model architecture to hyperoptimize
        - encoder: pretrained encoder to use
        - corpus: corpus to use in the hyperoptimization
        - modelsfolder: folder in which to save all models generated during the hyperoptimization
        - n_calls: number of hyperoptimization trials
        - verbose: verbosity level
        - valmask: binary mask for splitting the corpus in training / validation data
        - patience: maximum number of training epochs without validation improvement before stopping a trial
        - maxepochs: maximum allowed training epochs for each model
        - checkpointfile: name of the file to use for checkpointing the hyperoptimization progress. If the file
            already exists, its contents are used to warm-start the hyperoptimization

    Returns 
        - The trained model with the best parameters
    """
    # Hyperoptimization to find the best neural network parameters
    bestparams, bestloss, bestmodel, optres = findbestparams(
        modelclass,
        encoder,
        corpus,
        modelsfolder,
        n_calls,
        verbose=verbose,
        valmask=valmask,
        patience=patience,
        maxepochs=maxepochs,
        checkpointfile=checkpointfile
    )
    if verbose >= 1:
        print("Best parameters are", bestparams)
        print("Best validation loss is", bestloss)
        if verbose >= 3:
            plot_convergence(optres)
    return bestmodel
