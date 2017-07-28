#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:23:34 2017

Definitions of different text generation models.

@author: Álvaro Barbero Jiménez
"""

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Input, add, multiply, Dropout
from keras.layers.advanced_activations import ELU

def modelbyname(modelname):
    """Returns a model generating class by name"""
    models = {
        "dilatedconv" : DilatedConvModel
    }
    if modelname not in models:
        raise ValueError("Unknown model %s" % modelname)
    return models[modelname]

class DilatedConvModel():
    """Model based on dilated convolutions + pooling + dense layers"""
    
    paramgrid = [
        [2,3,4,5], # convlayers
        [4,8,16,32,64], # kernels
        (0.0, 1.0), # convdrop
        [0,1,2,3], # denselayers
        [16,32,64,128,256], # dense units
        (0.0, 1.0), # densedrop
        ['sgd', 'rmsprop', 'adam'], # optimizer
    ]
    
    def create(inputtokens, encoder, convlayers=5, kernels = 32,
               convdrop=0.1, denselayers=0, denseunits=64, densedrop=0.1,
               optimizer='adam'):
        kernel_size = 2
        pool_size = 2
        if convlayers < 1:
            raise ValueError("Number of layers must be at least 1")
            
        # First conv+pool layer
        model = Sequential()
        model.add(Conv1D(kernels, kernel_size, padding='causal', activation='relu', 
                         input_shape=(inputtokens, encoder.nchars)))
        model.add(Dropout(convdrop))
        model.add(MaxPooling1D(pool_size))
        # Additional dilated conv + pool layers
        for i in range(1, convlayers):
            model.add(Conv1D(kernels, kernel_size, padding='causal', 
                             dilation_rate=2**i, activation='relu'))
            model.add(Dropout(convdrop))
            model.add(MaxPooling1D(pool_size))
        # Flatten and dense layers
        model.add(Flatten())
        for i in range(denselayers):
            model.add(Dense(denseunits, activation='relu'))
            model.add(Dropout(densedrop))
        # Output layer
        model.add(Dense(encoder.nchars, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

##TODO: refine these models to make them workable

def convmodel():
    kernels = 64
    kernel_size = 2
    
    def adddilatedblock(model):
        model.add(Conv1D(kernels, kernel_size, padding='causal', input_shape=(inputtokens, encoder.nchars)))
        model.add(ELU())
        model.add(Conv1D(kernels, kernel_size, padding='causal', dilation_rate=2))
        model.add(ELU())
        model.add(Conv1D(kernels, kernel_size, padding='causal', dilation_rate=4))
        model.add(ELU())
        model.add(Conv1D(kernels, kernel_size, padding='causal', dilation_rate=8))
        model.add(ELU())
        model.add(Conv1D(kernels, kernel_size, padding='causal', dilation_rate=16))
        model.add(ELU())
        return model

    model = Sequential()
    model = adddilatedblock(model)
    #model = adddilatedblock(model)
    #model = adddilatedblock(model)
    model.add(Conv1D(1,1))
    model.add(ELU())
    model.add(Conv1D(1,1))
    model.add(Flatten())
    model.add(Dense(encoder.nchars, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def wavenet():
    kernels = 64
    kernel_size = 2
    wavenetblocks=1
    maxdilation = 64
    dropout = 0
    
    def gatedblock(dilation):
        """Creates a dilated convolution layer with Gated + ELU activations and skip connections"""
        def f(input_):
            # Dropout of inputs
            drop = Dropout(dropout)(input_)
            # Normal activation
            normal_out = Conv1D(kernels, kernel_size, padding='causal', dilation_rate = dilation
                               ,activation='tanh')(drop)
            
            # Gate
            gate_out = Conv1D(kernels, kernel_size, padding='causal', dilation_rate = dilation, 
                              activation='sigmoid')(drop)
            # Point-wise nonlinear · gate
            merged = multiply([normal_out, gate_out])
            # Activation after gate
            skip_out = ELU()(Conv1D(kernels, 1, padding='same')(merged))
            # Residual connections: allow the network input to skip the whole block if necessary
            out = add([skip_out, input_])
            return out, skip_out
        return f
    
    def wavenetblock(maxdilation):
        """Creates a whole wavenet block made by stacking gated block with exponentially increasing dilations"""
        def f(input_):
            dilation = 1
            flow = input_
            skip_connections = []
            # Increasing dilation rates
            while dilation < maxdilation:
                flow, skip = gatedblock(dilation)(flow)
                skip_connections.append(skip)
                dilation *= 2
            skip = add(skip_connections)
            return flow, skip
        return f
    
    input_ = Input(shape=(inputtokens, encoder.nchars))
    net = Conv1D(kernels, 1, padding='same')(input_)
    skip_connections = []
    for i in range(wavenetblocks):
        net, skip = wavenetblock(maxdilation)(net)
        skip_connections.append(skip)
    if wavenetblocks > 1:
        net = add(skip_connections)
    else:
        net = skip
    net = ELU()(net)
    net = Conv1D(kernels, 1)(net)
    net = ELU()(net)
    net = Conv1D(kernels, 1)(net)
    net = Flatten()(net)
    net = Dense(encoder.nchars, activation='softmax')(net)
    model = Model(inputs=input_, outputs=net)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model