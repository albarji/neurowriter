#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:23:34 2017

Definitions of different text generation models.

Models are defined in a way that when multiple GPUs are present in the
host, model parallelization is performed for faster training.

@author: Álvaro Barbero Jiménez
"""

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Input, Dropout, Activation, GlobalMaxPool1D, CuDNNLSTM
from keras.layers import add, multiply, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from tensorflow.python.client import device_lib
import re


def get_available_gpus():
    """Returns a list of the GPU devices found in the host
    
    Reference: 
        - https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_parallel(model, gpu_count):
    """Makes a keras model data-parallel on a set of gpus
    
    Original code extracted from
        https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
    
    Modifications by Álvaro Barbero.
    """
    if gpu_count <= 1:
        raise ValueError("At least 2 GPUs are required to make the model parallel")

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(tensorslice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        if gpu_count > 1:
            merged = []
            for outputs in outputs_all:
                merged.append(concatenate(outputs, axis=0))
        else:
            merged = outputs
            
        return Model(inputs=model.inputs, outputs=merged)


def tensorslice(data, idx, parts):
    """Slices a tensor of data into several parts, as equal as possible.
    
    Inputs:
        data: input data flow
        idx: index of the slice to extract
        parts: number of pieces in which to partition the data
       
    If the number of data patterns is smaller than parts, some slices will
    be empty.
    """
    shape = tf.shape(data)
    startidx = shape[:1] * idx // parts
    endidx = shape[:1] * (idx+1) // parts
    start = tf.concat([startidx, shape[1:]*0], axis=0)
    size = tf.concat([endidx-startidx, shape[1:]], axis=0)
    return tf.slice(data, start, size)


def getcoremodel(model):
    """Removes data-parallel scaffolding, for efficient prediction"""
    # Find the layer containing the internal model and return it
    # This is not a good way to do it, but I can't find a way to name
    # a submodel explicitly to recover it later
    for layer in model.layers:
        if re.match("model_", layer.name):
            return layer
    # Not found
    raise ValueError("Core model not found")


class ModelMixin:
    """Abstract class defining a text generation model"""

    # List of hyperparameter ranges for this model
    paramgrid = []

    @staticmethod
    def trim(model):
        """Removes parts of the model required for training but not for generation"""
        return model


class ParallelGpuModel(ModelMixin):
    """Abstract class defining a GPU parallelized model"""

    @staticmethod
    def trim(model):
        """Removes data-parallel scaffolding, for efficient prediction"""
        if len(get_available_gpus()) > 1:
            return getcoremodel(model)
        else:
            return model


class DilatedConvModel(ModelMixin):
    """Model based on dilated convolutions + pooling + dense layers"""
    
    paramgrid = [
        [2, 3, 4, 5],  # convlayers
        [4, 8, 16, 32, 64],  # kernels
        (0.0, 1.0),  # convdrop
        [0, 1, 2, 3],  # denselayers
        [16, 32, 64, 128, 256],  # dense units
        (0.0, 1.0),  # densedrop
        [32, 64, 128, 256, 512],  # size of the embedding
    ]

    @staticmethod
    def create(inputtokens, vocabsize, convlayers=5, kernels=32,
               convdrop=0.1, denselayers=0, denseunits=64, densedrop=0.1,
               embedding=32):
        kernel_size = 2
        pool_size = 2
        if convlayers < 1:
            raise ValueError("Number of layers must be at least 1")
            
        model = Sequential()        
        # Embedding layer
        model.add(Embedding(input_dim=vocabsize, output_dim=embedding,
                            input_length=inputtokens))
        # First conv+pool layer        
        model.add(Conv1D(kernels, kernel_size, padding='causal', 
                         activation='relu'))
        model.add(Dropout(convdrop))
        model.add(MaxPooling1D(pool_size))
        # Additional dilated conv + pool layers (if possible)
        for i in range(1, convlayers):
            try:
                model.add(Conv1D(kernels, kernel_size, padding='causal', 
                                 dilation_rate=2**i, activation='relu'))
                model.add(Dropout(convdrop))
                model.add(MaxPooling1D(pool_size))
            except:
                print("Warning: not possible to add %i-th layer, moving to output" % i)
                break
                
        # Flatten and dense layers
        model.add(Flatten())
        for i in range(denselayers):
            model.add(Dense(denseunits, activation='relu'))
            model.add(Dropout(densedrop))
        # Output layer
        model.add(Dense(vocabsize, activation='softmax'))
        return model


def gatedblock(dilation, dropout, kernels, kernel_size):
    """Keras compatible Dilated convolution layer

    Includes Gated activation, skip connections, batch normalization and dropout
    """

    def f(input_):
        norm = BatchNormalization()(input_)
        # Dropout of inputs
        drop = Dropout(dropout)(norm)
        # Normal activation
        normal_out = Conv1D(kernels, kernel_size, dilation_rate=dilation, activation='tanh', padding='same')(drop)
        # Gate
        gate_out = Conv1D(kernels, kernel_size, dilation_rate=dilation, activation='sigmoid', padding='same')(drop)
        # Point-wise nonlinear · gate
        merged = multiply([normal_out, gate_out])
        # Activation after gate
        skip_out = Conv1D(kernels, 1, activation='tanh')(merged)
        # Residual connections: allow the network input to skip the
        # whole block if necessary
        out = add([skip_out, input_])
        return out, skip_out

    return f


def wavenetblock(maxdilation, dropout, kernels, kernel_size):
    """Keras compatible Wavenet layer

    A Wavenet layer is made of a stack of gated blocks with exponentially increasing dilations
    """
    def f(input_):
        dilation = 1
        flow = input_
        skip_connections = []
        # Increasing dilation rates
        while dilation < maxdilation:
            flow, skip = gatedblock(dilation, dropout, kernels, kernel_size)(flow)
            skip_connections.append(skip)
            dilation *= 2
        skip = add(skip_connections)
        return flow, skip
    return f


class WavenetModel(ParallelGpuModel):
    """Implementation of Wavenet model
    
    The model is made of a series of blocks, each one made up of 
    exponentially increasing dilated convolutions, until the whole input
    sequence is covered. Residual connections are also included to speed
    up training
    
    As an addition to the original formulation, ReLU activations have
    been replaced by SELU units.
    
    This implementation is based on those provided by
        - https://github.com/basveeling/wavenet
        - https://github.com/usernaamee/keras-wavenet
        
    The original wavenet paper is available at
        - https://deepmind.com/blog/wavenet-generative-model-raw-audio/
        - https://arxiv.org/pdf/1609.03499.pdf
    """
    
    paramgrid = [
        [32, 64, 128, 256],  # kernels
        [1, 2, 3, 4, 5],  # wavenetblocks
        (0.0, 1.0),  # dropout
        [32, 64, 128, 256, 512]  # size of the embedding
    ]

    @staticmethod
    def create(inputtokens, vocabsize, kernels=64, wavenetblocks=1, dropout=0, embedding=32):
        kernel_size = 7
        maxdilation = inputtokens
        
        input_ = Input(shape=(inputtokens,), dtype='int32')
        # Embedding layer
        net = Embedding(input_dim=vocabsize, output_dim=embedding, input_length=inputtokens)(input_)
        net = Dropout(dropout)(net)
        # Wavenet starts!
        net = BatchNormalization()(net)
        net = Conv1D(kernels, 1, activation='tanh')(net)
        skip_connections = []
        for i in range(wavenetblocks):
            net, skip = wavenetblock(maxdilation, dropout, kernels, kernel_size)(net)
            skip_connections.append(skip)
        if wavenetblocks > 1:
            net = add(skip_connections)
        else:
            net = skip
        net = Conv1D(kernels, 1, activation='tanh')(net)
        net = Conv1D(kernels, 1)(net)
        net = Flatten()(net)
        net = Dense(vocabsize, activation='softmax')(net)
        model = Model(inputs=input_, outputs=net)
        
        # Make data-parallel
        ngpus = len(get_available_gpus())
        if ngpus > 1:
            model = make_parallel(model, ngpus)

        return model


class SmallWavenet(WavenetModel):
    """Implementation of very small Wavenet model for testing purposes"""
    paramgrid = [
        [4, 6, 8],  # kernels
        [1, 2],  # wavenetblocks
        (0.0, 1.0),  # dropout
        [4, 6, 8]  # size of the embedding
    ]


class StackedLSTMModel(ParallelGpuModel):
    """Implementation of stacked Long-Short Term Memory model
    
    Main reference is Andrej Karpathy post on text generation with LSTMs:
        - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    
    This implementation also includes an Embedding layer, a bidirectional
    LSTM as the first LSTM layer in the network, and residual connections
    for all intermediate LSTM layers.
    """
    
    paramgrid = [
        [1, 2, 3, 4, 5],  # layers
        [16, 32, 64, 128, 256, 512, 1024],  # units
        (0.0, 1.0),  # dropout
        [32, 64, 128, 256, 512]  # size of the embedding
    ]

    @staticmethod
    def create(inputtokens, vocabsize, layers=1, units=16, dropout=0, embedding=32):
        
        input_ = Input(shape=(inputtokens,), dtype='int32')
        
        # Embedding layer
        net = Embedding(input_dim=vocabsize, output_dim=embedding, input_length=inputtokens)(input_)
        net = Dropout(dropout)(net)
            
        # Bidirectional LSTM layer
        net = BatchNormalization()(net)
        net = Bidirectional(CuDNNLSTM(units, return_sequences=(layers > 1)))(net)
        net = Dropout(dropout)(net)
            
        # Rest of LSTM layers with residual connections (if any)
        for i in range(1, layers):
            if i < layers-1:
                block = BatchNormalization()(net)
                block = CuDNNLSTM(2*units, return_sequences=True)(block)
                block = Dropout(dropout)(block)
                net = add([block, net])
            else:
                net = BatchNormalization()(net)
                net = CuDNNLSTM(2*units)(net)
                net = Dropout(dropout)(net)
                    
        # Output layer
        net = Dense(vocabsize, activation='softmax')(net)
        model = Model(inputs=input_, outputs=net)
        
        # Make data-parallel
        ngpus = len(get_available_gpus())
        if ngpus > 1:
            model = make_parallel(model, ngpus)

        return model


class LSTMModel(ModelMixin):
    """Implementation of simple one layer bidirectional Long-Short Term Memory model
    
    Main reference is Andrej Karpathy post on text generation with LSTMs:
        - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    
    This implementation also includes an Embedding layer.
    """
    
    paramgrid = [
        [16, 32, 64, 128, 256, 512, 1024],  # units
        (0.0, 1.0),  # dropout
        [32, 64, 128, 256, 512]  # size of the embedding
    ]

    @staticmethod
    def create(inputtokens, vocabsize, units=16, dropout=0, embedding=32):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, output_dim=embedding, input_length=inputtokens)(input_)
        net = Dropout(dropout)(net)

        # Bidirectional LSTM layer
        net = BatchNormalization()(net)
        net = Bidirectional(CuDNNLSTM(units))(net)
        net = Dropout(dropout)(net)

        # Output layer
        net = Dense(vocabsize, activation='softmax')(net)
        model = Model(inputs=input_, outputs=net)

        # Make data-parallel
        ngpus = len(get_available_gpus())
        if ngpus > 1:
            model = make_parallel(model, ngpus)

        return model


class SmallLSTMModel(LSTMModel):
    """Implementation of very small Long-Short Term Memory model for testing purposes"""

    paramgrid = [
        [4, 6, 8],  # units
        (0.0, 1.0),  # dropout
        [4, 6, 8]  # size of the embedding
    ]


class CNNLSTMModel(ModelMixin):
    """Stack of Convolutional Layers followed by a Long-Short Term Memory model

    This model is loosely inspired in the encoder model used in the Tacotron 2 network:
        - https://research.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html
    """

    paramgrid = [
        [0, 1, 2, 3],  # convolutional layers
        [4, 8, 16, 32, 64, 128, 256, 512],  # convolutional kernels
        [3, 5, 7, 9],  # kernel size
        (0.0, 1.0),  # convolutional dropout
        [16, 32, 64, 128, 256],  # LSTM units
        (0.0, 1.0),  # LSTM dropout
        [32, 64, 128, 256, 512],  # size of the embedding
        (0.0, 1.0),  # Embedding dropout
    ]

    @staticmethod
    def create(inputtokens, vocabsize, convlayers=3, kernels=512, kernelsize=5, convdropout=0.5, lstmunits=256,
               lstmdropout=0.1, embedding=512, embdropout=0.5):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, output_dim=embedding, input_length=inputtokens)(input_)
        net = Dropout(embdropout)(net)

        # Convolutional layers (if any)
        for layer in range(convlayers):
            net = Conv1D(kernels, kernelsize, padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Dropout(convdropout)(net)

        # Bidirectional LSTM layer
        net = Bidirectional(CuDNNLSTM(lstmunits))(net)
        net = Dropout(lstmdropout)(net)

        # Output layer
        net = Dense(vocabsize, activation='softmax')(net)
        model = Model(inputs=input_, outputs=net)

        # Make data-parallel
        ngpus = len(get_available_gpus())
        if ngpus > 1:
            model = make_parallel(model, ngpus)

        return model


class PerceptronModel(ModelMixin):
    """Toy model that only uses embedding + dense layer"""

    paramgrid = [
        [2, 4, 8],  # dense units
        (0.0, 1.0),  # densedrop
        [16, 32, 64],  # size of the embedding
    ]

    @staticmethod
    def create(inputtokens, vocabsize, denseunits=8, densedrop=0.1, embedding=32):
        model = Sequential()
        # Embedding layer
        model.add(Embedding(input_dim=vocabsize, output_dim=embedding,
                            input_length=inputtokens))
        model.add(GlobalMaxPool1D())
        # Hidden layer
        model.add(Dense(denseunits, activation='relu'))
        model.add(Dropout(densedrop))
        # Output layer
        model.add(Dense(vocabsize, activation='softmax'))
        return model

"""Dictionary of model architectures indexed by a string"""
MODELSBYNAME = {
    "dilatedconv": DilatedConvModel,
    "wavenet": WavenetModel,
    "lstm": LSTMModel,
    "stackedlstm": StackedLSTMModel,
    "smalllstm": SmallWavenet,
    "cnnlstm": CNNLSTMModel,
    "pcp": PerceptronModel
}


def modelbyname(modelname):
    """Returns a model generating class by name"""
    if modelname not in MODELSBYNAME:
        raise ValueError("Unknown model %s" % modelname)
    return MODELSBYNAME[modelname]