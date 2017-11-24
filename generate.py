
# coding: utf-8

# # Text generation using a pre-trained model
# Generates a infinite amount of text using a pre-trained model, using the style learned in such model.

# ## Global config

# Name of pre-trained model/encoder (without enc/h5 extensions)
pretrainedname = "superheroes14_bpe.json"


# Seed text to use for generation. For free generation use an empty string.
seed = ""


# ### Process config
encodername = pretrainedname + '.enc'
modelname = pretrainedname + '.h5'


# Load pre-trained encoder
from neurowriter.encoding import Encoder, loadencoding
encoder = loadencoding(encodername)


# Load pre-trained model
from keras.models import load_model
model = load_model(modelname)


# ## Text generation
from neurowriter.writer import Writer
from neurowriter.encoding import END

print("Seed:", seed)
writer = Writer(model, encoder, creativity=0.5, batchsize=1, beamsize=1)
print("Generated:")
print(seed, end='')
for token in writer.generate(seed):
    print(token, end='')
    if token == END:
        print('\n')

