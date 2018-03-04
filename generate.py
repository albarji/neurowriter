
# coding: utf-8

# Text generation using a pre-trained model
# Generates a infinite amount of text using a pre-trained model, using the style learned in such model.

import argparse
from keras.models import load_model
from neurowriter.encoding import loadencoding
from neurowriter.writer import Writer
from neurowriter.encoding import END


def generate(modelname, encodername, seed, creativity):
    """Generates infinite text using a pre-trained model and a seed text"""
    # Load pre-trained encoder
    encoder = loadencoding(encodername)

    # Load pre-trained model
    model = load_model(modelname)

    # Text generation
    print("Seed:", seed)
    writer = Writer(model, encoder, creativity=creativity, batchsize=1, beamsize=1)
    print("Generated:")
    print(seed, end='')
    for token in writer.generate(seed):
        print(token, end='')
        if token == END:
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates text following a pre-trained style model.')
    parser.add_argument('model', type=str, help='name of the pre-trained model')
    parser.add_argument('encoder', type=str, help='name of the pre-trained encoder')
    parser.add_argument('--seed', type=str, help='seed to use to initialize the generator. Default: empty string',
                        default='')
    parser.add_argument('--creativity', type=float, help='amount of creativity in the generation. Default: 0.5',
                        default=0.5)
    args = parser.parse_args()

    generate(modelname=args.model, encodername=args.encoder, seed=args.seed, creativity=args.creativity)
