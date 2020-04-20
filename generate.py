
# coding: utf-8

# Text generation using a pre-trained model
# Generates a infinite amount of text using a pre-trained model, using the style learned in such model.

import argparse
from itertools import count

from neurowriter.model import Model


def generate(modelfolder, seed, temperature, maxdocuments=None, maxtokens=1000):
    """Generates text using a pre-trained model and a seed text"""
    # Load model
    model = Model(modelfolder)

    # Text generation
    for generated_documents in count(start=1):
        print(model.generate(seed=seed, temperature=temperature, appendseed=True, maxlength=maxtokens))
        if maxdocuments is not None and generated_documents >= maxdocuments:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates text following a pre-trained style model.')
    parser.add_argument('modelfolder', type=str, help='folder of the pre-trained model')
    parser.add_argument('--seed', type=str, help='seed to use to initialize the generator. Default: empty string',
                        default='')
    parser.add_argument('--temperature', type=float, help='amount of creativity in the generation. Default: 0.5',
                        default=0.5)
    parser.add_argument('--maxdocuments', type=int, help='how many documents to generate. Default: unlimited', default=None)
    parser.add_argument('--maxtokens', type=int, help='maximum number of tokens to generate per document. Default: 1000',
                        default=1000)
    args = parser.parse_args()

    generate(modelfolder=args.modelfolder, seed=args.seed, temperature=args.temperature, maxdocuments=args.maxdocuments, 
             maxtokens=args.maxtokens)
