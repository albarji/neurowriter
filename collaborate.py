# Human-machine collaborative text generation using a pre-trained model
# Alternates between asking the human for input and asking the model to
# generate, producing a collaborative composition

import argparse
from keras.models import load_model
from neurowriter.encoding import loadencoding
from neurowriter.writer import Writer
from neurowriter.encoding import END


def collaborate(modelname, encodername, creativity):
    """Generates infinite text using human input and a pre-trained model"""
    # Load pre-trained encoder
    encoder = loadencoding(encodername)

    # Load pre-trained model
    model = load_model(modelname)

    # Text generation loop
    writer = Writer(model, encoder, creativity=creativity, batchsize=1,
                    beamsize=1)
    while True:
        collaborate_document(writer)


def collaborate_document(writer, maxlines=14):
    """Creates a single document by collaboration between human and model

    Inputs
        writer: Writer object for automated text generation

    Returns the created collaboration.
    """
    print("Human, let's write a new composition! You start:\n\n")
    token = None
    composition = ""
    lines = 0

    while token != END and lines < maxlines:
        composition = composition + input("HUMAN-> ") + "\n"
        lines += 1
        print(f"--AI--> ", end='')
        seed = composition
        for token in writer.generate(seed):
            print(token, end='')
            composition += token
            if token == "\n":
                break
        lines += 1

    print("\n\n")
    print("Human, we created this composition together. "
          "Thanks for your inspiration!\n")
    print(composition)
    return composition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Human-machine collaborative text generation, '
                    'following the style of a pre-trained style model.')
    parser.add_argument('model', type=str, help='name of the pre-trained model')
    parser.add_argument('encoder', type=str,
                        help='name of the pre-trained encoder')
    parser.add_argument('--seed', type=str,
                        help='seed to use to initialize the generator. '
                             'Default: empty string',
                        default='')
    parser.add_argument('--creativity', type=float,
                        help='amount of creativity in the generation. '
                             'Default: 0.5',
                        default=0.5)
    args = parser.parse_args()

    collaborate(modelname=args.model, encodername=args.encoder,
                creativity=args.creativity)
