"""Main file for training a neurowriter model"""

import matplotlib as mpl
mpl.use("Agg")

import argparse
import tempfile

from neurowriter.corpus import FORMATTERSBYNAME
from neurowriter.tokenizer import TOKENIZERSBYNAME, tokenizerbyname
from neurowriter.models import MODELSBYNAME, modelbyname
from neurowriter.encoding import Encoder
from neurowriter.optimizer import hypertrain

def train(corpus, corpusformat, encoderfile, modelfile, architecture, tokenizer, trials, tmpmodels, checkpoint,
          maxepochs):
    """Trains a Neurowriter model"""
    # Load corpus
    corpus = FORMATTERSBYNAME[corpusformat](corpus)
    print("Training with corpus:", corpus[0][0:1000])

    # Encoding
    encoder = Encoder(corpus, tokenizerbyname(tokenizer) if tokenizer is not None else None)
    encoder.save(encoderfile)

    print("Computed encoder:", encoder.char2index)

    # Prepare temporary files
    if tmpmodels is None:
        tmpdir = tempfile.TemporaryDirectory()
        tmpmodels = tmpdir.name
    if checkpoint is None:
        tmpfile = tempfile.NamedTemporaryFile()
        checkpoint = tmpfile.name

    # Model training
    modelclass = modelbyname(architecture)

    model = hypertrain(modelclass, encoder, corpus, tmpmodels, n_calls=trials, verbose=2,
                       valmask=[False]*3+[True], checkpointfile=checkpoint, maxepochs=maxepochs)
    model.save(modelfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurowriter model")
    parser.add_argument("corpus", type=str, help="Corpus file to use for training")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("encoder", type=str, help="Name of output encoder file")
    parser.add_argument("model", type=str, help="Name of output model file")
    parser.add_argument("--architecture", type=str, default="lstm",
                        help="Model architecture: " + str(list(MODELSBYNAME)))
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer class: " + str(list(TOKENIZERSBYNAME)))
    parser.add_argument("--trials", type=int, default=100, help="Number of hyperoptimization trials")
    parser.add_argument("--tmpmodels", type=str, default=None, help="Directory where to save intermediate models")
    parser.add_argument("--checkpoint", type=str, default=None, help="Hyperoptimization checkpoint file")
    parser.add_argument("--maxepochs", type=int, default=1000, help="Maximum epochs to run per model trial")
    args = parser.parse_args()

    train(args.corpus, args.corpusformat, args.encoder, args.model, args.architecture, args.tokenizer, args.trials,
          args.tmpmodels, args.checkpoint, args.maxepochs)
