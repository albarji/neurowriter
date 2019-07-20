"""Main file for training a neurowriter model"""

import argparse
import tempfile

from neurowriter.corpus import FORMATTERSBYNAME
from neurowriter.dataset import Dataset
from neurowriter.optimizer import train
from neurowriter.tokenizer import Tokenizer


def run_train(corpus, corpusformat, outputdir, inputtokens, maxepochs):
    """Trains a Neurowriter model"""
    # Load corpus
    corpus = FORMATTERSBYNAME[corpusformat](corpus)
    print("Training with corpus:", corpus[0][0:1000])

    # Build dataset
    tokenizer = Tokenizer()
    dataset = Dataset(corpus, tokenizer, inputtokens)

    # Model training
    train(dataset, outputdir, inputtokens=inputtokens, maxepochs=maxepochs, patience=10, batchsize=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurowriter model")
    parser.add_argument("corpus", type=str, help="Corpus file to use for training")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("outputdir", type=str, help="Directory in which to save trained models")
    parser.add_argument("--inputtokens", type=int, default=128, help="Number of previous tokens to use for generation")
    parser.add_argument("--maxepochs", type=int, default=1000, help="Maximum epochs to run model training")
    args = parser.parse_args()

    run_train(args.corpus, args.corpusformat, args.outputdir, inputtokens=args.inputtokens, maxepochs=args.maxepochs)
