"""Main file for training a neurowriter model"""

import argparse
import logging
import tempfile

from neurowriter.corpus import FORMATTERSBYNAME
from neurowriter.dataset import Dataset
from neurowriter.model import Model
from neurowriter.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)


def run_train(corpus, corpusformat, outputdir, inputtokens, maxepochs, checkpointepochs, trainvalratio,
              batchsize):
    """Trains a Neurowriter model"""
    # Load corpus
    logging.info("Loading corpus...")
    corpus = FORMATTERSBYNAME[corpusformat](corpus)
    logging.info(f"Corpus sample: {corpus[0][0:1000]}")

    # Build dataset
    logging.info("Tokenizing corpus...")
    tokenizer = Tokenizer()
    dataset = Dataset(corpus, tokenizer, inputtokens, trainvalratio=trainvalratio, batchsize=batchsize)

    # Model training
    logging.info("Training model...")
    model = Model()
    model.fit(dataset, outputdir, maxepochs=maxepochs, patience=10, checkpointepochs=checkpointepochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurowriter model")
    parser.add_argument("corpus", type=str, help="Corpus file to use for training")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("outputdir", type=str, help="Directory in which to save trained models")
    parser.add_argument("--inputtokens", type=int, default=128, help="Number of previous tokens to use for generation")
    parser.add_argument("--maxepochs", type=int, default=1000, help="Maximum epochs to run model training")
    parser.add_argument("--checkpointepochs", type=int, default=10, help="Create a model checkpoint every n epochs")
    parser.add_argument("--trainvalratio", type=int, default=3, 
                        help="Number of training patterns for each validation pattern")
    parser.add_argument("--batchsize", type=int, default=8, help="Size of training batches")
    args = parser.parse_args()

    run_train(args.corpus, args.corpusformat, args.outputdir, inputtokens=args.inputtokens, maxepochs=args.maxepochs,
              checkpointepochs=args.checkpointepochs, trainvalratio=args.trainvalratio, batchsize=args.batchsize)
