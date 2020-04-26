"""Main file for training a neurowriter model"""

import argparse
import logging
import tempfile

from neurowriter.corpus import FORMATTERSBYNAME
from neurowriter.model import Model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def run_train(corpus, corpusformat, outputdir, pretrained_model, maxepochs, checkpointepochs, trainvalratio,
              batchsize, gradaccsteps, patience):
    """Trains a Neurowriter model"""
    # Load corpus
    logging.info("Loading corpus...")
    corpus = FORMATTERSBYNAME[corpusformat](corpus)
    logging.info(f"Corpus sample: {corpus[0][0:1000]}")

    # Model training
    logging.info("Training model...")
    model = Model(pretrained_model=pretrained_model)
    model.fit(corpus, outputdir, maxepochs=maxepochs, checkpointepochs=checkpointepochs, 
        gradient_accumulation_steps=gradaccsteps, patience=patience, batch_size=batchsize,
        trainvalratio=trainvalratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurowriter model")
    parser.add_argument("corpus", type=str, help="Corpus file to use for training")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("outputdir", type=str, help="Directory in which to save trained models")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-multilingual-cased", help="Pretrained Transformers model to use as base")
    parser.add_argument("--maxepochs", type=int, default=100, help="Maximum epochs to run model training")
    parser.add_argument("--checkpointepochs", type=int, default=10, help="Create a model checkpoint every n epochs")
    parser.add_argument("--trainvalratio", type=int, default=3, 
                        help="Number of training patterns for each validation pattern")
    parser.add_argument("--batchsize", type=int, default=8, help="Size of training batches")
    parser.add_argument("--gradaccsteps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Wait this many epochs without improvement to stop training")
    args = parser.parse_args()

    run_train(args.corpus, args.corpusformat, args.outputdir, pretrained_model=args.pretrained_model, maxepochs=args.maxepochs,
              checkpointepochs=args.checkpointepochs, trainvalratio=args.trainvalratio, batchsize=args.batchsize,
              gradaccsteps=args.gradaccsteps, patience=args.patience)
