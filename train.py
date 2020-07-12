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


def run_train(corpus, corpusformat, outputdir, pretrained_model, special_tokens_file, max_steps, checkpointsteps, 
        trainvalratio, batchsize, gradaccsteps, patience):
    """Trains a Neurowriter model"""
    # Load corpus
    logging.info("Loading corpus...")
    corpus = FORMATTERSBYNAME[corpusformat](corpus)
    logging.info(f"Corpus sample: {corpus[0][0:1000]}")

    # Load special tokens list
    if special_tokens_file is not None:
        with open(special_tokens_file, "r") as f:
            special_tokens = f.read().splitlines()
    else:
        special_tokens = []

    # Model training
    logging.info("Training model...")
    model = Model(pretrained_model=pretrained_model, special_tokens=special_tokens)
    model.fit(corpus, outputdir, max_steps=max_steps, checkpointsteps=checkpointsteps, 
        gradient_accumulation_steps=gradaccsteps, batch_size=batchsize, trainvalratio=trainvalratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurowriter model")
    parser.add_argument("corpus", type=str, help="Corpus file to use for training")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("outputdir", type=str, help="Directory in which to save trained models")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-multilingual-cased", help="Pretrained Transformers model to use as base")
    parser.add_argument("--special_tokens_file", type=str, default=None, help="File with special symbols to add as tokens to the tokenizer, one symbol per line")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps to run model training")
    parser.add_argument("--checkpointsteps", type=int, default=100, help="Create a model checkpoint every n steps")
    parser.add_argument("--trainvalratio", type=int, default=3, 
                        help="Number of training patterns for each validation pattern")
    parser.add_argument("--batchsize", type=int, default=8, help="Size of training batches")
    parser.add_argument("--gradaccsteps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Wait this many epochs without improvement to stop training")
    args = parser.parse_args()

    run_train(args.corpus, args.corpusformat, args.outputdir, pretrained_model=args.pretrained_model, special_tokens_file=args.special_tokens_file,
        max_steps=args.max_steps, checkpointsteps=args.checkpointsteps, trainvalratio=args.trainvalratio, batchsize=args.batchsize,
        gradaccsteps=args.gradaccsteps, patience=args.patience)
