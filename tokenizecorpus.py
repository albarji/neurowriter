"""Takes a corpus as input and applies a tokenizer, generating a new pre-processed corpus"""

import argparse

from neurowriter.corpus import Corpus, FORMATTERSBYNAME
from neurowriter.tokenizer import get_tokenizer


def tokenize(inputcorpus, corpusformat, outputcorpus):
    """Tokenizes a corpus and produces a new corpus of tokens in JSON format"""

    # Read corpus
    corpus = FORMATTERSBYNAME[corpusformat](inputcorpus)
    print(corpus[0:min(3, len(corpus))])

    # Tokenize corpus
    tokenizer = get_tokenizer()
    transformed = Corpus([tokenizer.tokenize(doc) for doc in corpus])
    print(transformed[0:min(3, len(corpus))])

    # Save resultant processed corpus
    transformed.save_json(outputcorpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizes a corpus")
    parser.add_argument("corpus", type=str, help="Corpus file to tokenize")
    parser.add_argument("corpusformat", type=str, help="Format of corpus file: " + str(list(FORMATTERSBYNAME)))
    parser.add_argument("tokenized", type=str, help="Name of output file in which to save tokenized corpus")
    args = parser.parse_args()

    tokenize(args.corpus, args.corpusformat, args.tokenized)
