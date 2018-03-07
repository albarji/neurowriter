"""
Tests for the main scripts.

@author: Álvaro Barbero Jiménez
"""

from subprocess import run
from tempfile import TemporaryDirectory


def test_tokenize_train_generate():
    """Tokenization, training and generation main scripts can be run correctly"""
    tmpdir = TemporaryDirectory()

    # Tokenization
    infile = "tokenizecorpus.py"
    tokenized = tmpdir.name + "/toyseries_bpe.json"
    run(["python",  infile, "corpus/toyseries.txt", "multilinetxt", tokenized], check=True)

    # Training
    encoding = tmpdir.name + "/toyseries.enc"
    model = tmpdir.name + "/toyseries.h5"
    run(["python", "train.py", tokenized, "json", encoding, model, "--architecture", "pcp", "--trials", "15",
         "--maxepochs", "10"], check=True)

    # Generation
    run(["python", "generate.py", model, encoding, "--maxtokens", "100"], check=True)
