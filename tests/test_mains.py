"""
Tests for the main scripts.

@author: Álvaro Barbero Jiménez
"""

from subprocess import run
from tempfile import TemporaryDirectory


def run_tokenize_train_generate(docker=None):
    """Tokenization, training and generation main scripts can be run correctly inside a Docker container"""
    tmpdir = TemporaryDirectory()

    if docker is None:
        dockercommands = []
    else:
        dockercommands = [docker, "run", "--rm", "-it", "-v", "%s:%s" % (tmpdir.name, tmpdir.name), "neurowriter"]

    # Tokenization
    infile = "tokenizecorpus.py"
    tokenized = tmpdir.name + "/toyseries_bpe.json"
    run(dockercommands + ["python",  infile, "corpus/toyseries.txt", "multilinetxt", tokenized], check=True)

    # Training
    encoding = tmpdir.name + "/toyseries.enc"
    model = tmpdir.name + "/toyseries.h5"
    run(dockercommands + ["python", "train.py", tokenized, "json", encoding, model, "--architecture", "pcp",
                          "--trials", "15", "--maxepochs", "10"], check=True)

    # Generation
    run(dockercommands + ["python", "generate.py", model, encoding, "--maxtokens", "100"], check=True)


def test_tokenize_train_generate():
    """Tokenization, training and generation main scripts can be run correctly"""
    run_tokenize_train_generate()


def _test_docker_tokenize_train_generate():
    """Tokenization, training and generation main scripts can be run correctly in docker"""
    run_tokenize_train_generate(docker="docker")


def _test_nvidiadocker_tokenize_train_generate():
    """Tokenization, training and generation main scripts can be run correctly in nvidia-docker"""
    run_tokenize_train_generate(docker="nvidia-docker")
