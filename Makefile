.PHONY: help build-image notebook-server train-batch tests

condafile = conda.txt
docker = docker
ifdef GPU
  condafile=conda-gpu.txt
  docker=nvidia-docker
endif

help:
	@echo "Running options:"
	@echo "\t python-deps \t Install all necessary dependencies in the local python conda environment"
	@echo "\t build-image \t Builds the project docker image"
	@echo "\t notebook-server \t Starts a Jupyter notebook server with all necessary dependencies"
	@echo "\t train-batch \t Launches the training notebook in batch mode"

python-deps:
	conda install -y --file=$(condafile)
	pip install -r pip.txt

build-image:
	$(docker) build -t neurowriter .

notebook-server:
	$(docker) run -it -v $(shell pwd):/neurowriter --net=host neurowriter

train-batch:
	$(docker) run -d -it -v $(shell pwd):/neurowriter --entrypoint bash neurowriter runbatch.sh train.ipynb

tests:
	nosetests -v --with-coverage --cover-package=neurowriter --cover-erase

