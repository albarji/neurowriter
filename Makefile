.PHONY: help build-image notebook-server train-batch

help:
	@echo "Running options:"
	@echo "\t python-deps \t Install all necessary dependencies in the local python conda environment"
	@echo "\t build-image \t Builds the project docker image"
	@echo "\t notebook-server \t Starts a Jupyter notebook server with all necessary dependencies"
	@echo "\t train-batch \t Launches the training notebook in batch mode"

python-deps:
	conda install -q -y --file=conda.txt
	pip install -r pip.txt

build-image:
	nvidia-docker build -t neurowriter .

notebook-server:
	nvidia-docker run -it -v $(shell pwd):/neurowriter --net=host neurowriter

train-batch:
	nvidia-docker run -d -it -v $(shell pwd):/neurowriter --entrypoint bash neurowriter runbatch.sh train.ipynb

