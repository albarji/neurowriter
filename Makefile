.PHONY: help install install-gpu build-image build-image-gpu notebook-server train-batch tests

help:
	@echo "Running options:"
	@echo "\t install \t Install all necessary dependencies in the local python conda environment"
	@echo "\t install-gpu \t Install all necessary dependencies in the local python conda environment with GPU support"
	@echo "\t build-image \t Builds the project docker image"
	@echo "\t build-image-gpu \t Builds the project docker image with GPU support"
	@echo "\t notebook-server \t Starts a Jupyter notebook server with all necessary dependencies"
	@echo "\t train-batch \t Launches the training notebook in batch mode"

install:
	conda install -y --file=conda.txt
	pip install -r pip.txt

install-gpu:
	conda install -y --file=conda-gpu.txt
	pip install -r pip.txt

build-image:
	docker build -t neurowriter --build-arg INSTALL=install .

build-image-gpu:
	docker build -t neurowriter --build-arg INSTALL=install-gpu .

notebook-server:
	nvidia-docker run -it -v $(shell pwd):/neurowriter --net=host neurowriter

train-batch:
	nvidia-docker run -d -it -v $(shell pwd):/neurowriter --entrypoint bash neurowriter runbatch.sh train.ipynb

tests:
	nosetests -v --nologcapture --with-coverage --cover-package=neurowriter --cover-erase
