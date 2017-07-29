.PHONY: build help notebook-server train-batch

build:
	nvidia-docker build -t neurowriter .

help:
	@echo "Running options:"
	@echo "\t notebook-server \t Starts a Jupyter notebook server with all necessary dependencies"
	@echo "\t train-batch \t Launches the training notebook in batch mode"

notebook-server:
	nvidia-docker run -d -it -v $(shell pwd):/neurowriter --net=host neurowriter

train-batch:
	nvidia-docker run -d -it -v $(shell pwd):/neurowriter --entrypoint bash neurowriter runbatch.sh train.ipynb

