.PHONY: help install install-gpu build-image build-image-gpu notebook-server tests tests-docker tests-nvidiadocker

help:
	@echo "Running options:"
	@echo "\t install \t\t Install all necessary dependencies in the local python conda environment"
	@echo "\t install-gpu \t\t Install all necessary dependencies in the local python conda environment with GPU support"
	@echo "\t build-image \t\t Builds the project docker image"
	@echo "\t build-image-gpu \t Builds the project docker image with GPU support"

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

tests:
	nosetests -v --nologcapture --with-coverage --cover-package=neurowriter --cover-erase

tests-docker:
	nosetests -v --nologcapture tests/test_mains.py:_test_docker_tokenize_train_generate

tests-nvidiadocker:
	nosetests -v --nologcapture tests/test_mains.py:_test_nvidiadocker_tokenize_train_generate
