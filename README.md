# Neurowriter

THIS IS WORK IN PROGRESS

## TODOs and possible improvements

* Add BatchNormalization layers.

From Facebook's convolutional translation paper

* Tokens are dealt with embeddings instead of one-hot encoder.
* The position of each token is also added as a parallel embedding
* Dropout for the embeddings and for the input of each convolutional block

## References

* WaveNet paper: https://arxiv.org/pdf/1609.03499.pdf
* A Keras implementation of WaveNet: https://github.com/usernaamee/keras-wavenet/blob/master/simple-generative-model.py
* Another one: https://github.com/basveeling/wavenet/blob/master/wavenet.py
* Facebook's convolutional translation paper: https://arxiv.org/pdf/1705.03122.pdf

Parallelization models in Keras

* One weird trick for parallelizing convolutional neural networks: https://arxiv.org/pdf/1404.5997.pdf
* Data parallelism in Keras: https://stackoverflow.com/questions/43821786/data-parallelism-in-keras
* Other approach to data parallelism in Keras: https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012


