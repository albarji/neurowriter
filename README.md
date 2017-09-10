<div align="center">
  <img src="img/log.png" height="300"><br>
</div>

-----------------

Tool for creating automated text generators, following the style of a given corpus of documents.

## Install

You can make use of neurowriter either through a dockerized version, or you may install it locally in your computer. 

### Local install

You will need an **Anaconda Python 3** distribution. Then run the following commands to install the required
python packages in your active environment:

    make python-deps

### Docker deployment

You will need [docker](https://www.docker.com/) and permissions to build and run images. Then run

    make build-image
    
to build the Neurowriter Docker image. Once built, you can start a notebook server accesible from you host machine with

    make make notebook-server 
    
## Usage

The basic process to create a text generator is the following:

* Prepare a **corpus** of documents in a **proper format**.
* Select a **Tokenizer** and parse the corpus with it.
* Select a **model architecture** to learn from the corpus and run the **training** process.
* Use the created model to generate new texts!

### Preparing a corpus

A corpus is a set of documents that will be used to train the text generator. The following corpus formats are accepted:

#### Single text

A text file containing a single document.

    This is is a single document corpus.
    All the lines from this file belong to the same document.
    And now for something different!
    
    PINEAPPLES!!!
    
Such a corpus can be loaded into neurowriter as follows:

    from neurowriter.corpus import Corpus
    corpus = Corpus.load_singletxt(filename)
    
#### Multiline text 

A text file containing multiple documents, one document per line. 
Note that documents with line breaks cannot be represented in this format.

    This is a multidocument.
    The file stores one document per line.
    So there are three documents here.
    
Such a corpus can be loaded into neurowriter as follows:

    from neurowriter.corpus import Corpus
    corpus = Corpus.load_multilinetxt(filename)

#### CSV

A CSV file with one row per document. 
If the file has several columns, the text of the documents is assumed to be contained in the first column.
Other columns present in the file are loaded, but at present not used in the learning process.

    title,genres
    Na Boca da Noite,['Drama']
    The Other Side of the Wind,['Drama']
    Prata Palomares,['Thriller']
    
Such a corpus can be loaded into neurowriter as follows:

    from neurowriter.corpus import Corpus
    corpus = Corpus.load_csv(datafile)
    
#### JSON

A JSON file in the form [{doc1}, {doc2}, ...] where each document must contain a "text" attibute with the contents
of the document. Othe fields present in the document are loaded, but at present not used in the learning process.

    [
        {
            "text" : "Na Boca da Noite",
            "genres" : ["Drama"]
        },
        {
            "text" : "The Other Side of the Wind",
            "genres" : ["Drama"]
        },
        {
            "text" : "Prata Palomares",
            "genres" : ["Thriller"]
        }
    ]
  
Such a corpus can be loaded into neurowriter as follows:

    from neurowriter.corpus import Corpus
    corpus = Corpus.load_json(datafile)
  
### Tokenizing the text

A Tokenizer is a procedure for breaking down a document into its basic pieces. Neurowriter provides the following
tokenizers.

* **CharTokenizer**: breaks down the document into basic characters.
* **WordTokenizer**: breaks down the document into basic characters + frequent words.
* **SubWordTokenizer**: breaks down the document into basic characters + frequent subword pieces, using a BPE algorithm.

For a corpus of documents that are more than a few words long, it is recommended to use the SubWordTokenizer. Note
however this tokenizer can be quite slow. 

To apply the tokenizer to a corpus, it is recommended to follow the steps in the notebook tokenize.ipynb. You will
need to provide:

* Name of the input corpus file
* Function to use to load the corpus
* Tokenizer class to use
* Name of output tokenize corpus file

### Training the generator

To train the generator follow the steps in the train.ipynb notebook. In particular, you will need to provide:

* Name of the corpus file
* Function to use to load the corpus
* Tokenizer to use (None if you are using an already tokenized corpus)
* Model architecture
* Number of hyperoptimization trials

The following model architectures are implemented in Neurowriter:

* **LSTMModel**: a bidirectional Long-Short Term Memory network.
* **StackedLSTMModel**: a bidirectional Long-Short Term Memory network + more stacked LSTM layers.
* **WavenetModel**: an implementation of the Wavenet model, adapted for text generation.
* **DilatedConvModel**: a model based on dilated convolutions + dense layers.

It is recommended to start with an LSTMModel. If the results are not good enough, moving to an StackedLSTMModel might
produce improvements.

Note the training process will be very slow, and if the connection with the notebook server is cut, the process will
stop. To this end, you can run the training procedure in offline batch mode by editing the notebook, configuration,
saving it, and then runnning

    make train-batch
    
Finally, if you wish to perform a training by hand-tuning the models hyperparameters, you can use the singletrain.ipynb
notebook instead.

### Generate text!

Just follow the steps in the generate.ipynb. You will need to provide the name of the model trained in the previous
step, and a seed to start the generation (which might be the empty string "")

For better results you can hand-tune the generation parameters at the bottom of the notebook. The **creativity** rate
is probably the most significant: small values force the model produce only high probability sequences, while higher
values introduce randomness in the generation. As a rule of thumb, of the generator keeps repeating the same patterns
again and again, an increase in creativity might help, whereas the generator producing garbage text will need a
decrease in creativity. Generally values between 0.2 and 0.75 give the best results.

## TODOs and possible improvements

Since this is still work in progress, here are some ideas I might try in the future:

* Try DenseNet architecture, or modifications thereof for Wavenet
* Add l2 regularization
* Include the position of each token in the document and/or in the input as a parallel embedding

Some amusing corpus to try:

* http://www.thecocktaildb.com/

## References

Learning models:

* WaveNet paper: https://arxiv.org/pdf/1609.03499.pdf
* A Keras implementation of WaveNet: https://github.com/usernaamee/keras-wavenet/blob/master/simple-generative-model.py
* Another one: https://github.com/basveeling/wavenet/blob/master/wavenet.py
* Facebook's convolutional translation paper: https://arxiv.org/pdf/1705.03122.pdf
* DenseNet: https://arxiv.org/pdf/1608.06993.pdf
	* Keras implementation: https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

Model parallelization in Keras:

* One weird trick for parallelizing convolutional neural networks: https://arxiv.org/pdf/1404.5997.pdf
* Data parallelism in Keras: https://stackoverflow.com/questions/43821786/data-parallelism-in-keras
* Other approach to data parallelism in Keras: https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
