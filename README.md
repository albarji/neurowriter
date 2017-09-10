<div align="center">
  <img src="img/logo.png" height="300"><br>
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

```python
from neurowriter.corpus import Corpus
corpus = Corpus.load_singletxt(filename)
```
    
#### Multiline text 

A text file containing multiple documents, one document per line. 
Note that documents with line breaks cannot be represented in this format.

    This is a multidocument.
    The file stores one document per line.
    So there are three documents here.
    
Such a corpus can be loaded into neurowriter as follows:

```python
from neurowriter.corpus import Corpus
corpus = Corpus.load_multilinetxt(filename)
```

#### CSV

A CSV file with one row per document. 
If the file has several columns, the text of the documents is assumed to be contained in the first column.
Other columns present in the file are loaded, but at present not used in the learning process.

    title,genres
    Na Boca da Noite,['Drama']
    The Other Side of the Wind,['Drama']
    Prata Palomares,['Thriller']
    
Such a corpus can be loaded into neurowriter as follows:

```python
from neurowriter.corpus import Corpus
corpus = Corpus.load_csv(datafile)
```
    
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

```python
from neurowriter.corpus import Corpus
corpus = Corpus.load_json(datafile)
```
  
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

## Generation examples

### Movie titles

**Corpus**: set of movie titles obtained from [IMDB](http://www.imdb.com/)

    Better Story<END>
    Last Company<END>
    The Love Balls: Part 2<END>
    The Salence Truth of Boys<END>
    Really Case to Disaster<END>
    Ana House Thief<END>
    The Secret of the Cast<END>
    The Countdust of Story<END>
    We We Travele<END>
    The Tale of the Trome<END>
    The Vecyme That White Edition<END>
    All Bedroom<END>
    Alive a Fall<END>
    Star Medial Candy<END>
    Star - To Polition Movie<END>
    A 10 Money of Presents<END>
    Search for All Are Episode Waters There Is the Superture and the Earth Home<END>
    Mike the Surprise<END>
    Last House<END>
    Amant Start<END>
    Secret Cast Hosudio<END>
    Martina Kitchel<END>
    The Man of the End of There's It Health Tall to Sea Pilot<END>
    The Star Secret Story<END>
    Ridet of the Dark Confession<END>
    Under the Beach<END>
    A 19<END>
    Is #6<END>
    Jack Just the Geast Comedy<END>
    The Problems of the Good<END>
    Headth St! Story of His for Million<END>
    Super Centry<END>
    Super D10000<END>
    The Company of the Rush Special<END>
    The Devil Is the Man of a Berrellist<END>
    The Story of the Body<END>
    Berney Engele<END>
    The Student for Cast<END>
    Anal Fire Part 2<END>
    Monky Semifinals<END>
    All Thing?<END>
    The Decille Day<END>
    
### Shot recipes (spanish)

**Corpus**: list of shot names and ingredients from [Wikipedal](http://wikipedal.org/Proyecto_Chupito)

    Dencie: Ron, licor de melocotón y lima.<END>
    Hiba: Pechè y naranja.<END>
    Tetsns: Vodka, licor de melocotón y Blue kiwi .<END>
    Aice Paja: Vodka, licor de melocotón y lima.<END>
    Cura: Martini y menta.<END>
    El venro: Licor 43, Batola: Vodka y granadina<END>
    Direta: Martini, licor de melocotón, zumo de fresa y lima.<END>
    More: Tequila, licor de melocotón, vodka y naranja.<END>
    Tate: Vodka y menta .<END>
    El menca: Vodka, licor de de mora.<END>
    Vanco: Patxarán, limón y kiwi.<END>
    Pitibe: Pechè y granadina.<END>
    Esko Mei: Granadina, ron, licor de melocotón y granadina<END>
    Rolas Pen: Ron y lima.<END>
    Chula Vara: Licor de mora, Licor de avellana y granadina<END>
    Doree: Patxarán, vodka y granadina<END>
    
### Sonnets (spanish)

**Corpus**: [Spanish Golden-Age Sonnets](https://github.com/bncolorado/CorpusSonetosSigloDeOro)

    LA LUZ DE MARTE
    
    Cuando en la esperanza de la frente
    de la mano de la fortuna cría
    por que en la esperanza y el pecho ardiente
    en la vida el sol de aquel que no guía.
    Par que se le han de que la luz de Marte,
    y la aurora la que es el mármol que siento
    en el mismo tiempo y en el cielo viene.<END>

    FE DE CERA
    
    Por un fe de su deidad más se ofrece
    de un tiempo y la tierra que se ve y en el cielo
    por fin de tus lágrimas, y por la mano
    de la vida y de más alas de la pena.
    Si el alma que en la luz desvelada
    la causa de esta parte de su aliento
    en ver de su virtud la fe más de cera,
    y no hay que de la luz de dolor no siento.<END>

    DICHOSO TÚ, CIEGO
    
    Yo un huego que de un nieto y su misma parte
    de tu mano alimenta un semblante
    de la vejez del tiempo de su gloria
    y en el que la queja lo que tu aliento
    su valor vuela el mal en tan segura
    tiene al sol, que en el rigor se atreve.
    Dichoso tú, ciego, me dio el que siento
    con que en que el mar de las estrellas toca
    el aviso al que revelan batalla,
    y el cuerpo se divide y en el cielo.
    ¿Qué el bien, ¿cómo es ver su sentido mío?<END>

Note: titles not generated, just manually added for effect

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
