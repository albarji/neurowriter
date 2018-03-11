<div align="center">
  <img src="img/logo.png" height="300"><br>
</div>

-----------------
[![Build Status](https://travis-ci.org/albarji/neurowriter.svg?branch=master)](https://travis-ci.org/albarji/neurowriter)
[![Coverage Status](https://coveralls.io/repos/github/albarji/neurowriter/badge.svg?branch=master)](https://coveralls.io/github/albarji/neurowriter?branch=master)
[![Code Climate](https://codeclimate.com/github/albarji/neurowriter.svg)](https://codeclimate.com/github/albarji/neurowriter)

Tool for creating automated text generators, following the style of a given corpus of documents.

## Install

You can make use of neurowriter either through a dockerized version, or you may install it locally in your computer. 

### Local install

You will need an **Anaconda Python 3** distribution. Then run the following commands to install the required
python packages in your active environment:

    make install
    
or, if you want to build the project with GPU support, run

    make install-gpu

### Docker deployment

You will need [docker](https://www.docker.com/) and permissions to build and run images. Then run

    make build-image
    
to build the Neurowriter Docker image. If instead you want to build this image with GPU support, you will also need 
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker), and perform the build as 

    make build-image-gpu
    
Running a container of this image opens a command line terminal inside the container, which you can use to execute
the main scripts interactively, or a particular script by providing the command as parameters to docker run. 
We highly recommend mounting a volume to store the processing results outside the container.    

## Usage

The basic process to create a text generator is the following:

* Prepare a **corpus** of documents in a **proper format**.
* **Tokenize** the corpus (optional, but strongly recommended).
* Select a **model architecture** to learn from the corpus and run the **training** process.
* Use the created model to generate new texts!

### Preparing a corpus

A corpus is a set of documents that will be used to train the text generator. You will need to adequate your corpus
according to one of the following formats:

#### Single text

A text file containing a single document.

    This is is a single document corpus.
    All the lines from this file belong to the same document.
    And now for something different!
    
    PINEAPPLES!!!
    
#### Multiline text 

A text file containing multiple documents, one document per line. 
Note that documents with line breaks cannot be represented in this format.

    This is a multidocument.
    The file stores one document per line.
    So there are three documents here.
    

#### CSV

A CSV file with one row per document. 
If the file has several columns, the text of the documents is assumed to be contained in the first column.
Other columns present in the file are loaded, but at present not used in the learning process.

    title,genres
    Na Boca da Noite,['Drama']
    The Other Side of the Wind,['Drama']
    Prata Palomares,['Thriller']

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
  
### Tokenizing the corpus

A Tokenizer is a procedure for breaking down a document into its basic pieces. Neurowriter provides the following
tokenizers.

* **CharTokenizer**: breaks down the document into basic characters.
* **WordTokenizer**: breaks down the document into basic characters + frequent words.
* **SubWordTokenizer**: breaks down the document into basic characters + frequent subword pieces, using a BPE algorithm.

For a corpus of documents that are more than a few words long, it is recommended to use the SubWordTokenizer. Note
however this tokenizer can be quite slow. 

To apply the tokenizer to a corpus, use the **tokenizecorpus.py** script. An example of use would be:

    python tokenizecorpus.py corpus/toyseries.txt multilinetxt toyseries_bpe.json

You will need to provide the following arguments: 

* Name of the input corpus file
* Corpus format 
* Name of output tokenize corpus file

By default a SubWordTokenizer will be used. Other tokenizers can be selected using the **--tokenizer** argument. 

### Training the generator

To train the generator use the **train.py** script. For example:

    python train.py toyseries_bpe.json json toyseries.enc toyseries.h5

The following arguments must be provided:

* Name of the (previously tokenized) corpus file
* Corpus format. Note tokenizer corpus always follow JSON format.
* Output file with model token encodings.
* Output file with model weights.

Many other tunable parameters exists: run the script with **-h** to check them. One particularly important is the
model architecture: by default an LSTM model is used, but faster or more expressive models are also included.
If the results are not good enough, moving to an StackedLSTMModel might produce improvements.

### Generate text!

Use the **generate.py** script to generate text!

    python generate.py toyseries.h5 toyseries.enc
    
Mandatory arguments are:

* File with previously trained model token encodings.
* File with previously trained model weights.

This should be enough to start generating text. Note that generation will proceed indefinitely, outputting a **<END>**
at the end of each generated document, then proceeding to generate another one.

For better results you can hand-tune other generation parameters. The **creativity** rate is probably the most 
significant: small values force the model to produce only high probability sequences, while higher
values introduce randomness in the generation. As a rule of thumb, if the generator keeps repeating the same patterns
again and again, an increase in creativity might help, whereas the generator producing garbage text will need a
decrease in creativity. Generally values between 0.2 and 0.75 give the best results.

You can also provide de model with a **seed** value that will be used to initialize the text generation. This is
useful to prompt the model to start writing a chapter with a given name, for instance.

## Generation examples

Pre-trained models are available for some of these examples: check the **samplemodels** folder.

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
    Rock Grand<END>
    The Secret Home<END>
    Morcia Raven<END>
    Alasan  the F Hacking Kay<END>
    The Internet World<END>
    The Get to Fly Andrea Me Me Pant<END>
    Betting Boss<END>
    State Kids<END>
    Spirit<END>
    Love That Brother<END>
    
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

### H.P. Lovecraft (english)

    EAS TMARSEN
    
    In the sound of the shadowy streets and stalk masonry of the black city, as the chill of vast dark and 
    motive.
    
    As I almost tried to be a revolver that the doctor drew the place to the cold and Sianian glen he had 
    come to the door and the and scenes of Georgian chiselled back to the liness, and strewn up again to a 
    continuous laws of the  most probable soul. The rites high in the town came in a scene to the eyes of 
    league as the motor-great hoveral  often happened in the cohort otic young priest and the entire wind 
    and and shocking and one with the intellectual  open streets of the corner of that crawl of something 
    the touch of the yellow polished graves and prying head to  the at the other city which had kept a 
    shaped thing, and in the parctive way gave it a strange body on the house  and or it not against him. 
    Dubl abode the gnawed and lower and illusion had been the dark and concreated daemon of on the 
    terrible ward--Alhazred horror, and with the sight of the moon and which a Roman the cryptic town had 
    ordered a kind of care as it had come. [...]
    
    ERES ARNSSAOF SADE FLEEV
    
    The the ancient Mwan Halus, and in the ancient but a nameless and face of the far door of the dark 
    altar and and the old man and in the northern and the a strange thing that the strange world was not 
    to be the glass and the and  uncovered cemeteries of the light and the illimitable prop of the tiny 
    Vusan of the the cellar and the nameless of the place. It was the old man in the old man when I had 
    seen the old man and the first and the in the wall of the old thing and the Saria of the town was of 
    the old coil. It was a face of the day of the queer people of the low bungalow and the corridors of the 
    the face of the Babylon, and when the cost of a the low cohort was every time 
    and uncovering the ritual of the moon of the door and held the very day. [...]
    
    THE DODOEKLEH

    THE HATES I YOOR

    But I had not said to the boy of his face and the sentient in the crowning whispering dance and opening 
    the city as I had one to the Pirkon, of what had been murderous and consoled to the before the room. 
    This night, it was a faint sun of then and loud and half-with the slumber it had been at the year, and 
    who was a small moon, and had still to nothing an unsancied blackness of some time to me that the Thur 
    had was the muffled poe of this where his dream and books and the blottle doesn of the honour. The great 
    Czar would ought to fly down to the the great paintings and  crouching of the unearthly hand. The boy 
    had been a room of body as one in the summit of the old man should not see. The cubes of my legion was 
    the next day and the mo and in the same moment "I was relieved that is over the sanity [...]
    
### El Quijote de Cervantes (spanish)

    Capítulo XXXVIII. De la aventura del sol fue Sancho, con la flaca de la señora
    Dulcinea del Toboso, y que el hombre hizo de las manos a su buen caballero,
    de que el que estaba apartada y con su espada de monte la fe que me
    había de hacer en el lugar de don Quijote de la Mancha, y alabó -dijo
    don Quijote-, y que no se le fue a otra vez ni el cercio, que no
    hizo pan de zapato por don Quijote, en el cual el nombre, a comedimiento y
    este caballo, fue a lo que el don Quijote se partiese de la mano, porque, antes
    que se usan, las flaqueza de las manos que no encargaban y lo que se
    dio la historia de la soledad, no se más de tocar de aquella cueva de
    Montesinos, y que ya le echó donde los caballeros. A no fueron la muerte, que
    no había venido a mi tierra, y no le dijese que él le pudiera.

    Finalmente, don Quijote le dijo:

    -Sí tenía -dijo don Quijote-, y es que ha de ser en fe y la dé a todos
    si mi remedio no no me acuerdo por esta madre, y en las virtudes que yo me
    no en qué favor les he visto de los hombres en el mundo, porque me quiere, y
    que yo te acuerdo de la fecha.

    A lo cual respondió don Quijote:

    -El cual se puede ser tomar la imagen de que le da de verse en
    ánimo que la atreva de las niñerías y de su que me tiene de la historia
    disposición de la cabeza, no hay contado alguna para dormir que más de
    volaro a lo que quisieres.

    -No lo que yo quisiere saca Dios -dijo Sancho-: veamos un real que no lo
    hubiera de ver a la ley que vio el cual, finalmente, yo no sé que no me
    rede con que en él no parece que no lo ha sido de la batalla.

    -Si la vida de una carca vuestra merced -respondió don Quijote-, que
    no es algo de un verdadera que se había de estar que yo lo había de hacer en
    el negocio de no lejos que trataba, y está un mundo no tiene la salud y
    querido la medario, y quizá con sus pensamientos como se amenaza.
    
### El Apocalipsis (spanish)

    EL Y los reyes de la tierra y de la tierra y de los siglos. Y el ángel tocó la trompeta, y el que está en 
    el cielo y las cosas que están en él, y de la tierra y el que está sentado sobre el mar, y la tierra y vi 
    a los hombres que no se halla de la tierra y de la tierra y de los siglos de los siglos. Y el templo de 
    Dios y del Cordero. 

    La mujer que estaba sentado en el cielo y las cosas que están en él me dijo: Estas son los que se llama 
    de los siglos. Y el ángel tocó la trompeta, y la gloria y la tierra y el que es el libro de la tierra y 
    el que está sentado en el cielo y las cosas que están en él el nombre de la tierra y de los siglos. 

    7 El que tiene oído, oiga lo que estaban en el cielo y las cosas que están en ella se ha a venir; y el 
    que estaba sentado en el cielo y las cosas que están en ella se ha sido con fuego y a la tierra y los que 
    había en su mano un ángel derramó su copa sobre el mar. 

    La mensada de oro, y los que había en el cielo y las cosas que están en ella se ha azufre. 
    
    
### Harry Potter (english)

    "I have never been able to get a matter of a Death Eater," said Harry, "and I knew what it is a little 
    the a few people who were a little Death Eaters -"
    "You are able to be in the castle and you do not be a mother."
    "I have been a few days ago -"
    "No time to break down, Severus, you are not going to keep it at the same time to see you," said 
    Dumbledore. "I shall be a little chance to make the power of the other last time now. You cant know what 
    you can have to get a lot of the other other Potter and her old boy and your head wants to be a chance 
    that you are hiding you to fight the place that You-Know-Who would be to keep the snake to see you in the 
    grounds."
    "What is it, Voldemort?" Harry asked, and he had never seen the Death Eaters in the castle. "You know how 
    I have been and the Death Eaters we found out of the forest of the Death Eaters will be to find him, he 
    can do it."
    "I see it for him and I must be able to get in the castle, and you are not the same time you think he was 
    the same time, and I know I will not have seen a little curse and to tell him that he could be able to 
    fight him and we have been a few years ago, and it was a word of the Order of the Phoenix with the boy - " 
    "I know what Voldemort had been going to fight the place where you are killed my name," said Harry. 
    "He was dead!"
    [...]

    "The Death Eaters will be able to do it -"
    "I could not speak to you, I got the way to the Chamber of Secrets," said Harry, looking at him at once.
    "I could be a Death Eater and you got the snake. If you are not a new boy - they can have to do it. Of 
    course, I thought I saw you to come to the Great Hall, and the Death Eaters were head of the room at 
    Hogwarts, and we can keep the students in the hall of the full castle, and they know that you can be the 
    last time he was listening to the plan to be a little fight. You have been in the castle for the place 
    where you have been and the Sorting Hat Potter is not on the back of the castle, and you can have a hundred 
    death of the Slytherin House of the Dark Arts in the world  well do you like to do a man and he was in the 
    forest, was all the same time to do it, and I can see that you were trying to be the way to have a few 
    steps to the castle, the only thing he was two inches at Hogwarts, and he had to do here, but you know what 
    he was going to be brave in the Order of the Phoenix to see it, but you were up to the castle and he was 
    lying here . . . I was at the moment that he was not to have a matter of the last time he was  that you 
    will be here to me with the Sorting Hat."
    [...]

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
