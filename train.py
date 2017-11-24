# coding: utf-8
import matplotlib as mpl
mpl.use("Agg")

# Corpus and models save folder
corpusname = "superheroes14_bpe.json"
tmpmodelsfolder = "tmpmodels/superheroes14_betterwavenet/"
#corpusname = "sonnets_bpe.json"
#tmpmodelsfolder = "tmpmodels/sonnets_stackedlstm/"

# Corpus loader method to use
from neurowriter.corpus import Corpus
corpusloader = Corpus.load_json

# Tokenizer object to use (can be None if text is already tokenized)
from neurowriter.tokenizer import CharTokenizer, WordTokenizer, SubwordTokenizer
tokenizer = None

# Network architecture class to use
from neurowriter.models import DilatedConvModel, WavenetModel, StackedLSTMModel, LSTMModel, SmallLSTMModel, SmallWavenet
#architecture = StackedLSTMModel
architecture = WavenetModel

# Number of hyperoptimization trials (recommended at least 15)
hypertrials = 100


# ### Process config

# Get all relevant file names

# In[8]:

corpusfile = 'corpus/' + corpusname
encodername = corpusname + '.enc'
modelname = corpusname + '.h5'


# ## Load corpus

# In[9]:

corpus = corpusloader(corpusfile)


# In[10]:

corpus[0][0:1000]


# ## Encoding

# In[11]:

from neurowriter.encoding import Encoder
encoder = Encoder(corpus, tokenizer)
encoder.save(encodername)


# In[12]:

encoder.char2index


# ## Model training

# Train the generator model, trying different hyperparameters and selecting the model producing lower loss in a  validation split of the data.
# 
# Note this might take a very long time, so during the optimization temporary versions of the model will be saved.

# In[13]:

from neurowriter.optimizer import hypertrain

model = hypertrain(architecture, encoder, corpus, tmpmodelsfolder, n_calls=hypertrials, verbose=2, 
                   valmask=[False]*3+[True], checkpointfile=tmpmodelsfolder+"checkpoints")
model.save(modelname)


# In[ ]:



