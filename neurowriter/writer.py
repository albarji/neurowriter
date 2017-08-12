
# coding: utf-8

# Module for generating written text using a pretrained model
#
# @author Álvaro Barbero Jiménez

import numpy as np
import itertools

from neurowriter.encoding import NULL

class Writer():      
    def __init__(self, model, encoder, creativity=0, beamsize=5, batchsize=3):
        """Creates a writer using a pretrained model
        
        Arguments:
            model: keras model to use for generation
            encoder: character encoder used
            creativity: creativity rate (probability temperature
            beamsize: size of beam search
            batchsize: number of tokens generated at the same time in beam search
        """
        self.model = model
        self.encoder = encoder
        self.creativity = creativity
        self.beamsize = beamsize
        self.batchsize = batchsize
        
    def write(self, seed="", length=1000):
        """Start writing characters
        
        Arguments:
            seed: text to initialize generation
            length: length of the generated text
            
        Return the generated text.
        """        
        # Iterate generation
        return itertools.islice(self.generate(seed), length)
    
    def generate(self, seed=""):
        """Infinite generator of text following the Writer style.
        
        Arguments:
            seed: text to initialize generation
        
        Returns one piece of text at a time.
        """
        # Prepare seed
        seedtokens = self.encoder.tokenizer.transform(seed)
        inputtokens = self.model.layers[0].input_shape[1]
        if len(seedtokens) < inputtokens:
            seedtokens = [NULL] * (inputtokens-len(seed)) + seedtokens
        seedcoded = self.encoder.encodetokens(seedtokens[-inputtokens:])
        
        # Iterate generation
        while True:
            # Predict token probabilities using model and beam search
            newcodes = self.beamsearch(seedcoded)
            # Drop old tokens, add new ones
            seedcoded = np.append(seedcoded[len(newcodes):], newcodes)
            # Yield generated tokens (in text form)
            for code in newcodes:
                newtoken = (
                    self.encoder.index2char[code] 
                    + self.encoder.tokenizer.intertoken
                )
                yield newtoken
            
    def beamsearch(self, seedcoded):
        """Generates token predictions using a beam search algorithm
        
        Inputs:
            seedcoded: numpy array with indexes of seed tokens
        
        Returns the best sequence of tokens found.
        """
        # Predict first token probabilities
        probs = self.model.predict(np.array([seedcoded]), verbose=0)[0]
        probs = alterprobs(probs, self.creativity)
        # Initiate candidates for beam search
        newcandidates = [(np.log(p), [i]) for i, p in enumerate(probs)]
        # Select most probable candidates
        candidates = topk(newcandidates, self.beamsize, key=lambda x: x[0])
        # Beam depth extension steps
        for _ in range(1, self.batchsize):
            # Extend each candidate
            newcandidates = []
            for logprob, tokens in candidates:
                # Update seed with candidate tokens
                candseed = np.append(seedcoded[len(tokens):], tokens)
                # Predictions for next token
                probs = self.model.predict(np.array([candseed]), verbose=0)[0]
                probs = alterprobs(probs, self.creativity)
                # Add to pool of next round tokens
                newcandidates.extend([
                    (logprob + np.log(p), tokens + [idx])
                    for idx, p in enumerate(probs)
                ])
            # Select top best candidates for next step
            candidates = topk(newcandidates, self.beamsize, key=lambda x: x[0])
        # Return tokens of best candidate found
        return max(candidates, key=lambda x: x[0])[1]
            
def normalize(probs):
    """Normalizes a list of probabilities, so that they sum up to 1"""
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]
    
def alterprobs(a, temperature=1.0):
    """Modifies probabilities with a given temperature, to add creativity""" 
    # Limit case where temperature is 0: don't change probs
    if temperature == 0:
        return a
    # Standard case: readjust probs by temperature and run random draws
    a = np.exp(np.log(a) / temperature)
    a = normalize(a)
    return np.random.multinomial(1, a, 1)[0]

def topk(l, k, key=lambda x: x):
    """Returns a sublist with the top k elements from a givenlist. Accepts key"""
    idx, _ = zip(*sorted(enumerate(l), key = lambda x: key(x[1]), 
                         reverse=True))
    return [l[i] for i in idx[0:k]]
    