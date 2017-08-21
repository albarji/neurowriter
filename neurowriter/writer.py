
# coding: utf-8

# Module for generating written text using a pretrained model
#
# @author Álvaro Barbero Jiménez

import numpy as np
import itertools

from neurowriter.symbols import NULL, START, END

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
            seedtokens = [NULL] * (inputtokens-len(seed)-1) + [START] + seedtokens
        seedcoded = self.encoder.encodetokens(seedtokens[-inputtokens:])
        
        # Iterate generation
        while True:
            # Predict token probabilities using model and beam search
            newcodes = self.beamsearch(seedcoded)
            # Drop old tokens, add new ones
            # Also drop tokens if longer than necessary
            seedcoded = np.append(seedcoded[len(newcodes):], newcodes)[:inputtokens]
            # Yield generated tokens (in text form)
            for code in newcodes:
                newtoken = self.encoder.index2char[code] 
                yield newtoken
                # If yielded end token, restart seed
                if newtoken == END:
                    restart = [NULL] * (inputtokens-1) + [START]
                    seedcoded = self.encoder.encodetokens(restart)
                    break
            
    def beamsearch(self, seedcoded):
        """Generates token predictions using a beam search algorithm
        
        Inputs:
            seedcoded: numpy array with indexes of seed tokens
        
        Returns the best sequence of tokens found.
        """
        # Store original length of seed
        maxlen = len(seedcoded)
        # Predict first token probabilities
        probs = self.model.predict(np.array([seedcoded]), verbose=0)[0]
        # Initiate candidates for beam search
        newcandidates = [(np.log(p), [i]) for i, p in enumerate(probs)]
        # Select candidates for beam search
        candidates = self.drawcandidates(newcandidates, self.beamsize)
        # Beam depth extension steps
        for _ in range(1, self.batchsize):
            # Extend each candidate
            newcandidates = []
            for logprob, tokens in candidates:
                # Update seed with candidate tokens
                # Also drop tokens if longer than necessary
                candseed = np.append(seedcoded[len(tokens):], tokens)[:maxlen]
                # Predictions for next token
                probs = self.model.predict(np.array([candseed]), verbose=0)[0]
                # Add to pool of next round tokens
                newcandidates.extend([
                    (logprob + np.log(p), tokens + [idx])
                    for idx, p in enumerate(probs)
                ])
            # Select candidates for next step
            candidates = self.drawcandidates(newcandidates, self.beamsize)
        # Return tokens of best candidate found
        return max(candidates, key=lambda x: x[0])[1]
    
    def drawcandidates(self, candidates, n):
        """Draws n candidates from a candidates list
        
        If no creativity has been configured, just draw the best candidates.
        If creativy has been configured, draw with random sampling according
        to the probability of each candidate.
        """
        # No creativity: take the top with highest probability
        if self.creativity == 0:
            return topk(candidates, self.beamsize, key=lambda x: x[0])
        # Creativity: random sampling
        else:
            probs = np.array([prob for prob, tokens in candidates])
            return [candidates[sample(probs, self.creativity)] for _ in range(n)]
            
def normalize(probs):
    """Normalizes a list of probabilities, so that they sum up to 1"""
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]
    
def sample(logprobs, temperature=1.0):
    """Modifies probabilities with a given temperature, to add creativity""" 
    probs = np.exp(logprobs / temperature)
    normprobs = normalize(probs)
    return np.argmax(np.random.multinomial(1, normprobs, 1))

def topk(l, k, key=lambda x: x):
    """Returns a sublist with the top k elements from a givenlist. Accepts key"""
    idx, _ = zip(*sorted(enumerate(l), key = lambda x: key(x[1]), 
                         reverse=True))
    return [l[i] for i in idx[0:k]]
    