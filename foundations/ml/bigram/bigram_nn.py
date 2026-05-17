import os
import torch
import torch.nn.functional as F

# from foundations.ml.micrograd.nn import MLP

"""
Bigram Language Model — Neural Network Version
Same goal as bigram.py but implemented as a neural network.
Instead of a counting table, we learn a weight matrix W that
produces the same probabilities through gradient descent.
"""


# ── 1. Load data + build vocabulary ───────────────────────────────────────────
# Same as bigram.py — reuse the same stoi/itos
words = open(os.path.join(os.path.dirname(__file__), 'names.txt'), 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))  # unique characters sorted alphabetically
chars = ['.'] + chars  # add '.' token at the beginning
stoi = {ch: i for i, ch in enumerate(chars)}  # char to int mapping
itos = {i: ch for ch, i in stoi.items()} 
# print(chars)  # ['emma', 'olivia', 'ava', 'isabella', 'sophia']

# ── 2. Build the trainin   g dataset ─────────────────────────────────────────────
# Convert every consecutive character pair in every word into:
#   xs — list of input character indices  e.g. [0, 5, 13, 13, 1, ...]
#   ys — list of target character indices e.g. [5, 13, 13, 1, 0, ...]
# Every (xs[i], ys[i]) pair is one training example:
#   "given character xs[i], the next character should be ys[i]"
# Hint: same zip(word, word[1:]) sliding window as bigram.py
xs, ys = [], []
for word in words:
    word = '.' + word + '.'  # add start and end tokens
    for ch1, ch2 in zip(word, word[1:]):  # slide window of 2 characters
        i1, i2 = stoi[ch1], stoi[ch2]  # convert chars to indices
        xs.append(i1)  # input character index
        ys.append(i2)  # target character index



# ── 3. Encode inputs as one-hot vectors ───────────────────────────────────────
# Neural net cannot take raw integers as input
# Convert xs into a matrix of one-hot vectors — shape: (num_examples, 27)
# Each row is all zeros except a single 1 at the character's index
# Hint: F.one_hot(torch.tensor(xs), num_classes=27).float()
xenc = F.one_hot(torch.tensor(xs), num_classes=len(stoi)).float()  # (num_examples, 27)
# print(xenc.shape)

# ── 4. Initialise the weight matrix W ─────────────────────────────────────────
# W is a (27, 27) matrix — randomly initialised
# Row i of W represents what the model thinks comes after character i
# This is what gets updated during training
# Hint: torch.randn((27, 27), requires_grad=True)
W = torch.randn(len(stoi), len(stoi), requires_grad=True)  # (27, 27)
# print(W)


# ── 5. Training loop ──────────────────────────────────────────────────────────
# For each step:
#   Forward pass:
#     - multiply one-hot inputs by W  →  logits (raw scores)
#     - convert logits to probabilities via softmax
#   Compute loss:
#     - for each training example, look up the probability assigned to the correct next character
#     - loss = negative log likelihood = -log(probability of correct character).mean()
#   Backward pass:
#     - zero gradients
#     - loss.backward()
#   Update:
#     - W.data -= learning_rate * W.grad

# model = MLP(27, [27])  # a simple model with one layer: input size 27, output size 27
learning_rate = 50.0
for step in range(200):
    # Forward pass
    logits = xenc @ W  # (num_examples, 27) = (num_examples, 27) @ (27, 27)
    probs = F.softmax(logits, dim=1)  # convert logits to probabilities
    loss = -probs[torch.arange(len(ys)), torch.tensor(ys)].log().mean()  # negative log likelihood
    # Compute loss
    # loss = -sum(p.log() for p in P) / len(P)  # negative log likelihood averaged over all examples
    # loss = -probs[torch.arange(len(ys)), torch.tensor(ys)].log().mean()  # negative log likelihood

    # Backward pass
    W.grad = None  # zero gradients
    loss.backward()  # compute gradients of loss with respect to W
    
    # Update weights
    W.data -= learning_rate * W.grad  # gradient descent step


# ── 6. Sample from the trained model ──────────────────────────────────────────
# Same loop as bigram.py but now using W instead of the counting table P:
#   - start with ix = 0 (the '.' token)
#   - one-hot encode ix
#   - multiply by W → logits → softmax → probabilities
#   - sample next character using torch.multinomial
#   - repeat until ix = 0 again
g = torch.Generator().manual_seed(2147483647)                                                                                                               
for _ in range(10):                                                                                                                                         
    out = ''                                                                                                                                                
    ix = 0                                                
    while True:                                                                                                                                             
        xenc_s = F.one_hot(torch.tensor([ix]), num_classes=len(stoi)).float()
        logits = xenc_s @ W                                                  
        probs = F.softmax(logits, dim=1)                                                                                                                    
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        if ix == 0:                                                                                                                                         
            break                                                                                                                                           
        out += itos[ix]
    print(out)                 




