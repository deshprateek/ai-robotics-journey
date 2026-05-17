import os
import torch
"""
Bigram Language Model
Predicts the next character given the current character.
Trained on a list of names from names.txt
"""



# ── 1. Load the data ──────────────────────────────────────────────────────────
# Read names.txt — one name per line
# e.g. ["emma", "olivia", "ava", ...]
words = open(os.path.join(os.path.dirname(__file__), 'names.txt'), 'r').read().splitlines()
print(f"total words: {len(words)}")   # 32032
print(f"first 5:     {words[:5]}")    # ['emma', 'olivia', 'ava', 'isabella', 'sophia']


# ── 2. Build the vocabulary ───────────────────────────────────────────────────
# Collect all unique characters across all names
# Add a special '.' token to represent start and end of a name
# Build stoi: character → integer  e.g. {'a': 0, 'b': 1, ..., '.': 26}
# Build itos: integer → character  e.g. {0: 'a', 1: 'b', ..., 26: '.'}
chars = sorted(list(set(''.join(words))))  # unique characters sorted alphabetically
chars = ['.'] + chars  # add '.' token at the beginning
stoi = {ch: i for i, ch in enumerate(chars)}  # char to int mapping
itos = {i: ch for ch, i in stoi.items()}  # int to char mapping


# ── 3. Build the bigram count table ───────────────────────────────────────────
# Create a 27x27 tensor N of zeros (27 = 26 letters + '.' token)
# For every name, wrap it with '.' on both sides: e.g. ".emma."
# Slide a window of 2 characters across it: (.→e), (e→m), (m→m), (m→a), (a→.)
# For each pair (ch1, ch2): N[stoi[ch1], stoi[ch2]] += 1
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)  # 27x27 count matrix
for word in words:
    word = '.' + word + '.'  # add start and end tokens
    for ch1, ch2 in zip(word, word[1:]):  # slide window of 2 characters
        i1, i2 = stoi[ch1], stoi[ch2]  # convert chars to indices
        N[i1, i2] += 1  # increment count for this bigram

# print(N)


# ── 4. Convert counts to probabilities ────────────────────────────────────────
# Each row in N represents one character
# Divide each row by its sum so the row adds up to 1.0
# Result P[i][j] = "probability that character j follows character i"
P = N.float()  # convert counts to float for division
P /= P.sum(1, keepdims=True)  # normalize rows to sum to 1
# print(P)

# ── 5. Sample from the model ──────────────────────────────────────────────────
# To generate a name:
#   - Start with the '.' token
#   - Look at the current character's row in P
#   - Sample the next character from that probability distribution
#   - Repeat until we sample '.' again (end of name)
#   - Convert sampled integers back to characters using itos

g=torch.Generator().manual_seed(2147483647) 
for _ in range(10):
     # for reproducibility
    out =''
    ix = 0
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()  # sample next char index
        out+=itos[ix]  # convert index to char and append to output
        if ix == 0:  # if we sampled '.', end of name
            break
    print(out)  # join list of chars into a string and print it


# ── 6. Evaluate the model — compute loss ─────────────────────────────────────
# For every bigram in the dataset, look up its probability in P
# Loss = average negative log likelihood across all bigrams
# Lower loss = model assigns higher probability to the actual next characters
# A perfect model memorises the training data — loss approaches 0
# A random model assigns equal probability to all — loss is log(27) ≈ 3.3
