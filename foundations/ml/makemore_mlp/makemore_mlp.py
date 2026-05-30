import os
import torch
import torch.nn.functional as F



# ── 1. Load data + build vocabulary ───────────────────────────────────────────
# Same as bigram.py — reuse the same stoi/itos
words = open(os.path.join(os.path.dirname(__file__), 'names.txt'), 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))  # unique characters sorted alphabetically
chars = ['.'] + chars  # add '.' token at the beginning
stoi = {ch: i for i, ch in enumerate(chars)}  # char to int mapping
itos = {i: ch for ch, i in stoi.items()} 
print(itos)