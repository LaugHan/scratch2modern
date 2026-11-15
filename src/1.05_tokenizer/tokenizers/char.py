"""
Character-level tokenizer: one token per character.
Writes data/tok/char.pt
"""

import os, random, torch

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
docs = [d.strip() for d in open(os.path.join(root, 'data', 'tinystories_valid.txt')).read().split('<|endoftext|>') if d.strip()]
docs = docs[1:] # the first story in the file is cut off
random.seed(42)
random.shuffle(docs)
n_train = int(0.9 * len(docs))
train_docs, val_docs = docs[:n_train], docs[n_train:n_train + 500]

# Every character in the data is a token, plus BOS between documents
chars = sorted(set(''.join(docs)))
stoi = {ch: i for i, ch in enumerate(chars)}
BOS = len(chars)
vocab = [ch.encode('utf-8') for ch in chars] + [b'\n'] # the bytes of each token; BOS stands for a newline

def encode(docs):
    return torch.tensor([t for doc in docs for t in [BOS] + [stoi[ch] for ch in doc]] + [BOS])

train, val = encode(train_docs), encode(val_docs)
torch.save({'train': train, 'val': val, 'vocab': vocab, 'bos': BOS}, os.path.join(root, 'data', 'tok', 'char.pt'))
print(f"char: vocab {len(vocab)}, train tokens {len(train)}, val tokens {len(val)}")
