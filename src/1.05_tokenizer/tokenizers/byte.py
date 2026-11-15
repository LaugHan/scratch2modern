"""
Byte-level tokenizer: one token per UTF-8 byte. Any text can be encoded with these 256 tokens.
Writes data/tok/byte.pt
"""

import os, random, torch

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
docs = [d.strip() for d in open(os.path.join(root, 'data', 'tinystories_valid.txt')).read().split('<|endoftext|>') if d.strip()]
docs = docs[1:] # the first story in the file is cut off
random.seed(42)
random.shuffle(docs)
n_train = int(0.9 * len(docs))
train_docs, val_docs = docs[:n_train], docs[n_train:n_train + 500]

# Tokens 0..255 are the bytes themselves, 256 is BOS
BOS = 256
vocab = [bytes([i]) for i in range(256)] + [b'\n'] # the bytes of each token; BOS stands for a newline

def encode(docs):
    return torch.tensor([t for doc in docs for t in [BOS] + list(doc.encode('utf-8'))] + [BOS])

train, val = encode(train_docs), encode(val_docs)
torch.save({'train': train, 'val': val, 'vocab': vocab, 'bos': BOS}, os.path.join(root, 'data', 'tok', 'byte.pt'))
print(f"byte: vocab {len(vocab)}, train tokens {len(train)}, val tokens {len(val)}")
