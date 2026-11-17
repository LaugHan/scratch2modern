"""
Byte-level BPE, written from scratch in the style of minbpe.
Start from the 256 bytes, then repeatedly merge the most frequent adjacent pair into a new token.
Usage: python bpe.py <vocab_size>     writes data/tok/bpe<vocab_size>.pt
"""

import os, sys, random, torch
from collections import Counter
import regex # like re, but understands \p{L} (any letter) and \p{N} (any number)

vocab_size = int(sys.argv[1])
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
docs = [d.strip() for d in open(os.path.join(root, 'data', 'tinystories_valid.txt')).read().split('<|endoftext|>') if d.strip()]
docs = docs[1:] # the first story in the file is cut off
random.seed(42)
random.shuffle(docs)
n_train = int(0.9 * len(docs))
train_docs, val_docs = docs[:n_train], docs[n_train:n_train + 500]

# Split text into chunks first (the GPT-2 pattern): merges never cross a chunk boundary,
# so " dog" can become a token but "dog." cannot
pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Training works on unique chunks with their counts, which is much faster than on the raw text
chunk_counts = Counter(chunk for doc in train_docs for chunk in pattern.findall(doc))
words = {tuple(chunk.encode('utf-8')): n for chunk, n in chunk_counts.items()}

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)

merges = {} # (id, id) -> new id, in the order they were learned
vocab = [bytes([i]) for i in range(256)] # the bytes of each token
for new_id in range(256, vocab_size - 1): # the last id is BOS
    pairs = Counter()
    for ids, n in words.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += n
    pair = max(pairs, key=pairs.get)
    merges[pair] = new_id
    vocab.append(vocab[pair[0]] + vocab[pair[1]])
    words = {merge(ids, pair, new_id): n for ids, n in words.items()}
    if new_id % 256 == 0 or new_id == vocab_size - 2:
        print(f"merge {new_id - 255:4d}: {vocab[pair[0]]!r} + {vocab[pair[1]]!r} -> {vocab[new_id]!r} ({pairs[pair]} times)")
BOS = vocab_size - 1
vocab.append(b'\n') # BOS stands for a newline

# Encoding a chunk: keep merging the pair that was learned earliest, until none is left
cache = {}
def encode_chunk(chunk):
    if chunk not in cache:
        ids = tuple(chunk.encode('utf-8'))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float('inf')))
            if pair not in merges:
                break
            ids = merge(ids, pair, merges[pair])
        cache[chunk] = ids
    return cache[chunk]

def encode(docs):
    return torch.tensor([t for doc in docs for t in [BOS] + [i for chunk in pattern.findall(doc) for i in encode_chunk(chunk)]] + [BOS])

train, val = encode(train_docs), encode(val_docs)
torch.save({'train': train, 'val': val, 'vocab': vocab, 'bos': BOS}, os.path.join(root, 'data', 'tok', f'bpe{vocab_size}.pt'))
print(f"bpe{vocab_size}: vocab {len(vocab)}, train tokens {len(train)}, val tokens {len(val)}")
