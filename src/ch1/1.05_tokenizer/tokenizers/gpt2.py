"""
GPT-2 BPE via tiktoken: 50257 tokens, trained by OpenAI on web text.
Writes data/tok/gpt2.pt
"""

import os, random, torch
import tiktoken

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
docs = [d.strip() for d in open(os.path.join(root, 'data', 'tinystories_valid.txt')).read().split('<|endoftext|>') if d.strip()]
docs = docs[1:] # the first story in the file is cut off
random.seed(42)
random.shuffle(docs)
n_train = int(0.9 * len(docs))
train_docs, val_docs = docs[:n_train], docs[n_train:n_train + 500]

enc = tiktoken.get_encoding('gpt2')
BOS = enc.eot_token # <|endoftext|>, the last token (50256)
vocab = [enc.decode_single_token_bytes(i) for i in range(enc.n_vocab - 1)] + [b'\n'] # BOS stands for a newline

def encode(docs):
    return torch.tensor([t for ids in enc.encode_ordinary_batch(docs) for t in [BOS] + ids] + [BOS])

train, val = encode(train_docs), encode(val_docs)
torch.save({'train': train, 'val': val, 'vocab': vocab, 'bos': BOS}, os.path.join(root, 'data', 'tok', 'gpt2.pt'))
print(f"gpt2: vocab {len(vocab)}, train tokens {len(train)}, val tokens {len(val)}")
