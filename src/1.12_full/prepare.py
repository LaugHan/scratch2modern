"""
把完整的 TinyStories 训练集编码成 token，供 1.12 训练用。
分词器沿用 1.05 的 bpe1024。1.05 只存了词表、没存合并规则，所以这里用和 1.05 完全相同的数据和步骤重新学一遍，
再检查学出来的词表和 1.05 存下的完全一致。验证集直接沿用 1.05 的那 500 个故事。
用法：python prepare.py     先把 TinyStories-train.txt 下载到 data/tinystories_train.txt，写出 data/tok/bpe1024_full.pt
"""

import os, random, torch
from collections import Counter
from multiprocessing import Pool
import regex

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
read_docs = lambda name: [d.strip() for d in open(os.path.join(root, 'data', name)).read().split('<|endoftext|>') if d.strip()]
pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") # GPT-2 的切分规则，和 1.05 相同

def merge(ids, pair, new_id):
    # 把 ids 里所有相邻的 pair 换成 new_id
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)

# 一、重新学合并规则：和 1.05 的 bpe.py 相同的数据、相同的打乱顺序、相同的步骤
saved = torch.load(os.path.join(root, 'data', 'tok', 'bpe1024.pt'))
docs = read_docs('tinystories_valid.txt')[1:] # 文件里第一个故事是不完整的
random.seed(42)
random.shuffle(docs)
words = Counter(tuple(chunk.encode('utf-8')) for doc in docs[:int(0.9 * len(docs))] for chunk in pattern.findall(doc))
merges, vocab = {}, [bytes([i]) for i in range(256)]
for new_id in range(256, len(saved['vocab']) - 1): # 最后一个 id 是 BOS
    pairs = Counter()
    for ids, n in words.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += n
    pair = max(pairs, key=pairs.get)
    merges[pair] = new_id
    vocab.append(vocab[pair[0]] + vocab[pair[1]])
    words = {merge(ids, pair, new_id): n for ids, n in words.items()}
assert vocab + [b'\n'] == saved['vocab'], "重新学到的词表和 1.05 的不一致"
BOS = saved['bos']

# 二、编码：每个 chunk 反复合并最早学到的那一对，直到没有可合并的；同样的 chunk 只算一次
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
    # 每个故事前面加一个 BOS；词表只有 1024 个，用 int16 存，省一半以上的显存
    return torch.tensor([t for doc in docs for t in [BOS] + [i for chunk in pattern.findall(doc) for i in encode_chunk(chunk)]], dtype=torch.int16)

if __name__ == '__main__':
    docs = read_docs('tinystories_train.txt')
    batches = [docs[i:i + 10000] for i in range(0, len(docs), 10000)]
    with Pool(32) as pool: # 按故事分批，多进程并行编码
        train = torch.cat(pool.map(encode, batches) + [torch.tensor([BOS], dtype=torch.int16)])
    torch.save({'train': train, 'val': saved['val'], 'vocab': saved['vocab'], 'bos': BOS}, os.path.join(root, 'data', 'tok', 'bpe1024_full.pt'))
    print(f"stories: {len(docs)}, train tokens: {len(train)}, val tokens: {len(saved['val'])}, bytes per token: {sum(map(len, docs)) / len(train):.2f}")
