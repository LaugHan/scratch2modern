"""
把 FineWeb-Edu 的一个分片编码成 GPT-2 的 token，供第二章使用。
先下载 sample-10BT 的第一个分片到 data/fineweb_edu/000_00000.parquet（约 2 GB）：
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
写出 data/tok/fineweb_edu_train.bin 和 fineweb_edu_val.bin：uint16，每篇文档前面加一个 <|endoftext|>。
"""

import os

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
enc = tiktoken.get_encoding('gpt2')
n_val_docs = 10_000  # 前 1 万篇做验证集，其余做训练集；分片里文档的顺序本来就是打乱的


def write(docs, split, chunk=100_000):
    # 分块编码、分块写盘，免得整份 token 列表占满内存
    path = os.path.join(ROOT, 'data', 'tok', f'fineweb_edu_{split}.bin')
    n_tokens = 0
    with open(path, 'wb') as f:
        for i in range(0, len(docs), chunk):
            ids = enc.encode_ordinary_batch(docs[i:i + chunk], num_threads=32)
            tokens = np.concatenate([np.array([enc.eot_token] + doc, dtype=np.uint16) for doc in ids])
            tokens.tofile(f)
            n_tokens += len(tokens)
    print(f"{split}: {len(docs):,} docs, {n_tokens:,} tokens")
    return n_tokens


docs = pq.read_table(os.path.join(ROOT, 'data', 'fineweb_edu', '000_00000.parquet'), columns=['text'])
docs = docs.column('text').to_pylist()
n_bytes = sum(len(d.encode('utf-8')) for d in docs[n_val_docs:])
n_tokens = write(docs[n_val_docs:], 'train')
write(docs[:n_val_docs], 'val')
print(f"bytes per token: {n_bytes / n_tokens:.2f}")
