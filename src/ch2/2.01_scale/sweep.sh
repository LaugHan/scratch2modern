#!/bin/bash
# 2.01 跑过的全部命令。用 GPU 0
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")"

# 扫学习率：每个 2048 步（跑的时候默认初始化还是 0.08，现在重跑要加 init=0.08）
for lr in 0.005 0.01 0.02 0.04; do
    python train.py lr=$lr steps=2048
done


# 初始化的对照：0.08 是 1.10 在宽 48 上选的；0.044 ≈ 1/√512，0.02 是 GPT-2 的做法，0.01 看 0.02 是不是谷底。lr 0.01、2048 步，和扫学习率时的 lr0.01 对照
for init in 0.044 0.02 0.01; do
    python train.py init=$init steps=2048
done

# 正式训练：默认配置（8 层，宽 512，lr 0.01，init 0.02，16384 步）
python train.py
# 对照：第一章的初始化 0.08（这次跑的时候默认值还是 0.08，结果目录从 default 改名为 init0.08）
python train.py init=0.08
