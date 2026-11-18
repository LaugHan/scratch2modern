#!/bin/bash
# 2.02 的全部命令。训练每个约 1 小时，依次跑。learned 就是 2.01 的基线，不重跑
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
for pos in rope alibi none sincos sin binary frac int; do
    python train.py pos=$pos > ../../../out/ch2/2.02_position/pos$pos.txt 2>&1
done
python eval.py > ../../../out/ch2/2.02_position/eval.txt 2>&1   # 每个位置的 loss、外推
python attention.py                                               # 外推时注意力落在哪
python plot.py
