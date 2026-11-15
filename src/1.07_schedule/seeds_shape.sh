# 在每种调度的最佳峰值学习率上，再换 2 个种子各跑一次
cd "$(dirname "$0")"
for a in "constant 0.005" "linear 0.01" "cosine 0.01" "wsd 0.01"; do
    set -- $a
    for seed in 43 44; do ../../.venv/bin/python train.py $1 $2 0 4096 $seed; echo; done
done
