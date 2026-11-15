# 在每个优化器的最佳学习率上，再换 2 个种子各跑一次，看结果的波动有多大
cd "$(dirname "$0")"
for a in "sgd 2" "momentum 0.707" "adam 0.01" "adamw 0.01" "lion 0.00125" "muon 0.02"; do
    set -- $a
    for seed in 43 44; do ../../.venv/bin/python train.py $1 $2 $seed; echo; done
done
