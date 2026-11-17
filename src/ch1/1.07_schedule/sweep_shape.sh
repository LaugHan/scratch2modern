# 第一组：衰减的形状。不加 warmup，每种调度扫 5 个峰值学习率（constant 的最佳点在最左边，再往左补一个），依次跑
cd "$(dirname "$0")"
for schedule in constant linear cosine wsd; do
    for lr in 0.005 0.01 0.02 0.04 0.08; do ../../.venv/bin/python train.py $schedule $lr; echo; done
done
../../.venv/bin/python train.py constant 0.0025
