# 每一级加速各训练一遍（3 个种子），依次跑。loss 应该不变，只有速度变
cd "$(dirname "$0")"
for seed in 42 43 44; do
    for level in baseline no_sync fused_adam sdpa compile cudagraph; do
        ../../.venv/bin/python train.py $level $seed; echo
    done
done
