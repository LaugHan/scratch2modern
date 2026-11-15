# 每种精度再换 2 个种子各跑一次（差距都在 0.02 以内，和种子的波动同一个量级）
cd "$(dirname "$0")"
for precision in fp64 fp32 tf32 bf16_autocast bf16_pure fp16_scaler fp16_noscaler fp8; do
    for seed in 43 44; do ../../.venv/bin/python train.py $precision $seed; echo; done
done
