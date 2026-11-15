# 每种精度训练一次，依次跑
cd "$(dirname "$0")"
for precision in fp64 fp32 tf32 bf16_autocast bf16_pure fp16_scaler fp16_noscaler fp8; do
    ../../.venv/bin/python train.py $precision; echo
done
