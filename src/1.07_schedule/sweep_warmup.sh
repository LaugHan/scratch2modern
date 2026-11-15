# 第二组：warmup。衰减形状固定为 WSD，warmup 取 0、128、512 步，每种都把峰值学习率扫到发散为止
cd "$(dirname "$0")"
for warmup in 0 128 512; do
    for lr in 0.01 0.02 0.04 0.08 0.16; do
        [ -f ../../out/1.07_schedule/wsd/lr${lr}_w${warmup}_n4096_s42/log.csv ] || { ../../.venv/bin/python train.py wsd $lr $warmup; echo; } # 第一组已经跑过的不再跑
    done
done
