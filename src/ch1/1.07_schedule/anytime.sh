# 第三组：WSD 可以随时停。WSD 训练 N 步，前 80% 就是 constant，所以同一个种子下，
# 不同 N 的 WSD 和 constant 在衰减之前是同一条曲线。对比每个 N 从头训练的 cosine。
cd "$(dirname "$0")"
for steps in 1024 2048; do
    for schedule in wsd cosine; do ../../.venv/bin/python train.py $schedule 0.01 0 $steps; echo; done
done
