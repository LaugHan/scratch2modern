# 训练集只用 1/20（约 31 万个 token，4096 步约过 53 遍），让模型死记数据，再比较 weight decay、梯度裁剪、dropout
cd "$(dirname "$0")"
run() { # 用法：run wd=adamw seed=42
    local init=0.08 wd=none clip=none dropout=0 lr=0.01 seed=42
    local "$@"
    local dir=../../out/1.10_details/init${init}_wd${wd}_clip${clip}_drop${dropout}_lr${lr}_s${seed}_f20
    [ -f $dir/final.csv ] || [ -f $dir/log.csv ] && return # 已经跑过（包括发散的）
    ../../.venv/bin/python train.py "$@" frac=20; echo
}
for seed in 42 43 44; do
    for wd in none adamw adamw_noemb; do run wd=$wd seed=$seed; done
    run clip=1.0 seed=$seed
    for dropout in 0.1 0.1late; do run dropout=$dropout seed=$seed; done
done
