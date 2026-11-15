# 四项训练细节，每次只改一项，依次跑。和默认配置相同的运行只跑一次
cd "$(dirname "$0")"
run() { # 用法：run init=xavier lr=0.005 seed=42
    local init=0.08 wd=none clip=none dropout=0 lr=0.01 seed=42
    local "$@"
    local dir=../../out/1.10_details/init${init}_wd${wd}_clip${clip}_drop${dropout}_lr${lr}_s${seed}
    [ -f $dir/final.csv ] || [ -f $dir/log.csv ] && return # 已经跑过（包括发散的）
    ../../.venv/bin/python train.py "$@"; echo
}

# 一、初始化：每种扫 3 个学习率（全零只跑一次，反正训练不动）
run init=zero
for init in xavier kaiming 0.02 0.08 zero_proj; do
    for lr in 0.005 0.01 0.02; do run init=$init lr=$lr; done
done

for seed in 42 43 44; do
    # 二、weight decay
    for wd in none l2 adamw adamw_noemb; do run wd=$wd seed=$seed; done
    # 三、梯度裁剪：学习率 0.01 和 0.04（1.07 里 0.04 会中途发散）
    for lr in 0.01 0.04; do
        for clip in none 1.0; do run clip=$clip lr=$lr seed=$seed; done
    done
    # 四、dropout
    for dropout in 0 0.1 0.1late; do run dropout=$dropout seed=$seed; done
done
