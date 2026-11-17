# 现在的配置（adam + linear），以及换上 Muon、WSD 之后：两两组合，各 3 个种子
cd "$(dirname "$0")"
for seed in 42 43 44; do
    for opt in adam muon; do
        for sched in linear wsd; do
            [ -f ../../out/1.11_summary/${opt}_${sched}_s${seed}/final.csv ] && continue
            ../../.venv/bin/python train.py opt=$opt sched=$sched seed=$seed; echo
        done
    done
done
