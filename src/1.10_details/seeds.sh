# 初始化：Kaiming 的最佳学习率在扫描范围的边上，再往大补一个点；然后每种初始化在各自最好的学习率上补 2 个种子
cd "$(dirname "$0")"
out=../../out/1.10_details
../../.venv/bin/python train.py init=kaiming lr=0.04; echo
for init in xavier kaiming 0.02 0.08 zero_proj; do
    best=$(for f in $out/init${init}_wdnone_clipnone_drop0_lr*_s42/final.csv; do echo "$(tail -1 $f | cut -d, -f2) $(basename $(dirname $f))"; done | sort -n | head -1 | sed 's/.*_lr\([0-9.]*\)_s42/\1/')
    echo "$init: best lr $best"
    for seed in 43 44; do
        [ -f $out/init${init}_wdnone_clipnone_drop0_lr${best}_s${seed}/final.csv ] || { ../../.venv/bin/python train.py init=$init lr=$best seed=$seed; echo; }
    done
done
