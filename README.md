# scratch2modern

从 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 出发，一次只加一样东西，看它对模型有什么影响。

每一节是 `src/` 下的一个目录，单独能跑。相邻两节 diff 一下，就是这一节的全部改动。

```
data/                   # 数据集（不进 git）
src/
└── 1.01_microgpt/
    ├── train.py        # 这一节的完整代码
    └── notes.md        # 改了什么、结果、笔记
out/
└── 1.01_microgpt/      # 这一节的输出：log.csv、samples.txt
```
