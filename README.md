# scratch2modern

从 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 出发，一次只加一样东西，看它对模型有什么影响。

每一章是 `src/` 下的一个目录，每一节是章里的一个目录。相邻两节 diff 一下，就是这一节的全部改动。代码和 notes 的写法见 [src/CONVENTIONS.md](src/CONVENTIONS.md)。

```
data/                   # 数据集（不进 git），各章共用
src/
├── CONVENTIONS.md      # 项目、代码、notes 的规范
├── ch1/
│   └── 1.01_microgpt/
│       ├── train.py    # 这一节的完整代码
│       └── notes.md    # 改了什么、结果、笔记
└── ch2/
out/
└── ch1/1.01_microgpt/  # 这一节的输出：log.csv、samples.txt
```
