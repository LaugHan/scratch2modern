# scratch2modern

从一个 200 行的 GPT 出发，一步步搭出现代大模型。每一步都有实验。

[![在线阅读](https://img.shields.io/badge/在线阅读-laughan.github.io%2Fscratch2modern-9a3b26?style=flat)](https://laughan.github.io/scratch2modern/) · ![网站](https://img.shields.io/badge/网站-写到_1.08-3d63a8?style=flat) · ![实验](https://img.shields.io/badge/实验-第二章_2.01-3d63a8?style=flat) · [![起点](https://img.shields.io/badge/起点-Karpathy_microgpt-555?style=flat)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

起点是 Karpathy 的 microgpt：200 行纯 Python，不依赖任何库，从读数据、自动求导、模型，到训练和推理，一个完整的 GPT 都在里面。但它只是最简单的版本：一层，很小，在 CPU 上慢慢跑。

这个项目从它出发，一次只改一样东西，每改一样都做对照实验，看它到底带来了什么。第一章模型不动，只改怎么训练；第二章训练方式不动，只改模型结构，一步步补上主流模型的部件。代码和实验在这个仓库里，讲解写在[网站](https://laughan.github.io/scratch2modern/)上。

## 目录

- [更新](#更新)
- [进度](#进度)
- [项目结构](#项目结构)

## 更新

- **2026-11-18** — 网站上线，第一章写到 1.08。
  - 网站：每节一页，按"为什么讲这一节 → 是什么 → 怎么做 → 结果"来写，公式、关键代码、图都在页面里。目前写完了 1.01 microgpt 到 1.08 数值精度。
  - 实验：第一章 12 节全部做完，最后在完整的 TinyStories 上训练了一个 531 万参数的模型；第二章开始，2.01 换成 FineWeb-Edu、上下文 1024、8 层宽 512，作为第二章的基线。

## 进度

### 第一章：训练

模型结构不动，只改怎么训练。从 microgpt 出发，它用纯 Python 一个数一个数地求导，在 CPU 上跑；然后依次换成 PyTorch 张量、batch、GPU、BPE 分词器、bf16 混合精度和工程加速，中间也比较了优化器、学习率调度和各种训练细节。在同一个 1 层、13.8 万参数的模型上，训练时间从 93.5 秒降到 4.0 秒，验证集 bpb 从 2.52 降到 1.18；算上最开始的纯 Python，吞吐快了约 20 万倍。最后用选定的配置，在完整的 TinyStories 上训练了一个 6 层、531 万参数的模型，bpb 降到 0.589。

<p align="center"><img src="out/ch1/1.11_summary/ladder.png" width="600" alt="第一章的阶梯：训练时间和验证集 bpb"></p>

每个点是一次训练，在上一个点的基础上多改一样东西：① CPU、字符分词 → ② 搬到 GPU → ③ 加大 batch → ④ 换成 bpe1024、模型加宽 → ⑤ bf16 混合精度 → ⑥ 工程加速。

### 第二章：模型架构

训练方式不动，换到一个更大的模型和更多样的数据上（8 层、宽 512、7720 万参数，FineWeb-Edu，上下文 1024），对比位置编码、归一化、MLP、注意力、MoE 等架构上的改进。目前完成了基线。

## 项目结构

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
docs/                   # 网站
```
