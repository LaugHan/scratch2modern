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

✅ 已完成 · 🚧 进行中 · ⬜ 计划中。点节号看网站上的讲解，点"代码"看这一节的代码和 notes。

### 第一章：训练

模型不动，只改怎么训练。

| 节 | 主题 | 这一节回答什么 | 实验 | 网站 |
|---|---|---|---|---|
| [1.01](https://laughan.github.io/scratch2modern/1.01.html) | microgpt | 200 行纯 Python 的 GPT 是怎么工作的 | ✅ [代码](src/ch1/1.01_microgpt) | ✅ |
| [1.02](https://laughan.github.io/scratch2modern/1.02.html) | PyTorch | 换成张量运算，为什么能快几十倍 | ✅ [代码](src/ch1/1.02_torch) | ✅ |
| [1.03](https://laughan.github.io/scratch2modern/1.03.html) | Batch | 一次喂多条数据，速度和效果各怎么变 | ✅ [代码](src/ch1/1.03_batch) | ✅ |
| [1.04](https://laughan.github.io/scratch2modern/1.04.html) | GPU | 搬到 GPU 上快多少，瓶颈在哪里 | ✅ [代码](src/ch1/1.04_cuda) | ✅ |
| [1.05](https://laughan.github.io/scratch2modern/1.05.html) | Tokenizer | 字符、字节、BPE，怎么切 token 最好 | ✅ [代码](src/ch1/1.05_tokenizer) | ✅ |
| [1.06](https://laughan.github.io/scratch2modern/1.06.html) | 优化器 | 从 SGD 到 Muon，各自解决了什么问题 | ✅ [代码](src/ch1/1.06_optimizer) | ✅ |
| [1.07](https://laughan.github.io/scratch2modern/1.07.html) | 学习率调度 | 为什么要衰减，形状和 warmup 重不重要 | ✅ [代码](src/ch1/1.07_schedule) | ✅ |
| [1.08](https://laughan.github.io/scratch2modern/1.08.html) | 数值精度 | fp64 到 fp8，位数少了会损失什么、快多少 | ✅ [代码](src/ch1/1.08_precision) | ✅ |
| 1.09 | 工程加速 | 数学不变，SDPA、compile、CUDA Graph 等一项项叠上去能快多少 | ✅ [代码](src/ch1/1.09_engineering) | ⬜ |
| 1.10 | 训练细节 | 初始化、weight decay、梯度裁剪、dropout 有没有用 | ✅ [代码](src/ch1/1.10_details) | ⬜ |
| 1.11 | 回顾 | 把前面串起来，每一步各贡献了多少 | ✅ [代码](src/ch1/1.11_summary) | ⬜ |
| 1.12 | 完整训练 | 用选定的配置，在完整的 TinyStories 上能训练到什么程度 | ✅ [代码](src/ch1/1.12_full) | ⬜ |

### 第二章：模型架构

训练方式不动，只改模型本身，一项项补上主流模型（Llama、Qwen3、DeepSeek-V3 等）的部件。

| 节 | 主题 | 这一节回答什么 | 实验 | 网站 |
|---|---|---|---|---|
| 2.01 | 更大的模型，更多的数据 | 换成多样的数据、更长的上下文、更大的模型，第一章的配置还管不管用 | ✅ [代码](src/ch2/2.01_scale) | ⬜ |
| 2.02 | 位置编码 | 模型怎么知道 token 的先后顺序，能不能外推到更长 | ⬜ | ⬜ |
| 2.03 | 归一化 | norm 放在哪、放几个、用哪种 | ⬜ | ⬜ |
| 2.04 | MLP | 激活函数和门控，值不值得多一个矩阵 | ⬜ | ⬜ |
| 2.05 | 注意力的 K、V | 少存一些 K、V，效果和速度各怎么变 | ⬜ | ⬜ |
| 2.06 | 形状 | 深一点好还是宽一点好；embedding 要不要和输出层共享 | ⬜ | ⬜ |
| 2.07 | MoE | 参数多、每次只用一部分，能不能更好 | ⬜ | ⬜ |
| 2.08 | 注意力的小改动 | 几个防止 logit 失控的小技巧有没有用 | ⬜ | ⬜ |
| 2.09 | 合起来 | 把每节胜出的放在一起，和 2.01 比 | ⬜ | ⬜ |

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
