# scratch2modern

从一个 200 行的 GPT 出发，一步步搭出现代大模型。每一步都有实验。
From a 200-line GPT to a modern LLM, one change at a time, every step backed by experiments.

**[在线阅读 →](https://laughan.github.io/scratch2modern/)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero_dark.png">
  <img src="docs/assets/hero_light.png" width="720" alt="第一章：训练时间和验证集 bpb 的变化">
</picture>

## 代码怎么读

每一节是 `src/` 下一个完整、能单独运行的目录，`notes.md` 里记着这一节改了什么、结果如何。相邻两节 diff 一下，就是这一节的全部改动：

```bash
git diff --no-index src/ch1/1.01_microgpt/train.py src/ch1/1.02_torch/train.py
```

实验输出（`log.csv`、`samples.txt`）在 `out/` 下的同名目录。写法规范见 [CONVENTIONS.md](src/CONVENTIONS.md)。

## 进度

第一章（训练）写到 1.08，后面几节在写。第二章（模型结构）换到 8 层、7720 万参数、FineWeb-Edu，目前完成了基线。之后是后训练和 Infra。

---

起点代码来自 Andrej Karpathy 的 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)。MIT 许可。
