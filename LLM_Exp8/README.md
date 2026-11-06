# 实验八　自制语言模型的知识蒸馏（纯 CPU）

用一个小型字符级 GPT 做知识蒸馏：教师 0.737 M 参数，学生 0.147 M（压缩 5.0×），
损失 L = (1−α)·CE + α·T²·KL(教师软标签 ‖ 学生)。全部实验在 CPU 上跑，教师 200 步 6.7 s。

## 文件

| 文件 | 说明 |
| --- | --- |
| `distill_lm.py` | 全部代码：语料、分词器、GPT、KD 训练、消融、拓展、绘图 |
| `scripts/fill_report.py` | 把 `runs_distill/*.json` 与图片填进报告模板（姓名/学号从环境变量读） |
| `runs_distill/` | 训练产物：`results_*.json`、`curves.png`、`heatmap.png`、`ablation.png`、`steps.png`、`teacher_scale.png`（`*.pt` 不入库） |
| `docs/` | 空白模板与指南（本地保留，不入库） |

## 复现

```
python distill_lm.py --mode quick                   # 环境自检（约 5 s）
python distill_lm.py --mode teacher --retrain       # 教师 200 步（6.7 s）
python distill_lm.py --mode all                     # 从零 + 蒸馏，复用 teacher.pt
python distill_lm.py --mode ablate --abl-iters 120  # 16 组 T×α
python distill_lm.py --mode steps                   # 学生步数扫描 60/120/200/400
python distill_lm.py --mode ext                     # 拓展 A~E
```

`--feat 0.1` 打开 FitNet 特征蒸馏；`DISTILL_THREADS=12` 控制 CPU 线程数。

## 主要结果（留出集）

| 设置 | loss | PPL | 与教师一致率 |
| --- | --- | --- | --- |
| 教师 0.737 M / 200 步 | 0.9986 | 2.71 | — |
| 学生 0.147 M / 120 步，从零 | 1.5060 | 4.51 | 0.619 |
| 学生 0.147 M / 120 步，蒸馏 (T=4, α=0.5) | 1.4612 | 4.31 | 0.684 |
| 最优消融 (T=8, α=0.5) | 1.4133 | 4.14 | 0.672 |

- 16 组消融：T=4~8 明显优于 T=1~2；α=0.9 在所有 T 上都变差。
- 步数扫描：60/120/200/400 步的收益为 +1.1%/+3.0%/+0.8%/−7.3%，收益集中在欠拟合区。
- 拓展：FitNet 1.4612→1.4166；自蒸馏 120 步得 1.3579；int8 量化 674 606 B→324 552 B（2.08×），
  loss 只涨 0.0011，但 CPU 上单序列前向反而从 0.77 ms 变 0.89 ms；教师规模 C=96/160/256 的
  收益为 +3.27%/+2.98%/+3.88%。

## 注意事项

- KL 之前必须先把 logits reshape 成 (B·T, V) 再用 `kl_div(reduction="batchmean")`，否则序列维
  会被累加，KL 被放大 block_size 倍。
- matplotlib 在 WSL 里默认找不到 CJK 字体，脚本用 `fm.fontManager.addfont()` 按 .ttc 注册。
