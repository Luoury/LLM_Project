# LLM_Exp8 — 知识蒸馏 · 剪枝 · int8 量化（纯 CPU）

一个小型字符级 GPT 的完整压缩阶梯：**教师 0.74 M → 蒸馏学生 0.15 M（5.0×）
→ 幅度剪枝 → 动态 int8（0.31 MB，相对教师 9.4×）**，外加一个零依赖的网页演示。
全部实验在 CPU（WSL2，24 线程）上完成，教师 200 步 6.7 s，整篇论文的算力不到 5 CPU-分钟。

本目录同时是课程实验八与期末报告（arXiv 风格英文论文）的代码与产物。

## 文件

| 文件 | 说明 |
| --- | --- |
| `distill_lm.py` | 语料、分词器、GPT、KD 训练、T×α 消融、步数扫描、拓展 A–E、绘图 |
| `pipeline.py` | 剪枝 + 微调 + int8 量化 + 学生宽度扫描，写 `results_pipeline.json` |
| `make_deploy.py` | 导出可部署权重、测单/多线程前向延迟与生成速度，写 `results_bench.json` |
| `paper_figs.py` | 论文插图 Fig. 1–4（300 dpi，英文标注） |
| `demo/server.py`、`demo/index.html` | 零依赖演示系统：标准库 `http.server` + 单页前端 |
| `make_paper.py` | 生成期末报告 docx（姓名/学号从 `REPORT_NAME` / `REPORT_ID` 环境变量读） |
| `scripts/fill_report.py` | 生成实验八报告 docx（同上） |
| `figs/` | 论文插图：`fig1_arch` … `fig4_tradeoff`、`fig5_demo_2x` |
| `runs_distill/` | 训练产物：`results_*.json`、`*.png`；`*.pt` 权重不入库 |
| `docs/` | 模板与指南（本地保留，不入库） |

## 复现全部结果

```
python distill_lm.py --mode quick                # 环境自检                (~5 s)
python distill_lm.py --mode teacher --retrain    # 教师 200 步             (~7 s)
python distill_lm.py --mode all                  # 从零 vs 蒸馏            (~16 s)
python distill_lm.py --mode ablate               # 16 组 T×α               (~78 s)
python distill_lm.py --mode steps                # 步数扫描 60/120/200/400 (~42 s)
python distill_lm.py --mode ext                  # 拓展 A–E                (~100 s)
python pipeline.py                               # 剪枝 / int8 / 宽度扫描  (~90 s)
python make_deploy.py                            # 部署基准                (~1 min)
python paper_figs.py                             # 论文插图
python demo/server.py                            # http://localhost:8600
```

`--feat 0.1` 打开 FitNet 特征蒸馏；`DISTILL_THREADS=12` 控制 CPU 线程数。
生成报告：`REPORT_NAME=... REPORT_ID=... python make_paper.py`（期末报告）
与 `REPORT_NAME=... REPORT_ID=... python scripts/fill_report.py`（实验八报告）。

## 主要结果（留出集 5 357 字符）

| 模型 | loss | PPL | 与教师一致率 | 体积 | 1 线程前向 |
| --- | --- | --- | --- | --- | --- |
| 教师 0.737 M / 200 步 | 0.9986 | 2.71 | — | 2.89 MB | 1.66 ms |
| 学生 0.147 M / 120 步，从零 | 1.5060 | 4.51 | 0.619 | 0.64 MB | 0.67 ms |
| 学生 0.147 M / 120 步，蒸馏 (T=4, α=0.5) | 1.4612 | 4.31 | 0.684 | 0.64 MB | 0.67 ms |
| ＋剪枝 20%（含 120 步微调） | 1.3177 | 3.73 | 0.716 | 0.64 MB | 0.66 ms |
| ＋动态 int8 | 1.3184 | 3.74 | 0.715 | **0.31 MB** | 0.76 ms |

- 消融：最优 (T=8, α=0.5) = 1.4133（相对从零 −6.2%），最差 (T=1, α=0.9) = 1.4989；
  默认格 (T=4, α=0.5) 精确复现 1.4612，说明 16 格与主实验同协议。
- 步数扫描：60/120/200/400 步的收益 +1.1%/+3.0%/+0.8%/−7.3%——收益在欠拟合区最大、400 步转负，
  说明软标签同时携带“信息”和“约束”。
- 剪枝必须与**同预算对照组**比：只多训 120 步也有 1.3081，剪 20% 仅比它差 0.0096；
  20% 稀疏在稠密存储下不省体积（CSR 估算 0.73 MB > 0.64 MB）。
- int8 几乎不掉点（+0.0007）但 CPU 上不快（0.76 vs 0.67 ms），量化省的是体积而非算力。
- 教师规模 C=96/160/256：参数 0.295/0.737/1.769 M，学生收益 +3.27%/+2.98%/+3.88%——瓶颈在学生容量。
- 拓展：FitNet 1.4612→1.4166；自蒸馏 120 步得 1.3579。

## 演示系统

```
python demo/server.py --port 8600      # 浏览器打开 http://localhost:8600
```

同一提示词、同一温度/top-k/种子同时喂给三个模型，页面下方给出离线指标与实测单线程延迟。

## 注意事项

- KL 之前必须先把 logits reshape 成 (B·T, V) 再用 `kl_div(reduction="batchmean")`，否则序列维
  会被累加，KL 被放大 block_size 倍。
- 复用 `teacher.pt` 时要单独回填训练曲线（`results_teacher.json`），否则图 1 教师曲线为空。
- matplotlib 在 WSL 里默认找不到 CJK 字体，脚本用 `fm.fontManager.addfont()` 按 .ttc 注册。
- `torch.load` 读本仓库权重需 `weights_only=False`（保存的是 `GPT` 实例）。
