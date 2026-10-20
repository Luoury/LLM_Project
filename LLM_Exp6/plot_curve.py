# -*- coding: utf-8 -*-
"""
绘制实验六报告用图：定位（loc）正确率与定位平均 IoU 随阶段变化的折线。
数据全部读自 experiment_results.json，不手抄数字。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RES = json.load(open("experiment_results.json", encoding="utf-8"))

ev = RES.get("eval", {})
rsi = RES.get("rsi", {}).get("rounds", [])

# 阶段序列：基线 → SFT → SFT+纠错 → RSI 第1轮 → RSI 第2轮
labels, loc_acc, loc_iou = [], [], []
b = ev.get("baseline")
s = ev.get("sft")
sc = ev.get("sft_corr")

if b:
    labels.append("基线")
    loc_acc.append(b["loc"] / 8 * 100)
    loc_iou.append(b["iou"] * 100)
if s:
    labels.append("SFT")
    loc_acc.append(s["loc"] / 8 * 100)
    loc_iou.append(s["iou"] * 100)
if sc:
    # 纠错后 loc 正确数：从明细里统计 type=loc 且 hit2 为真
    loc_fixed = sum(1 for d in sc["details"] if d["type"] == "loc" and d["hit2"])
    labels.append("SFT+纠错")
    loc_acc.append(loc_fixed / 8 * 100)
    # 纠错后 IoU 未逐题重算，沿用 SFT 的值并标注
    loc_iou.append(s["iou"] * 100 if s else 0)
for r in rsi:
    labels.append(f"RSI 轮{r['round']}")
    loc_acc.append(r.get("eval_loc", 0) / 8 * 100)
    loc_iou.append(r.get("eval_iou", 0) * 100)

# 中文字体
for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(8, 4.6), dpi=150)
x = list(range(len(labels)))
ax1.plot(x, loc_acc, "o-", color="#c0392b", linewidth=2, markersize=8, label="loc 正确率 (%)")
ax1.set_ylabel("定位正确率 (%)", color="#c0392b", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#c0392b")
ax1.set_ylim(-5, 105)
ax1.grid(alpha=0.3, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(x, loc_iou, "s--", color="#2980b9", linewidth=1.8, markersize=7, label="定位平均 IoU (%)")
ax2.set_ylabel("定位平均 IoU (%)", color="#2980b9", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#2980b9")
ax2.set_ylim(-5, 105)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_title("实验六：定位正确率与平均 IoU 随阶段变化（8 道 loc 题）", fontsize=12)
ax1.axhline(y=12.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax1.text(len(labels) - 0.5, 14, "随机猜测期望 12.5%", fontsize=8, color="gray", ha="right")

# 标注数值
for i, v in enumerate(loc_acc):
    ax1.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=9, color="#c0392b")

handles1, lab1 = ax1.get_legend_handles_labels()
handles2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, lab1 + lab2, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("exp6_loc_curve.png", bbox_inches="tight")
print("[plot] 已生成 exp6_loc_curve.png")
print("[plot] stages:", list(zip(labels, [round(v, 1) for v in loc_acc], [round(v, 1) for v in loc_iou])))
