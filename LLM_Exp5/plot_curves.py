# -*- coding: utf-8 -*-
"""生成 SFT loss 曲线和 DPO loss/margin 曲线"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体
font_path = "/home/luosury/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf"
try:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans SC"
except Exception:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open("experiment_results.json", encoding="utf-8"))

# 图1: SFT loss 曲线
fig, ax = plt.subplots(figsize=(8, 4.5))
losses = d["sft_losses"]
ax.plot(range(1, len(losses) + 1), losses, "b-o", markersize=3, linewidth=1.5)
ax.set_xlabel("训练步数 (Step)", fontsize=12)
ax.set_ylabel("SFT Loss (交叉熵)", fontsize=12)
ax.set_title("QLoRA SFT 训练损失曲线 (r=8, 60步, lr=1e-4)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.annotate(f"start: {losses[0]:.2f}", xy=(1, losses[0]), xytext=(8, 8),
            textcoords="offset points", fontsize=10, color="blue")
ax.annotate(f"end: {losses[-1]:.2f}", xy=(len(losses), losses[-1]),
            xytext=(-60, 10), textcoords="offset points", fontsize=10, color="red")
plt.tight_layout()
plt.savefig("sft_loss_curve.png", dpi=150)
print("saved sft_loss_curve.png")

# 图2: DPO loss & margin 曲线
fig, ax1 = plt.subplots(figsize=(8, 4.5))
records = d["dpo_records"]
steps = [r["step"] for r in records]
losses_dpo = [r["loss"] for r in records]
margins = [r["margin"] for r in records]

color1 = "tab:red"
ax1.set_xlabel("训练步数 (Step)", fontsize=12)
ax1.set_ylabel("DPO Loss", color=color1, fontsize=12)
ax1.plot(steps, losses_dpo, "r-o", markersize=4, linewidth=1.5, label="DPO Loss")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.axhline(y=0.693, color="gray", linestyle="--", alpha=0.5, label="random (0.693)")

ax2 = ax1.twinx()
color2 = "tab:blue"
ax2.set_ylabel("Margin", color=color2, fontsize=12)
ax2.plot(steps, margins, "b-s", markersize=4, linewidth=1.5, label="Margin")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.axhline(y=0, color="green", linestyle=":", alpha=0.5)

plt.title("DPO 训练曲线 (β=0.1, 20步, lr=5e-5)", fontsize=13)
fig.tight_layout()
plt.savefig("dpo_loss_margin_curve.png", dpi=150)
print("saved dpo_loss_margin_curve.png")
