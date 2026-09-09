"""
实验作业一：使用 TRAE IDE 设计更好的神经网络
对照实验运行脚本（基线 + 8 组改进 + 失败实验 + 学习率扫描）
严格遵循实验指南：load_digits、8:2 分层划分(seed=42)、全批量训练、30 epoch、控制变量
"""
import json
import os
import time

# 保证 GPU 训练结果可复现（须在 CUDA 初始化前设置）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 固定随机算法，保证每组实验在同一台机器上可精确复现
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# 中文字体（指南要求 Microsoft YaHei；Linux 上显式注册 Noto Sans SC 避免中文方框）
from matplotlib import font_manager as fm
_FONT_PATH = os.path.expanduser("~/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf")
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei",
                                   "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. 数据（所有实验共用同一份划分） ----------
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0          # 像素 0~16 归一化到 [0,1]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)   # 8:2 分层抽样
X_tr = torch.tensor(X_tr).to(device)
y_tr = torch.tensor(y_tr, dtype=torch.long).to(device)  # 必须 Long
X_te = torch.tensor(X_te).to(device)
y_te = torch.tensor(y_te, dtype=torch.long).to(device)


# ---------- 2. 可配置 MLP ----------
class MLP(nn.Module):
    def __init__(self, hidden=(32,), act="sigmoid", dropout=0.0, use_bn=False):
        super().__init__()
        act_layer = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
                     "relu": nn.ReLU, "gelu": nn.GELU}[act]
        layers, prev = [], 64
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:                       # 改进点：BatchNorm
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:                  # 改进点：Dropout
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10))   # 输出层 = 类别数 10
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- 3. 训练与评估（全批量） ----------
def run(mcfg, opt_name="sgd", lr=0.1, epochs=30, wd=0.0, seed=42):
    torch.manual_seed(seed)                  # 先固定种子，再构造模型，保证权重初始化可复现
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MLP(**mcfg).to(device)

    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"loss": [], "acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)    # 前向 + 交叉熵损失（全批量）
        loss.backward()                      # 反向传播
        opt.step()                           # 参数更新

        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
    return hist, time.time() - t0


# ---------- 4. 画图 ----------
def plot_hist(hist, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["loss"], color="tab:blue", marker="o", ms=3)
    axes[0].set_title(f"{title} — 训练损失")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("损失 (Loss)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot([a * 100 for a in hist["acc"]], color="tab:green", marker="o", ms=3)
    axes[1].set_title(f"{title} — 测试准确率（最终 {hist['acc'][-1]*100:.2f}%）")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("准确率 (%)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")


# ---------- 5. 实验组定义（控制变量：相对基线只改一处） ----------
EXPERIMENTS = [
    # key, 名称, 模型参数, 训练参数, 图片名
    ("baseline", "基线 Baseline",
     dict(hidden=(32,), act="sigmoid"),
     dict(opt_name="sgd", lr=0.1, epochs=30),
     "result_baseline.png"),
    ("exp1", "实验1 换激活函数 Sigmoid→ReLU",
     dict(hidden=(32,), act="relu"),
     dict(opt_name="sgd", lr=0.1, epochs=30),
     "result_exp1.png"),
    ("exp2", "实验2 加深网络 2层×128（仍Sigmoid）",
     dict(hidden=(128, 128), act="sigmoid"),
     dict(opt_name="sgd", lr=0.1, epochs=30),
     "result_exp2.png"),
    ("exp3", "实验3 换优化器 SGD→Adam(0.01)",
     dict(hidden=(32,), act="sigmoid"),
     dict(opt_name="adam", lr=0.01, epochs=30),
     "result_exp3.png"),
    ("exp4", "实验4 调学习率 SGD lr=0.01",
     dict(hidden=(32,), act="sigmoid"),
     dict(opt_name="sgd", lr=0.01, epochs=30),
     "result_exp4.png"),
    ("exp5", "实验5 加L2正则 weight_decay=1e-4",
     dict(hidden=(32,), act="sigmoid"),
     dict(opt_name="sgd", lr=0.1, epochs=30, wd=1e-4),
     "result_exp5.png"),
    ("exp6", "实验6 加Dropout p=0.2",
     dict(hidden=(32,), act="sigmoid", dropout=0.2),
     dict(opt_name="sgd", lr=0.1, epochs=30),
     "result_exp6.png"),
    ("exp7", "实验7 加BatchNorm",
     dict(hidden=(32,), act="sigmoid", use_bn=True),
     dict(opt_name="sgd", lr=0.1, epochs=30),
     "result_exp7.png"),
]

# 最优组合：ReLU + 2层×128 + BatchNorm + Adam(0.001)
BEST_MODEL = dict(hidden=(128, 128), act="relu", use_bn=True)
BEST_TRAIN = dict(opt_name="adam", lr=0.001, epochs=30)

# 失败实验：基线配置 + lr=10.0
FAIL_TRAIN = dict(opt_name="sgd", lr=10.0, epochs=30)


def main():
    print(f"使用设备: {device} | 训练样本 {X_tr.shape[0]} | 测试样本 {X_te.shape[0]}")
    # CUDA 预热，避免首次调用的上下文初始化时间计入训练耗时
    _ = torch.zeros(1, device=device)
    results = []

    # ---- 基线 + 实验1~7 ----
    for key, name, mcfg, tcfg, img in EXPERIMENTS:
        print(f"\n>>> {name}")
        hist, dt = run(mcfg, **tcfg)
        acc = hist["acc"][-1] * 100
        print(f"    最终测试准确率: {acc:.2f}% | 耗时: {dt:.2f}s")
        plot_hist(hist, name, img)
        results.append(dict(key=key, name=name, model=mcfg, train=tcfg,
                            acc=acc, time=dt,
                            final_loss=hist["loss"][-1],
                            loss_curve=hist["loss"], acc_curve=hist["acc"]))

    # ---- 实验8：最优组合，固定种子复测 3 次 ----
    print("\n>>> 实验8 最优组合（3 次复测）")
    best_runs = []
    for i, seed in enumerate([42, 43, 44], start=1):
        hist, dt = run(BEST_MODEL, seed=seed, **BEST_TRAIN)
        acc = hist["acc"][-1] * 100
        best_runs.append(dict(seed=seed, acc=acc, time=dt))
        print(f"    第{i}次 (seed={seed}): {acc:.2f}% | {dt:.2f}s")
        plot_hist(hist, f"实验8 最优组合 第{i}次(seed={seed})",
                  f"result_exp8_run{i}.png")
    mean_acc = float(np.mean([r["acc"] for r in best_runs]))
    mean_time = float(np.mean([r["time"] for r in best_runs]))
    print(f"    3 次平均准确率: {mean_acc:.2f}% | 平均耗时: {mean_time:.2f}s")
    results.append(dict(key="exp8", name="实验8 最优组合",
                        model=BEST_MODEL, train=BEST_TRAIN,
                        acc=mean_acc, time=mean_time, runs=best_runs,
                        loss_curve=hist["loss"], acc_curve=hist["acc"]))

    # ---- 失败实验：lr=10.0 ----
    print("\n>>> 失败实验：学习率 lr=10.0")
    hist_fail, dt_fail = run(dict(hidden=(32,), act="sigmoid"), **FAIL_TRAIN)
    fail_acc = hist_fail["acc"][-1] * 100
    nan_loss = any(np.isnan(hist_fail["loss"]))
    print(f"    最终测试准确率: {fail_acc:.2f}% | 耗时: {dt_fail:.2f}s"
          f" | 损失出现NaN: {nan_loss}")
    plot_hist(hist_fail, "失败实验 lr=10.0", "result_lr_too_big.png")

    # ---- 附加：学习率扫描 0.1/1.0/5.0/10.0/20.0（加分项） ----
    print("\n>>> 学习率扫描（基线配置）")
    scan = []
    for lr in [0.1, 1.0, 5.0, 10.0, 20.0]:
        h, _ = run(dict(hidden=(32,), act="sigmoid"),
                   opt_name="sgd", lr=lr, epochs=30)
        scan.append(dict(lr=lr, acc=h["acc"][-1] * 100,
                         final_loss=h["loss"][-1],
                         nan=any(np.isnan(h["loss"]))))
        print(f"    lr={lr:<6} 最终准确率: {scan[-1]['acc']:.2f}%"
              f"  最终损失: {scan[-1]['final_loss']}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lrs = [s["lr"] for s in scan]
    accs = [s["acc"] for s in scan]
    ax.plot(lrs, accs, marker="s", color="tab:red")
    for s in scan:
        ax.annotate(f"{s['acc']:.1f}%", (s["lr"], s["acc"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("学习率 lr（对数刻度）")
    ax.set_ylabel("30 epoch 后测试准确率 (%)")
    ax.set_title("学习率扫描：发散临界点观察")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("result_lr_scan.png", dpi=150)
    plt.close()
    print("  图片已保存: result_lr_scan.png")

    # ---- 结果落盘 ----
    out = dict(device=str(device), results=results,
               fail=dict(acc=fail_acc, time=dt_fail, nan=nan_loss,
                         loss_curve=hist_fail["loss"],
                         acc_curve=hist_fail["acc"]),
               lr_scan=scan)
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n全部实验完成，结果已写入 experiment_results.json")


if __name__ == "__main__":
    main()
