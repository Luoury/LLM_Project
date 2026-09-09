"""
手写数字分类实验框架（MLP + PyTorch）
数据集：sklearn load_digits (8x8 灰度图, 10 类)
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 画图中文字体设置（Linux 显式注册 Noto Sans SC，避免中文显示为方框）
import os
from matplotlib import font_manager as fm
_font = os.path.expanduser("~/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf")
if os.path.exists(_font):
    fm.fontManager.addfont(_font)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei",
                                   "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 设备：有 GPU 用 GPU，否则 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- 1. 数据加载 -------------------------
def load_data(test_size=0.2, seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0   # 像素 0~16 归一化到 0~1
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # numpy -> torch 张量；标签必须是 long，CrossEntropyLoss 才不报错
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    return X_train, X_test, y_train, y_test


# ------------------------- 2. MLP 模型 -------------------------
class MLP(nn.Module):
    """
    可配置的多层感知机
    :param input_dim:   输入维度
    :param hidden_sizes: 各隐藏层神经元数，如 [64, 32]
    :param num_classes:  输出类别数
    :param activation:   激活函数名: "relu" / "tanh" / "sigmoid"
    :param dropout:      Dropout 概率，0 表示不使用
    :param use_batchnorm: 是否在隐藏层后加 BatchNorm1d
    """

    def __init__(self, input_dim, hidden_sizes, num_classes,
                 activation="relu", dropout=0.0, use_batchnorm=False):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        act_cls = act_map[activation.lower()]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))   # 输出层维度 = 类别数
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------- 3. 训练函数 -------------------------
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=100, lr=0.01, weight_decay=0.0,
                optimizer_type="adam", batch_size=64):
    """
    :param optimizer_type: "sgd" 或 "adam"
    """
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_type}")

    n = X_train.shape[0]
    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            logits = model(xb)                  # 前向传播
            loss = criterion(logits, yb)        # 交叉熵损失
            optimizer.zero_grad()               # 梯度清零
            loss.backward()                     # 反向传播
            optimizer.step()                    # 参数更新
            epoch_loss += loss.item() * xb.size(0)

        train_losses.append(epoch_loss / n)
        test_accs.append(evaluate(model, X_test, y_test))

    return train_losses, test_accs


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


# ------------------------- 4/5. 画图 -------------------------
def plot_curves(train_losses, test_accs, save_path="result_baseline.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(train_losses, color="tab:blue")
    axes[0].set_title("训练损失曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("损失 (Loss)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_accs, color="tab:green")
    axes[1].set_title("测试准确率曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率 (Accuracy)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"曲线已保存: {save_path}")


# ------------------------- 主流程 -------------------------
def main():
    print(f"使用设备: {device}")

    # 1. 数据
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, seed=42)
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 2. 模型（baseline 配置，均可调）
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_sizes=[64, 32],
        num_classes=10,
        activation="relu",
        dropout=0.0,
        use_batchnorm=False,
    )
    print(model)

    # 3. 训练
    start = time.time()
    train_losses, test_accs = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=100,
        lr=0.01,
        weight_decay=0.0,
        optimizer_type="adam",
        batch_size=64,
    )
    total_time = time.time() - start

    # 6. 结果
    final_acc = test_accs[-1]
    print(f"\n最终测试准确率: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print(f"训练总耗时: {total_time:.2f} 秒")

    # 5. 画图
    plot_curves(train_losses, test_accs, save_path="result_baseline.png")


if __name__ == "__main__":
    main()
