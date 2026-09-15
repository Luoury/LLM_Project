# -*- coding: utf-8 -*-
"""
实验作业二：ConvLSTM 的算法应用与改进
弹跳小球时空序列预测：用前 K_IN 帧预测下一帧
完整对照实验：基线 + 6 组控制变量 + 最优组合（3 次复测）
"""
import json
import os
import time

# 保证可复现（须在 CUDA 初始化前设置）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# 中文字体（指南要求 Microsoft YaHei；Linux 上注册 Noto Sans SC 避免方框）
from matplotlib import font_manager as fm
_FONT_PATH = os.path.expanduser("~/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf")
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei",
                                   "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =====================================================================
# 1. 数据合成：弹跳小球序列（n_seq 条，每条 T 帧 size×size，半径 r）
# =====================================================================
def make_sequences(n_seq=2200, T=10, size=32, r=2, seed=42):
    """生成弹跳小球序列，返回 (n_seq, T, size, size, 1) float32 数组。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), np.float32)
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
                x = np.clip(x, r, size - r)
            if y < r or y > size - r:
                vy = -vy
                y = np.clip(y, r, size - r)
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
    return seqs[..., None]  # (N, T, H, W, 1)


# 全局数据（所有实验共用同一份划分）
N_SEQ = 2200          # 2000 训练 + 200 测试（无重叠）
TOTAL_FRAMES = 10
DATA = make_sequences(n_seq=N_SEQ, T=TOTAL_FRAMES)
TRAIN_DATA = DATA[:2000]
TEST_DATA = DATA[2000:]
print(f"数据：训练 {TRAIN_DATA.shape[0]} 条 | 测试 {TEST_DATA.shape[0]} 条 | "
      f"每条 {TOTAL_FRAMES} 帧 × 32×32")


def get_data(k_in):
    """按输入帧数 k_in 切分训练/测试张量。"""
    tr_x = torch.tensor(TRAIN_DATA[:, :k_in]).permute(0, 1, 4, 2, 3).to(device)
    tr_y = torch.tensor(TRAIN_DATA[:, k_in]).squeeze(-1).to(device)
    te_x = torch.tensor(TEST_DATA[:, :k_in]).permute(0, 1, 4, 2, 3).to(device)
    te_y = torch.tensor(TEST_DATA[:, k_in]).squeeze(-1).to(device)
    return tr_x, tr_y, te_x, te_y


# =====================================================================
# 2. 模型：ConvLSTMCell / ConvLSTM / FlattenLSTM
# =====================================================================
class ConvLSTMCell(nn.Module):
    """ConvLSTM 细胞：4 个门共用一个卷积（合并实现）。"""

    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    """多层 ConvLSTM：沿时间循环，最后用 Conv2d 从隐藏状态预测下一帧。"""

    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)])
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x):  # x: (B, T, C, H, W)
        b, _, _, hsize, wsize = x.shape
        states = [(torch.zeros(b, c.hid, hsize, wsize, device=x.device),
                   torch.zeros(b, c.hid, hsize, wsize, device=x.device))
                  for c in self.cells]
        for t in range(x.shape[1]):
            xt = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]
        return self.out(states[-1][0]).squeeze(1)  # (B, H, W)


class FlattenLSTM(nn.Module):
    """全连接 LSTM 对照：展平每一帧为 1024 维，过 nn.LSTM，再 Linear 重塑。"""

    def __init__(self, k_in=4, hid=256):
        super().__init__()
        self.k_in = k_in
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hid, batch_first=True)
        self.fc = nn.Linear(hid, 1024)

    def forward(self, x):  # x: (B, T, 1, 32, 32)
        b = x.shape[0]
        x = x.squeeze(2).reshape(b, self.k_in, -1)  # (B, T, 1024)
        out, _ = self.lstm(x)                        # (B, T, hid)
        out = self.fc(out[:, -1, :])                 # (B, 1024)
        return out.reshape(b, 32, 32)


# =====================================================================
# 3. 训练与评估
# =====================================================================
def run(model, train_x, train_y, test_x, test_y,
        loss_fn=None, lr=0.001, epochs=5, bs=64, seed=42):
    """训练并返回 (history, 耗时秒, 最终测试预测)。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loss_fn = loss_fn or nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]

    hist = {"train_loss": [], "test_mse": [], "test_mae": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            mse = (pred - test_y).pow(2).mean().item()
            mae = (pred - test_y).abs().mean().item()
        hist["train_loss"].append(ep_loss)
        hist["test_mse"].append(mse)
        hist["test_mae"].append(mae)
        print(f"    epoch {ep + 1}/{epochs}  train_loss={ep_loss:.5f}  "
              f"test_MSE={mse:.5f}  test_MAE={mae:.5f}")
    return hist, time.time() - t0, pred


# =====================================================================
# 4. 绘图
# =====================================================================
def plot_curve(hist, title, save_path):
    """绘制训练损失与测试 MSE/MAE 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["train_loss"], color="tab:blue", marker="o", ms=3)
    axes[0].set_title(f"{title} — 训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE 损失")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["test_mse"], color="tab:red", marker="o", ms=3, label="MSE")
    axes[1].plot(hist["test_mae"], color="tab:green", marker="s", ms=3, label="MAE")
    axes[1].set_title(f"{title} — 测试误差")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("误差")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")


def plot_prediction(model, test_x, test_y, k_in, save_path, n_samples=3):
    """绘制“输入帧 | 真实帧 | 预测帧”对比图。"""
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
    fig, axes = plt.subplots(n_samples, k_in + 2,
                             figsize=(2.2 * (k_in + 2), 2.4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]
    for s in range(n_samples):
        # 输入帧
        for t in range(k_in):
            ax = axes[s, t]
            ax.imshow(test_x[s, t, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"输入 t-{k_in - t}" if t < k_in - 1 else "输入 t-1",
                         fontsize=9)
            ax.axis("off")
        # 真实帧
        ax = axes[s, k_in]
        ax.imshow(test_y[s].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title("真实 t", fontsize=9)
        ax.axis("off")
        # 预测帧
        ax = axes[s, k_in + 1]
        ax.imshow(pred[s].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        mse = (pred[s] - test_y[s]).pow(2).mean().item()
        ax.set_title(f"预测 t\nMSE={mse:.5f}", fontsize=9)
        ax.axis("off")
    plt.suptitle("输入帧 | 真实下一帧 | 预测帧 对比", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图片已保存: {save_path}")


# =====================================================================
# 5. 实验配置（控制变量：相对基线只改一处）
# =====================================================================
# 基线：1 层，C_h=32，K=3，看 4 帧，MSE
BASELINE = dict(model_cls=ConvLSTM, model_kw=dict(in_ch=1, hid=32, k=3, layers=1),
                k_in=4, loss_fn=nn.MSELoss(), lr=0.001, epochs=5, bs=64)

EXPERIMENTS = [
    # key, 名称, 模型类, 模型参数, 输入帧数, 损失, 图片名, 改动说明
    ("baseline", "基线 Baseline",
     ConvLSTM, dict(in_ch=1, hid=32, k=3, layers=1), 4, nn.MSELoss(),
     "result_baseline.png", "—"),
    ("exp1", "实验1 加深层数（2 层）",
     ConvLSTM, dict(in_ch=1, hid=32, k=3, layers=2), 4, nn.MSELoss(),
     "result_exp1.png", "1 层 → 2 层 ConvLSTM"),
    ("exp2", "实验2 隐藏通道 32→64",
     ConvLSTM, dict(in_ch=1, hid=64, k=3, layers=1), 4, nn.MSELoss(),
     "result_exp2.png", "C_h 32 → 64"),
    ("exp3", "实验3 卷积核 K=3→5",
     ConvLSTM, dict(in_ch=1, hid=32, k=5, layers=1), 4, nn.MSELoss(),
     "result_exp3.png", "K 3 → 5"),
    ("exp4", "实验4 输入帧数 4→8",
     ConvLSTM, dict(in_ch=1, hid=32, k=3, layers=1), 8, nn.MSELoss(),
     "result_exp4.png", "看 4 帧 → 看 8 帧"),
    ("exp5", "实验5 结构对照（全连接 LSTM）",
     FlattenLSTM, dict(k_in=4, hid=256), 4, nn.MSELoss(),
     "result_exp5.png", "ConvLSTM → 全连接 LSTM"),
    ("exp6", "实验6 损失函数 MSE→L1",
     ConvLSTM, dict(in_ch=1, hid=32, k=3, layers=1), 4, nn.L1Loss(),
     "result_exp6.png", "MSE → L1(MAE)"),
]

# 最优组合：综合有效结构改进（2 层 + 64 通道 + K=5 + 8 帧 + MSE）
BEST_CFG = dict(model_cls=ConvLSTM,
                model_kw=dict(in_ch=1, hid=64, k=5, layers=2),
                k_in=8, loss_fn=nn.MSELoss(), lr=0.001, epochs=5, bs=64)
BEST_SEEDS = [42, 43, 44]


def build_and_run(cfg, seed=42, save_curve=None, plot_pred=False):
    """构建模型并运行一次实验。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = cfg["model_cls"](**cfg["model_kw"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_x, train_y, test_x, test_y = get_data(cfg["k_in"])
    hist, dt, pred = run(model, train_x, train_y, test_x, test_y,
                         loss_fn=cfg["loss_fn"], lr=cfg["lr"],
                         epochs=cfg["epochs"], bs=cfg["bs"], seed=seed)
    final_mse = hist["test_mse"][-1]
    final_mae = hist["test_mae"][-1]

    if save_curve:
        plot_curve(hist, cfg.get("name", ""), save_curve)
    if plot_pred:
        plot_prediction(model, test_x, test_y, cfg["k_in"], "result_pred.png")

    return dict(n_params=n_params, mse=final_mse, mae=final_mae,
                time=dt, hist=hist)


# =====================================================================
# 6. 主流程
# =====================================================================
RESULTS_JSON = "experiment_results.json"


def save_results(base_params, results):
    """增量保存结果，防止中途崩溃丢失数据。"""
    out = dict(device=str(device), baseline_params=base_params, results=results)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_results():
    """读取已有结果用于断点续跑。"""
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print(f"\n设备: {device} | 训练 2000 条 | 测试 200 条\n", flush=True)

    # ---- 基线参数量核对 ----
    torch.manual_seed(42)
    base_model = ConvLSTM(in_ch=1, hid=32, k=3, layers=1)
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[参数量核对] 基线 ConvLSTM 总参数量: {base_params}")
    print(f"  细胞 Conv2d(33,128,3): weight=33*128*9={33*128*9} + bias=128 = {33*128*9+128}")
    print(f"  输出 Conv2d(32,1,3): weight=32*1*9={32*1*9} + bias=1 = {32*9+1}")
    print(f"  合计: {33*128*9+128} + {32*9+1} = {33*128*9+128+32*9+1}")
    print(flush=True)

    # ---- 断点续跑：加载已有结果 ----
    existing = load_results()
    results = existing["results"] if existing else []
    done_keys = {r["key"] for r in results}
    if results:
        print(f"[断点续跑] 已完成: {sorted(done_keys)}", flush=True)

    # ---- 基线 + 6 组对照 ----
    for key, name, mcls, mkw, k_in, loss, img, change in EXPERIMENTS:
        if key in done_keys:
            print(f"[跳过] {name}（已完成）", flush=True)
            continue
        print(f"\n>>> {name}（{change}）", flush=True)
        cfg = dict(model_cls=mcls, model_kw=mkw, k_in=k_in, loss_fn=loss,
                   lr=0.001, epochs=5, bs=64, name=name)
        res = build_and_run(cfg, seed=42, save_curve=img)
        print(f"    最终 MSE={res['mse']:.5f}  MAE={res['mae']:.5f}  "
              f"耗时={res['time']:.1f}s  参数量={res['n_params']}", flush=True)
        results.append(dict(key=key, name=name, change=change, **res))
        save_results(base_params, results)  # 每组完成立即落盘

    # ---- 最优组合：3 次复测 ----
    if "best" not in done_keys:
        print("\n>>> 最优组合（综合有效改进，3 次复测）", flush=True)
        best_runs = []
        best_cfg = dict(BEST_CFG)
        best_cfg["name"] = "最优组合"
        for i, seed in enumerate(BEST_SEEDS, start=1):
            print(f"  --- 第 {i} 次 (seed={seed}) ---", flush=True)
            r = build_and_run(best_cfg, seed=seed,
                              save_curve=f"result_best_run{i}.png",
                              plot_pred=(i == 1))
            print(f"    MSE={r['mse']:.5f}  MAE={r['mae']:.5f}  耗时={r['time']:.1f}s",
                  flush=True)
            best_runs.append(dict(seed=seed, **r))
            # 每次复测后也落盘（标记 best 未完成，用临时 key 保存）
            tmp_results = [r for r in results if r["key"] != "best_partial"]
            tmp_results.append(dict(key="best_partial", runs=best_runs))
            save_results(base_params, tmp_results)
        avg_mse = float(np.mean([r["mse"] for r in best_runs]))
        avg_mae = float(np.mean([r["mae"] for r in best_runs]))
        avg_time = float(np.mean([r["time"] for r in best_runs]))
        print(f"    3 次平均: MSE={avg_mse:.5f}  MAE={avg_mae:.5f}  耗时={avg_time:.1f}s",
              flush=True)
        # 移除临时 key，写入正式 best
        results = [r for r in results if r["key"] != "best_partial"]
        results.append(dict(key="best", name="最优组合",
                            change="2层+64通道+K=5+8帧",
                            mse=avg_mse, mae=avg_mae, time=avg_time,
                            n_params=best_runs[0]["n_params"],
                            runs=best_runs, hist=best_runs[-1]["hist"]))
        save_results(base_params, results)
    else:
        print("[跳过] 最优组合（已完成）", flush=True)

    # ---- 保存最终结果 ----
    save_results(base_params, results)
    print("\n全部实验完成，结果已写入 experiment_results.json", flush=True)


if __name__ == "__main__":
    main()
