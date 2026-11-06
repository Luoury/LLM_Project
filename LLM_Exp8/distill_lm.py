# -*- coding: utf-8 -*-
"""
实验八 自制语言模型的知识蒸馏 —— 纯 CPU 可运行实现
对应教材第 15 章《模型量化与压缩技术》+ 第 5 章《Transformer 模型》

单文件、零外部数据：
  * 内置「仿古诗句」合成语料生成器（可复现，不依赖网络与版权）
  * 字符级 GPT（教师 / 学生同架构、不同宽度）
  * 三种训练：教师监督训练、学生从零训练（hard label）、学生蒸馏训练（soft label）
  * 温度 T / 权重 alpha 消融、中间层特征蒸馏、自蒸馏、教师规模扫描、动态量化导出

常用命令：
  python distill_lm.py --mode quick    # 30 秒自检，确认环境没问题
  python distill_lm.py --mode teacher  # 只训练教师并保存 teacher.pt（约 5 分钟）
  python distill_lm.py --mode all      # 教师 -> 从零学生 -> 蒸馏学生 -> 画图
  python distill_lm.py --mode ablate   # 温度 / 权重 4x4 消融
  python distill_lm.py --mode ext      # 拓展实验 A/B/C/D/E
  python distill_lm.py --mode all --feat 0.1   # 加特征蒸馏
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("DISTILL_THREADS", "8")))
torch.manual_seed(0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")

# ============================================================
# 1. 语料：可复现的合成「仿古诗句」
# 四个槽位 + 两个收尾模板，使下一个字的分布具有真实的多样性与结构，
# 这正是教师 soft label 能传给学生的「暗知识」。
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]

POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0):
    """用固定随机种子生成训练语料；留出集换一个种子，检验泛化而非记忆。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s，%s。"
                       % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s道：“%s%s。”%s。"
                       % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
    out.append("\n")
    out.extend(POEMS)
    return "".join(out)


# ============================================================
# 2. 字符级分词
# ============================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============================================================
# 3. 最小 GPT（Pre-Norm，与作业三同构，只改宽度/深度）
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================
# 4. 批采样：scratch 与 kd 使用同一 generator 种子，保证数据顺序一致
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的 top-1 一致率，衡量「学到了多少」"""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot


# ============================================================
# 5. 训练：hard label / soft label（KD）
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag=""):
    """
    alpha = 0 -> 纯交叉熵（从零训练的对照组）
    alpha > 0 -> L = (1-alpha)*CE + alpha*T^2*KL(教师软标签 || 学生)
    feat_weight -> 额外加最后一层隐状态 MSE（FitNet 式特征蒸馏）
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        kd = None
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            # 关键：先 reshape 成 (B*T, V) 再算 KL。
            # 若直接对 (B,T,V) 调用 kl_div(reduction="batchmean")，它只除以 B，
            # 序列维会被重复累加，KL 被放大 T 倍（T=96 时数值大得离谱）。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if kd is not None:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), kd.item())
            print("  [%s] iter %4d/%d loss %.4f%s (%.1fs)"
                  % (tag or "train", it + 1, iters, loss.item(), extra,
                     time.time() - t0), flush=True)
    return hist, time.time() - t0


# ============================================================
# 6. 主流程
# ============================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999)  # 同语法、不同种子 -> 留出集
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def save_json(res, tag):
    path = os.path.join(OUT_DIR, "results_%s.json" % tag)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return path


# ------------------------------------------------------------
# 7. 绘图
# ------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # WSL 的 matplotlib 字体缓存里往往没有 CJK 字体，这里直接按文件注册，
    # 并按实际注册到的族名设置，避免中文标题变成方框。
    fams = ["Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
    for path in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        try:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                fams.insert(0, fm.FontProperties(fname=path).get_name())
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = fams + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_curves(res, path):
    try:
        plt = _plt()
    except Exception as e:
        print("[plot] 跳过绘图（%s）" % e)
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("teacher", "教师 Teacher"), ("scratch", "学生·从零"),
                   ("kd", "学生·蒸馏")):
        if k in res and res[k].get("curve"):
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("训练损失曲线")
    if ax[0].get_lines():
        ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if res.get("ablation"):
        for T in sorted({r["T"] for r in res["ablation"]}):
            xs, ys = [], []
            for r in res["ablation"]:
                if r["T"] == T:
                    xs.append(r["alpha"])
                    ys.append(r["val_loss"])
            ax[1].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.2)
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("val loss")
        ax[1].set_title("温度 / 权重消融")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    elif "scratch" in res and "kd" in res:
        vals = [res["scratch"]["val_loss"], res["kd"]["val_loss"]]
        bars = ax[1].bar(["从零", "蒸馏"], vals, color=["#4c72b0", "#dd8452"], width=0.5)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.004, "%.4f" % v,
                       ha="center", va="bottom", fontsize=9)
        if res.get("teacher"):
            ax[1].axhline(res["teacher"]["val_loss"], ls="--", lw=1.0,
                          color="#55a868", label="教师 %.4f" % res["teacher"]["val_loss"])
            ax[1].legend(fontsize=8)
        lo, hi = min(vals), max(vals + [res.get("teacher", {}).get("val_loss", min(vals))])
        ax[1].set_ylim(lo - 0.08, hi + 0.06)
        ax[1].set_ylabel("留出集 loss")
        ax[1].set_title("留出集 loss（越低越好）")
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[plot] saved %s" % path)


def plot_heatmap(rows, path):
    try:
        plt = _plt()
        import numpy as np
    except Exception as e:
        print("[plot] 跳过热力图（%s）" % e)
        return
    Ts = sorted({r["T"] for r in rows})
    As = sorted({r["alpha"] for r in rows})
    grid = np.zeros((len(Ts), len(As)))
    for i, T in enumerate(Ts):
        for j, a in enumerate(As):
            grid[i, j] = [r["val_loss"] for r in rows
                          if r["T"] == T and r["alpha"] == a][0]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(As)))
    ax.set_xticklabels(["α=%.1f" % a for a in As])
    ax.set_yticks(range(len(Ts)))
    ax.set_yticklabels(["T=%d" % T for T in Ts])
    mid = grid.min() + 0.5 * (grid.max() - grid.min())
    for i in range(len(Ts)):
        for j in range(len(As)):
            ax.text(j, i, "%.4f" % grid[i, j], ha="center", va="center",
                    color="white" if grid[i, j] > mid else "black", fontsize=8)
    ax.set_title("留出集 loss：T × α（红框为最优）")
    fig.colorbar(im, ax=ax, label="val loss")
    # 用红框标出最优点，报告 7.3 节直接引用
    bi, bj = np.unravel_index(int(grid.argmin()), grid.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.4, zorder=5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[plot] saved %s" % path)


# ------------------------------------------------------------
# 8. 消融
# ------------------------------------------------------------
def plot_abl_lines(rows, path):
    """4 条 T 曲线：横轴 alpha，纵轴留出集 loss。"""
    try:
        plt = _plt()
    except Exception as e:
        print("[plot] 跳过绘图（%s）" % e)
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for T in sorted({r["T"] for r in rows}):
        xs = [r["alpha"] for r in rows if r["T"] == T]
        ys = [r["val_loss"] for r in rows if r["T"] == T]
        ax.plot(xs, ys, marker="o", linewidth=1.4, label="T=%d" % T)
    ax.set_xlabel("alpha（软标签权重）")
    ax.set_ylabel("留出集 loss")
    ax.set_title("温度 T 与权重 alpha 的二维消融")
    ax.legend(fontsize=8, title="温度")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[plot] saved %s" % path)


def ablate(args, tok, train_ids, teacher, val_ids, bs, block, res):
    """T x alpha 网格扫描。

    与主实验完全同协议（同样的初始化种子、数据顺序、步数与优化器），
    只改 T 和 alpha，因此表 6 中 (T=4, alpha=0.5) 这一格应当复现表 5 的蒸馏结果。
    """
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]

    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            t0 = time.time()
            train(m, train_ids, args.abl_iters, bs, block, lr=args.lr,
                  seed=args.seed, teacher=teacher, T=T, alpha=a,
                  log_every=args.abl_iters, tag="T%d-a%.1f" % (T, a))
            v = evaluate(m, val_ids, bs, block)
            ag = agreement(m, teacher, val_ids, bs, block)
            print("  T=%d alpha=%.1f -> val %.4f agree %.3f (%.1fs)"
                  % (T, a, v, ag, time.time() - t0), flush=True)
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v),
                         "agree": ag})

    res["ablation"] = rows
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f ｜ 最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]))
    plot_heatmap(rows, os.path.join(OUT_DIR, "heatmap.png"))
    plot_abl_lines(rows, os.path.join(OUT_DIR, "ablation.png"))


def plot_teacher_scale(rows, path):
    """左：教师自身 loss 与学生蒸馏后 loss；右：蒸馏相对从零的降幅。"""
    try:
        plt = _plt()
    except Exception as e:
        print("[plot] 跳过教师规模图（%s）" % e)
        return
    xs = [r["C"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(xs, [r["teacher_val"] for r in rows], marker="^", label="教师自身")
    ax[0].plot(xs, [r["kd"] for r in rows], marker="s", label="学生（蒸馏后）")
    ax[0].set_xlabel("教师嵌入维度 C")
    ax[0].set_ylabel("留出集 loss")
    ax[0].set_title("教师越强，学生蒸馏后也越好？")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    bars = ax[1].bar(["C=%d" % r["C"] for r in rows],
                     [r["gain_rel"] for r in rows], width=0.5,
                     color=["#4c72b0", "#dd8452", "#55a868"])
    for b, r in zip(bars, rows):
        ax[1].text(b.get_x() + b.get_width() / 2, r["gain_rel"] + 0.08,
                   "%.2f%%" % r["gain_rel"], ha="center", fontsize=9)
    ax[1].set_ylabel("相对从零的降幅 (%)")
    ax[1].set_title("蒸馏收益 vs 教师规模")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[plot] saved %s" % path)


def plot_steps(rows, path):
    """蒸馏收益随学生训练步数的变化（回答思考题 4）。"""
    try:
        plt = _plt()
    except Exception as e:
        print("[plot] 跳过绘图（%s）" % e)
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs = [r["iters"] for r in rows]
    ax.plot(xs, [r["scratch"] for r in rows], marker="o", label="学生·从零")
    ax.plot(xs, [r["kd"] for r in rows], marker="s", label="学生·蒸馏")
    for r in rows:
        g = 100 * (r["scratch"] - r["kd"]) / r["scratch"]
        ax.annotate("%+.1f%%" % g, (r["iters"], r["kd"]), textcoords="offset points",
                    xytext=(0, 10 if g < 0 else -16), ha="center", fontsize=8,
                    color="#c44e52")
    ax.set_xlabel("学生训练步数")
    ax.set_ylabel("留出集 loss")
    ax.set_title("蒸馏收益随训练步数的变化（标注为相对从零的降幅）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[plot] saved %s" % path)


def steps_sweep(args, tok, train_ids, teacher, val_ids, bs, block, res):
    """学生步数扫描：每个步数都跑一组从零、一组蒸馏，用来看收益出现在哪个区间。"""
    rows = []
    for it in (60, 120, 200, 400):
        torch.manual_seed(args.seed + 7)
        m0 = GPT(tok.vocab_size, STUDENT_CFG)
        train(m0, train_ids, it, bs, block, lr=args.lr, seed=args.seed,
              log_every=it, tag="scratch-%d" % it)
        v0 = evaluate(m0, val_ids, bs, block)
        torch.manual_seed(args.seed + 7)
        m1 = GPT(tok.vocab_size, STUDENT_CFG)
        train(m1, train_ids, it, bs, block, lr=args.lr, seed=args.seed,
              teacher=teacher, T=args.T, alpha=args.alpha,
              log_every=it, tag="kd-%d" % it)
        v1 = evaluate(m1, val_ids, bs, block)
        rows.append({"iters": it, "scratch": v0, "kd": v1,
                     "gain_rel": 100 * (v0 - v1) / v0,
                     "agree_scratch": agreement(m0, teacher, val_ids, bs, block),
                     "agree_kd": agreement(m1, teacher, val_ids, bs, block)})
        print("[steps %3d] 从零 %.4f 蒸馏 %.4f 收益 %.1f%%"
              % (it, v0, v1, rows[-1]["gain_rel"]), flush=True)
    res["steps"] = rows
    plot_steps(rows, os.path.join(OUT_DIR, "steps.png"))
    save_json(res, "steps")


# ------------------------------------------------------------
# 9. 主流程
# ------------------------------------------------------------
def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d train_chars=%d val_chars=%d block=%d batch=%d threads=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids), block, bs,
             torch.get_num_threads()), flush=True)

    teacher_path = os.path.join(OUT_DIR, "teacher.pt")
    res = {"vocab": tok.vocab_size, "block": block, "batch": bs,
           "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)),
           "threads": torch.get_num_threads(),
           "torch": torch.__version__}

    # ---------- (1) 教师：已训练过就直接复用，避免重复等待 ----------
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, TEACHER_CFG)
    print("[teacher] params = %.3f M" % (teacher.n_param / 1e6), flush=True)
    if os.path.exists(teacher_path) and not args.retrain and args.mode != "teacher":
        teacher.load_state_dict(torch.load(teacher_path))
        tv = evaluate(teacher, val_ids, bs, block)
        # 复用 teacher.pt 时训练曲线不在内存里，从 results_teacher.json 里捡回来，
        # 这样图 1 的教师曲线不会缺一段。
        th, ttime = [], 0.0
        tjson = os.path.join(OUT_DIR, "results_teacher.json")
        if os.path.exists(tjson):
            try:
                with open(tjson, encoding="utf-8") as f:
                    tj = json.load(f)
                th = tj.get("teacher", {}).get("curve", []) or []
                ttime = tj.get("teacher", {}).get("time_s", 0.0)
            except Exception:
                pass
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": ttime, "curve": th,
                          "cached": True}
        print("[teacher] 复用 %s val_loss=%.4f ppl=%.2f"
              % (os.path.basename(teacher_path), tv, math.exp(tv)), flush=True)
    else:
        th, ttime = train(teacher, train_ids, args.teacher_iters, bs, block,
                          lr=args.lr, seed=args.seed, log_every=args.log_every,
                          tag="teacher")
        tv = evaluate(teacher, val_ids, bs, block)
        torch.save(teacher.state_dict(), teacher_path)
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": ttime, "curve": th}
        print("[teacher] val_loss=%.4f ppl=%.2f %.1fs"
              % (tv, math.exp(tv), ttime), flush=True)

    if args.mode == "teacher":
        save_json(res, "teacher")
        return

    if args.mode == "ablate":
        ablate(args, tok, train_ids, teacher, val_ids, bs, block, res)
        save_json(res, "ablate")
        print("\n[done] 结果已写入 %s" % OUT_DIR)
        return

    if args.mode == "ext":
        run_ext(args, tok, train_ids, val_ids, teacher, bs, block, res)
        return

    if args.mode == "steps":
        steps_sweep(args, tok, train_ids, teacher, val_ids, bs, block, res)
        print("\n[done] 结果已写入 %s" % OUT_DIR)
        return

    # ---------- (2) 学生配置 ----------
    torch.manual_seed(args.seed + 7)
    probe = GPT(tok.vocab_size, STUDENT_CFG)
    sp = probe.n_param
    print("[student] params = %.3f M (压缩率 %.1fx)"
          % (sp / 1e6, teacher.n_param / sp), flush=True)

    def new_student():
        torch.manual_seed(args.seed + 7)  # 同初始化，保证公平对比
        return GPT(tok.vocab_size, STUDENT_CFG)

    def eval_student(m, hist, tcost, name, **extra):
        v = evaluate(m, val_ids, bs, block)
        row = {"params": sp, "val_loss": v, "ppl": math.exp(v), "time_s": tcost,
               "curve": hist, "agree": agreement(m, teacher, val_ids, bs, block)}
        row.update(extra)
        print("[%s] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs"
              % (name, v, math.exp(v), row["agree"], tcost), flush=True)
        res[name] = row
        return row

    # ---------- (3) 从零训练（hard label 对照组） ----------
    s0 = new_student()
    h0, t0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, log_every=args.log_every, tag="scratch")
    row0 = eval_student(s0, h0, t0, "scratch", T=None, alpha=0.0)

    # ---------- (4) 蒸馏训练（soft label） ----------
    s1 = new_student()
    h1, t1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                   feat_weight=args.feat, log_every=args.log_every, tag="kd")
    row1 = eval_student(s1, h1, t1, "kd", T=args.T, alpha=args.alpha, feat=args.feat)

    print("\n[对比] 从零 %.4f -> 蒸馏 %.4f 相对下降 %.1f%%"
          % (row0["val_loss"], row1["val_loss"],
             100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"]),
          flush=True)

    # ---------- (5) 生成样例 ----------
    prompt = "月"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    samples = {}
    for name, m in (("teacher", teacher), ("scratch", s0), ("kd", s1)):
        with torch.no_grad():
            g = m.generate(ids, 60, temperature=0.9, top_k=20)
        s = tok.decode(g[0].tolist())
        samples[name] = s
        print("[generate:%s] %s" % (name, s.replace("\n", " ")), flush=True)
    res["sample"] = samples

    # ---------- (6) 画图 ----------
    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))
    save_json(res, args.mode)
    torch.save(s1.state_dict(), os.path.join(OUT_DIR, "student_kd.pt"))
    torch.save(s0.state_dict(), os.path.join(OUT_DIR, "student_scratch.pt"))
    print("\n[done] 结果与曲线已写入 %s" % OUT_DIR)


# ------------------------------------------------------------
# 10. 拓展实验 A / B / C / D / E
# ------------------------------------------------------------
def run_ext(args, tok, train_ids, val_ids, teacher, bs, block, res):
    V = tok.vocab_size
    ext = {}
    sp = GPT(V, STUDENT_CFG).n_param

    def fresh_student():
        torch.manual_seed(args.seed + 7)
        return GPT(V, STUDENT_CFG)

    # ---------- A 特征蒸馏 FitNet ----------
    print("\n=== A 特征蒸馏（FitNet）===", flush=True)
    feat_rows = []
    for w in (0.0, 0.1):
        s = fresh_student()
        h, cost = train(s, train_ids, args.student_iters, bs, block, lr=args.lr,
                        seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                        feat_weight=w, log_every=100, tag="feat%.1f" % w)
        v = evaluate(s, val_ids, bs, block)
        feat_rows.append({"feat": w, "val_loss": v, "ppl": math.exp(v),
                          "time_s": cost,
                          "agree": agreement(s, teacher, val_ids, bs, block)})
        print("[feat %.1f] val_loss=%.4f ppl=%.2f agree=%.3f"
              % (w, v, math.exp(v), feat_rows[-1]["agree"]), flush=True)
    ext["feat"] = feat_rows

    # ---------- B 热力图（读取消融结果） ----------
    print("\n=== B 温度/权重热力图 ===", flush=True)
    ab_path = os.path.join(OUT_DIR, "results_ablate.json")
    if os.path.exists(ab_path):
        with open(ab_path, encoding="utf-8") as f:
            ab = json.load(f)
        ext["ablation"] = ab.get("ablation", [])
        plot_heatmap(ext["ablation"], os.path.join(OUT_DIR, "heatmap.png"))
        best = min(ext["ablation"], key=lambda r: r["val_loss"])
        ext["ablation_best"] = best
        print("[heatmap] 最优 T=%d alpha=%.1f val=%.4f"
              % (best["T"], best["alpha"], best["val_loss"]), flush=True)
    else:
        print("[heatmap] 缺少 results_ablate.json，跳过", flush=True)

    # ---------- C 自蒸馏 ----------
    print("\n=== C 自蒸馏（教师=训练更久的学生）===", flush=True)
    torch.manual_seed(args.seed + 7)
    big = GPT(V, STUDENT_CFG)
    hb, tbig = train(big, train_ids, 400, bs, block, lr=args.lr, seed=args.seed,
                     log_every=100, tag="selft-teacher")
    vbig = evaluate(big, val_ids, bs, block)
    s = fresh_student()
    hs, ts = train(s, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=big, T=4.0, alpha=0.5,
                   log_every=100, tag="selft-kd")
    vs = evaluate(s, val_ids, bs, block)
    ext["self_distill"] = {
        "teacher_long": {"iters": 400, "val_loss": vbig, "ppl": math.exp(vbig),
                         "time_s": tbig},
        "student_kd": {"iters": args.student_iters, "val_loss": vs,
                       "ppl": math.exp(vs), "time_s": ts,
                       "agree": agreement(s, big, val_ids, bs, block)},
        "gain_vs_long_teacher": vbig - vs,
    }
    print("[self-distill] 长训学生(400) val=%.4f -> 蒸馏学生(120) val=%.4f"
          % (vbig, vs), flush=True)

    # ---------- D 动态量化 ----------
    print("\n=== D 动态量化 int8 ===", flush=True)
    kd_path = os.path.join(OUT_DIR, "student_kd.pt")
    s = fresh_student()
    if os.path.exists(kd_path):
        s.load_state_dict(torch.load(kd_path))
    else:
        train(s, train_ids, args.student_iters, bs, block, lr=args.lr,
              seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
              log_every=100, tag="kd")
    fp32_path = os.path.join(OUT_DIR, "student_kd_fp32.pt")
    torch.save(s.state_dict(), fp32_path)
    q = torch.quantization.quantize_dynamic(s, {nn.Linear}, dtype=torch.qint8)
    int8_path = os.path.join(OUT_DIR, "student_kd_int8.pt")
    torch.save(q.state_dict(), int8_path)
    v32 = evaluate(s, val_ids, bs, block)
    v8 = evaluate(q, val_ids, bs, block)

    def timeit(m, n=5):
        m.eval()
        x = torch.randint(0, V, (1, block))
        with torch.no_grad():
            m(x)
            t0 = time.time()
            for _ in range(n):
                for _ in range(20):
                    m(x)
            return (time.time() - t0) / (n * 20) * 1000

    t32, t8 = timeit(s), timeit(q)
    ext["quant"] = {
        "fp32_bytes": os.path.getsize(fp32_path),
        "int8_bytes": os.path.getsize(int8_path),
        "size_ratio": os.path.getsize(fp32_path) / os.path.getsize(int8_path),
        "fp32_val_loss": v32, "int8_val_loss": v8, "delta": v8 - v32,
        "fp32_forward_ms": t32, "int8_forward_ms": t8,
        "speedup": t32 / t8,
    }
    print("[quant] %.2f MB -> %.2f MB (%.2fx)  val %.4f -> %.4f  %.2f ms -> %.2f ms"
          % (ext["quant"]["fp32_bytes"] / 1e6, ext["quant"]["int8_bytes"] / 1e6,
             ext["quant"]["size_ratio"], v32, v8, t32, t8), flush=True)

    # ---------- E 教师规模扫描 ----------
    print("\n=== E 教师规模扫描 C ∈ {96,160,256} ===", flush=True)
    scale_rows = []
    for C in (96, 160, 256):
        cfg = dict(n_layer=2, n_head=4, n_embd=C, block_size=96, dropout=0.0)
        tpath = os.path.join(OUT_DIR, "teacher_C%d.pt" % C)
        if C == 160:
            t_model = teacher
            tv = res["teacher"]["val_loss"]
            tcost = res["teacher"].get("time_s", 0.0)
        else:
            torch.manual_seed(args.seed)
            t_model = GPT(V, cfg)
            if os.path.exists(tpath):
                t_model.load_state_dict(torch.load(tpath))
                tv = evaluate(t_model, val_ids, bs, block)
                tcost = 0.0
            else:
                _, tcost = train(t_model, train_ids, args.teacher_iters, bs, block,
                                 lr=args.lr, seed=args.seed, log_every=200,
                                 tag="T-C%d" % C)
                tv = evaluate(t_model, val_ids, bs, block)
                torch.save(t_model.state_dict(), tpath)
        # scratch 与 kd 两组学生，同一初始化与数据顺序
        s0 = fresh_student()
        train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
              seed=args.seed, log_every=200, tag="C%d-scratch" % C)
        v0 = evaluate(s0, val_ids, bs, block)
        s1 = fresh_student()
        train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
              seed=args.seed, teacher=t_model, T=args.T, alpha=args.alpha,
              log_every=200, tag="C%d-kd" % C)
        v1 = evaluate(s1, val_ids, bs, block)
        scale_rows.append({"C": C, "teacher_params": t_model.n_param,
                           "teacher_val": tv, "teacher_time_s": tcost,
                           "scratch": v0, "kd": v1,
                           "gain_abs": v0 - v1,
                           "gain_rel": 100 * (v0 - v1) / v0})
        print("[scale C=%d] teacher %.3fM val=%.4f | scratch %.4f kd %.4f 收益 %.2f%%"
              % (C, t_model.n_param / 1e6, tv, v0, v1,
                 100 * (v0 - v1) / v0), flush=True)
    ext["teacher_scale"] = scale_rows
    plot_teacher_scale(scale_rows, os.path.join(OUT_DIR, "teacher_scale.png"))

    res["ext"] = ext
    res["student_params"] = sp
    save_json(res, "ext")
    print("\n[done] 拓展结果已写入 %s" % OUT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate", "steps", "ext"])
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                    help="忽略已缓存的 teacher.pt，重新训练教师")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    if args.mode == "quick":
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "quick")
        args.teacher_iters, args.student_iters, args.abl_iters = 30, 20, 10
        args.batch_size, args.log_every = 8, 10
        args.mode = "all"
    run(args)


if __name__ == "__main__":
    main()
