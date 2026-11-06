# -*- coding: utf-8 -*-
"""
期末报告用：从蒸馏学生出发的「剪枝 + int8 量化」压缩流水线。
复用 distill_lm.py 的模型/数据/评测，不重复实现训练逻辑。

产物：
  runs_distill/results_pipeline.json
  runs_distill/fig_pipeline.png
用法：
  python pipeline.py                       # 全流程（约 1 分钟，纯 CPU）
  python pipeline.py --ratios 0.2 0.4 0.6  # 自定义剪枝率
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn

import distill_lm as D

OUT = D.OUT_DIR


# ------------------------------------------------------------
# 1. 稀疏化工具
# ------------------------------------------------------------
def linear_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and m.weight is not None]


def make_masks(model, ratio):
    """全局幅度剪枝：按 |w| 排序，保留前 (1-ratio) 比例"""
    w = torch.cat([m.weight.detach().abs().flatten() for _, m in linear_layers(model)])
    thresh = torch.quantile(w.float(), ratio)  # 去掉最小的 ratio 比例
    return {name: m.weight.detach().abs() >= thresh for name, m in linear_layers(model)}


def apply_masks(model, masks):
    with torch.no_grad():
        for name, m in linear_layers(model):
            m.weight.mul_(masks[name].to(m.weight.dtype))


def sparsity_of(masks):
    tot = sum(m.numel() for m in masks.values())
    kept = sum(int(m.sum().item()) for m in masks.values())
    return 1.0 - kept / tot, kept, tot


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def finetune_ce(model, train_ids, iters, bs, block, seed, lr=1e-3, masks=None):
    """短程 CE 微调；masks 非空时每步重新施加稀疏掩码。返回耗时（秒）"""
    if iters <= 0:
        return 0.0
    t0 = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 7)
    model.train()
    for _ in range(iters):
        x, y = D.get_batch(train_ids, bs, block, gen)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if masks is not None:
            apply_masks(model, masks)
    return round(time.time() - t0, 1)


# ------------------------------------------------------------
# 2. 计时与体积
# ------------------------------------------------------------
@torch.no_grad()
def latency_ms(model, block, reps=50, warmup=5):
    """单线程单序列前向延迟（ms），取最小值以压低调度抖动"""
    model.eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randint(0, 322, (1, block))
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev)
    model.train()
    return round(min(ts), 3)


def state_bytes(model):
    """序列化后的实际字节数（fp32 模型即 4 B/参数，量化模型更小）"""
    path = os.path.join(OUT, "_size_tmp.pt")
    torch.save(model.state_dict(), path)
    n = os.path.getsize(path)
    os.remove(path)
    return n


# ------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.75])
    ap.add_argument("--finetune", type=int, default=120)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = D.make_data(args.block)
    teacher = D.GPT(tok.vocab_size, D.TEACHER_CFG)
    teacher.load_state_dict(torch.load(os.path.join(OUT, "teacher.pt"), map_location="cpu"))
    base = D.GPT(tok.vocab_size, D.STUDENT_CFG)
    base.load_state_dict(torch.load(os.path.join(OUT, "student_kd.pt"), map_location="cpu"))

    def metrics(model, tag):
        loss = D.evaluate(model, val_ids, args.bs, args.block)
        return dict(tag=tag, params=count_params(model), loss=round(loss, 4),
                    ppl=round(float(torch.tensor(loss).exp()), 2),
                    agree=round(D.agreement(model, teacher, val_ids, args.bs, args.block), 4),
                    latency_ms=latency_ms(model, args.block),
                    nbytes=state_bytes(model))

    res = {"baselines": [], "pruned": [], "pipeline": []}
    res["baselines"].append(metrics(teacher, "teacher"))
    res["baselines"].append(metrics(base, "student_kd"))

    # 对照组：不剪枝、同样再跑 finetune 步，排除「多训练」这一混杂因素
    ctrl = copy.deepcopy(base)
    ctrl_s = finetune_ce(ctrl, train_ids, args.finetune, args.bs, args.block, args.seed)
    ctrl_stat = metrics(ctrl, "ctrl_more_steps")
    ctrl_stat.update(dict(ratio=0.0, sparsity=0.0, ft_iters=args.finetune, ft_s=ctrl_s))
    res["pruned"].append(ctrl_stat)
    print("[control ] no pruning, +%d CE steps -> loss %.4f (%s s)"
          % (args.finetune, ctrl_stat["loss"], ctrl_s))

    # —— 剪枝 + 短程微调（mask 每步重新应用）——
    best = None
    for r in args.ratios:
        model = copy.deepcopy(base)
        masks = make_masks(model, r)
        apply_masks(model, masks)
        sp, kept, tot = sparsity_of(masks)
        loss_before = D.evaluate(model, val_ids, args.bs, args.block)
        ft_s = finetune_ce(model, train_ids, args.finetune, args.bs, args.block,
                           args.seed, masks=masks)
        stat = metrics(model, "prune_%.0f" % (r * 100))
        stat.update(dict(ratio=r, sparsity=round(sp, 4), kept=kept, total=tot,
                         loss_before_prune_ft=round(loss_before, 4),
                         ft_iters=args.finetune, ft_s=ft_s))
        res["pruned"].append(stat)
        print("[prune %2.0f%%] kept %d/%d  loss %.4f -> %.4f  (%s s)"
              % (r * 100, kept, tot, loss_before, stat["loss"], ft_s))
        if best is None or stat["loss"] < best[1]["loss"]:
            best = (r, stat, model, masks)

    # —— 学生宽度扫描：容量—质量曲线（真实减参，不是掩码）；3 个种子取均值 ——
    res["width"] = []
    for w in (32, 48, 64, 80):
        cfg = dict(D.STUDENT_CFG)
        cfg["n_embd"] = w
        runs, t_all = [], 0.0
        for s in (0, 1, 2):
            m = D.GPT(tok.vocab_size, cfg)
            t0 = time.time()
            D.train(m, train_ids, args.finetune, args.bs, args.block,
                    teacher=teacher, T=4.0, alpha=0.5, seed=s, log_every=1000)
            t_all += time.time() - t0
            runs.append(metrics(m, "kd_w%d" % w))
        mean = lambda k: round(sum(r[k] for r in runs) / len(runs), 4)  # noqa: E731
        std = lambda k: round((sum((r[k] - mean(k)) ** 2 for r in runs) / len(runs)) ** 0.5, 4)  # noqa: E731
        row = dict(tag="kd_w%d" % w, width=w, params=runs[0]["params"],
                   loss=mean("loss"), loss_std=std("loss"), ppl=mean("ppl"),
                   agree=mean("agree"), latency_ms=runs[0]["latency_ms"],
                   nbytes=runs[0]["nbytes"], train_s=round(t_all / len(runs), 1),
                   seeds=[r["loss"] for r in runs])
        res["width"].append(row)
        print("[width %2d] params %6d  loss %.4f±%.4f  ppl %.2f  agree %.3f  %.2f MB"
              % (w, row["params"], row["loss"], row["loss_std"], row["ppl"],
                 row["agree"], row["nbytes"] / 1024 / 1024))

    # —— 最优剪枝模型做动态 int8 量化 ——
    ratio, _, model_p, masks = best
    apply_masks(model_p, masks)
    q = torch.ao.quantization.quantize_dynamic(model_p, {nn.Linear}, dtype=torch.qint8)
    loss_q = D.evaluate(q, val_ids, args.bs, args.block)
    res["pipeline"].append(dict(
        tag="prune_%.0f_int8" % (ratio * 100), ratio=ratio,
        params=count_params(model_p), nonzero=best[1]["kept"],
        loss=round(loss_q, 4), ppl=round(float(torch.tensor(loss_q).exp()), 2),
        agree=round(D.agreement(q, teacher, val_ids, args.bs, args.block), 4),
        latency_ms=latency_ms(q, args.block), nbytes=state_bytes(q)))

    # —— 端到端体积比（相对教师 fp32）——
    t_bytes = res["baselines"][0]["nbytes"]
    for group in ("baselines", "pruned", "pipeline", "width"):
        for row in res[group]:
            row["size_mb"] = round(row["nbytes"] / 1024 / 1024, 2)
            row["ratio_vs_teacher"] = round(t_bytes / row["nbytes"], 2)

    with open(os.path.join(OUT, "results_pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for group in ("baselines", "pruned", "pipeline", "width"):
        for row in res[group]:
            print("%-22s loss %.4f ppl %5.2f agree %.3f lat %.3f ms  %.2f MB  x%.2f"
                  % (row["tag"], row["loss"], row["ppl"], row["agree"],
                     row["latency_ms"], row["size_mb"], row["ratio_vs_teacher"]))
    print("[done] saved results_pipeline.json")


if __name__ == "__main__":
    main()
