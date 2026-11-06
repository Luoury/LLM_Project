# -*- coding: utf-8 -*-
"""
部署与基准脚本：
  1) 由蒸馏学生导出「剪枝 20%」与「剪枝 20% + int8」两个可部署模型；
  2) 用统一口径重测延迟（warmup + 100 次取中位数，单线程 / 8 线程各一遍）；
  3) 结果写 runs_distill/results_bench.json，供演示页面和论文表格使用。

用法：python make_deploy.py
"""
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn

import distill_lm as D
import pipeline as P

RUNS = D.OUT_DIR
BLOCK = D.TEACHER_CFG["block_size"]
BS = 24


def measure(model, threads, reps=100, warmup=20):
    """固定 batch=1、序列长度 block 的前向延迟（毫秒，取中位数）"""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randint(0, 300, (1, BLOCK))
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    ts.sort()
    return dict(median=round(statistics.median(ts), 3),
                p90=round(ts[int(0.9 * len(ts))], 3),
                mean=round(statistics.mean(ts), 3))


def gen_speed(model, n_new=60, threads=8):
    """端到端生成速度：字符/秒（含 tokenizer 之外的纯前向开销）"""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    idx = torch.tensor([[100]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        model.generate(idx.clone(), max_new_tokens=10, temperature=0.9, top_k=20)
        t0 = time.perf_counter()
        model.generate(idx.clone(), max_new_tokens=n_new, temperature=0.9, top_k=20)
        dt = time.perf_counter() - t0
    torch.set_num_threads(old)
    return round(n_new / dt, 1)


def state_mb(model):
    p = os.path.join(RUNS, "_tmp.pt")
    torch.save(model.state_dict() if hasattr(model, "state_dict") else model, p)
    n = os.path.getsize(p)
    os.remove(p)
    return round(n / 1024 / 1024, 3), n


def main():
    torch.manual_seed(0)
    tok, train_ids, val_ids = D.make_data(BLOCK)

    teacher = D.GPT(tok.vocab_size, D.TEACHER_CFG)
    teacher.load_state_dict(torch.load(os.path.join(RUNS, "teacher.pt")))
    teacher.eval()

    student = D.GPT(tok.vocab_size, D.STUDENT_CFG)
    student.load_state_dict(torch.load(os.path.join(RUNS, "student_kd.pt")))
    student.eval()

    # ---- 剪枝 20% + 微调 ----
    pruned = D.GPT(tok.vocab_size, D.STUDENT_CFG)
    pruned.load_state_dict(torch.load(os.path.join(RUNS, "student_kd.pt")))
    masks = P.make_masks(pruned, 0.2)
    P.apply_masks(pruned, masks)
    loss_before = D.evaluate(pruned, val_ids, BS, BLOCK)
    t0 = time.time()
    P.finetune_ce(pruned, train_ids, 120, BS, BLOCK, 0, masks=masks)
    apply_s = time.time() - t0
    P.apply_masks(pruned, masks)
    pruned.eval()
    sp, kept, total = P.sparsity_of(masks)
    print("[prune 20%%] zeroed %.1f%% (%d/%d)  loss %.4f -> %.4f  (%.1f s)"
          % (sp * 100, total - kept, total, loss_before,
             D.evaluate(pruned, val_ids, BS, BLOCK), apply_s), flush=True)

    torch.save(pruned.state_dict(), os.path.join(RUNS, "student_prune20.pt"))

    # ---- 动态 int8 量化 ----
    tiny = torch.ao.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
    tiny.eval()
    torch.save(tiny, os.path.join(RUNS, "student_prune20_int8.pt"))

    models = [("teacher", teacher, "fp32"), ("student_kd", student, "fp32"),
              ("prune20", pruned, "fp32"), ("prune20_int8", tiny, "int8")]
    out = {"block": BLOCK, "batch": 1, "reps": 100, "threads": torch.get_num_threads(),
           "torch": torch.__version__, "models": {}}

    for tag, m, dtype in models:
        loss = float(D.evaluate(m, val_ids, BS, BLOCK))
        mb, nbytes = state_mb(m)
        row = dict(tag=tag, dtype=dtype, loss=round(loss, 4), ppl=round(math.exp(loss), 2),
                   agree=round(float(D.agreement(m, teacher, val_ids, BS, BLOCK)), 4),
                   nbytes=nbytes, mb=mb,
                   lat1=measure(m, 1), lat8=measure(m, 8),
                   gen_cps=gen_speed(m, 60, 8),
                   params=sum(p.numel() for p in m.parameters()))
        out["models"][tag] = row
        print("%-14s loss %.4f ppl %5.2f agree %.3f  %.2f MB  lat1 %.2f ms  lat8 %.2f ms  %.0f ch/s"
              % (tag, row["loss"], row["ppl"], row["agree"], mb,
                 row["lat1"]["median"], row["lat8"]["median"], row["gen_cps"]), flush=True)

    # 剪枝后的稀疏存储估计：非零元素 + 32 bit 索引上界
    nnz = int(kept)
    out["sparse_estimate_mb"] = round((nnz * 4 * 2) / 1024 / 1024, 2)
    out["sparsity"] = round(sp, 4)
    with open(os.path.join(RUNS, "results_bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done] results_bench.json   sparse-format estimate %.2f MB (%.0f%% zeros)"
          % (out["sparse_estimate_mb"], sp * 100))


if __name__ == "__main__":
    main()
