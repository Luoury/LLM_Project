# -*- coding: utf-8 -*-
"""
实验作业三：一键运行全部实验
- 基线：2000 步
- 6 组对照实验（exp1~exp5 各子项，1000 步）
- 采样实验（加载基线模型，5 组 temperature/top_k）
- 汇总结果到 experiment_results.json
"""
import json
import math
import os
import random
import time

import torch

import min_llm as M

device = "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 4))
SEED = 42

# 构建语料与分词器（全局共享）
TEXT = M.build_corpus("corpus_extra.txt")
TOK = M.CharTokenizer(TEXT)
DATA = torch.tensor(TOK.encode(TEXT), dtype=torch.long)
V = TOK.vocab_size
print(f"语料 {len(TEXT)} 字符 | 词表大小 V = {V}")


def train_and_eval(tag, n_embd=128, n_head=4, n_layer=2, block_size=128,
                   lr=1e-3, iters=2000, batch_size=32, use_pos=True,
                   save_model=False, prompts="春,月"):
    """训练一个配置并返回结果字典。"""
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = M.MiniGPT(V, n_embd, n_head, n_layer, block_size, use_pos=use_pos)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"[{tag}] 参数量={n_params:,} | L={n_layer} H={n_head} C={n_embd} "
          f"T={block_size} lr={lr} pos={'on' if use_pos else 'off'} iters={iters}")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(1, iters + 1):
        xb, yb = M.get_batch(DATA, block_size, batch_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0 or step == 1:
            speed = step / (time.time() - t0)
            eta = (iters - step) / speed
            print(f"  step {step:5d}/{iters} | loss {loss.item():.4f} | "
                  f"{speed:.2f} it/s | ETA {eta:.0f}s")
    train_time = time.time() - t0
    final_loss = losses[-1]
    init_loss = sum(losses[:100]) / 100
    print(f"  完成: 最终 loss={final_loss:.4f} | 初始 loss={init_loss:.4f} | "
          f"耗时={train_time:.1f}s ({train_time/60:.1f}min)")

    # 保存 loss 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(losses, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Training Loss ({tag})")
        plt.tight_layout()
        fname = f"loss_curve_{tag}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  已保存 {fname}")
    except Exception as e:
        print(f"  绘图失败: {e}")

    # 保存模型（仅基线）
    if save_model:
        torch.save(model.state_dict(), "baseline_model.pt")
        print(f"  模型已保存: baseline_model.pt")

    # 生成文本
    samples = {}
    gen_lines = []
    for prompt in prompts.split(","):
        prompt = prompt.strip()
        out = model.generate(torch.tensor([TOK.encode(prompt)]),
                             200, 1.0, 20)
        sample = TOK.decode(out[0].tolist())
        samples[prompt] = sample
        gen_lines.append(f"=== 提示词「{prompt}」 ===\n{sample}\n")
    with open(f"generated_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(gen_lines))

    return dict(tag=tag, n_params=n_params, vocab_size=V,
                final_loss=final_loss, init_loss=init_loss,
                train_time=train_time,
                config=dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            block_size=block_size, lr=lr, iters=iters,
                            use_pos=use_pos),
                samples=samples)


def sampling_exp(tag, model, temperature, top_k, prompt="春"):
    """用已有模型做采样实验。"""
    torch.manual_seed(SEED)
    random.seed(SEED)
    out = model.generate(torch.tensor([TOK.encode(prompt)]),
                         200, temperature, top_k if top_k < V else None)
    sample = TOK.decode(out[0].tolist())
    print(f"  [{tag}] temp={temperature} top_k={top_k}: {sample[:60]}...")
    return sample


def main():
    results = {}

    # ========== 基线 ==========
    results["baseline"] = train_and_eval(
        "L2_E128_lr0.001", iters=2000, save_model=True)
    print(f"\n基线参数量: {results['baseline']['n_params']:,}")
    print(f"基线最终 loss: {results['baseline']['final_loss']:.4f}")
    print(f"基线初始 loss: {results['baseline']['init_loss']:.4f} (应≈ln{V}={math.log(V):.2f})")

    # ========== 对照实验（1000 步）==========
    ITERS = 1000

    # exp1: 学习率
    results["exp1a_lr0.01"] = train_and_eval(
        "L2_E128_lr0.01", lr=0.01, iters=ITERS, prompts="春")
    results["exp1b_lr0.0001"] = train_and_eval(
        "L2_E128_lr0.0001", lr=0.0001, iters=ITERS, prompts="春")

    # exp2: 层数
    results["exp2a_L1"] = train_and_eval(
        "L1_E128_lr0.001", n_layer=1, iters=ITERS, prompts="春")
    results["exp2b_L4"] = train_and_eval(
        "L4_E128_lr0.001", n_layer=4, iters=ITERS, prompts="春")

    # exp3: 嵌入维度
    results["exp3a_E64"] = train_and_eval(
        "L2_E64_lr0.001", n_embd=64, n_head=2, iters=ITERS, prompts="春")
    results["exp3b_E256"] = train_and_eval(
        "L2_E256_lr0.001", n_embd=256, n_head=8, iters=ITERS, prompts="春")

    # exp4: 上下文长度
    results["exp4_T32"] = train_and_eval(
        "L2_E128_T32_lr0.001", block_size=32, iters=ITERS, prompts="春")

    # exp5: 位置编码
    results["exp5_no_pos"] = train_and_eval(
        "no_pos", use_pos=False, iters=ITERS, prompts="春")

    # ========== 采样实验（加载基线模型）==========
    print(f"\n{'='*60}")
    print("采样实验（使用基线模型）")
    print(f"{'='*60}")
    base_model = M.MiniGPT(V, 128, 4, 2, 128, use_pos=True)
    base_model.load_state_dict(torch.load("baseline_model.pt", map_location="cpu",
                                          weights_only=True))
    base_model.to(device)

    sampling_results = {}
    sampling_configs = [
        ("s1_base", 1.0, 20),
        ("s2_temp05", 0.5, 20),
        ("s3_temp15", 1.5, 20),
        ("s4_topk5", 1.0, 5),
        ("s5_topk_full", 1.0, V),
    ]
    for st, temp, tk in sampling_configs:
        # 每组生成 3 段
        samples = []
        for _ in range(3):
            s = sampling_exp(st, base_model, temp, tk)
            samples.append(s)
        sampling_results[st] = dict(temperature=temp, top_k=tk, samples=samples)

    # 保存采样对比图
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager as fm
        import matplotlib.pyplot as plt
        # 注册中文字体，避免中文显示为方框
        _font_path = os.path.expanduser(
            "~/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf")
        if os.path.exists(_font_path):
            fm.fontManager.addfont(_font_path)
            plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
        else:
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(len(sampling_configs), 1,
                                 figsize=(12, 2.5 * len(sampling_configs)))
        for i, (st, temp, tk) in enumerate(sampling_configs):
            text = "\n".join(sampling_results[st]["samples"])
            axes[i].text(0.02, 0.95, f"temp={temp}, top_k={tk}\n{text[:200]}...",
                         transform=axes[i].transAxes, fontsize=7, va="top",
                         wrap=True)
            axes[i].set_title(f"temp={temp}, top_k={tk}", fontsize=9)
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig("sampling_comparison.png", dpi=150)
        plt.close()
        print("已保存 sampling_comparison.png")
    except Exception as e:
        print(f"采样对比图保存失败: {e}")

    # 保存采样文本
    with open("generated_sampling.txt", "w", encoding="utf-8") as f:
        for st, temp, tk in sampling_configs:
            f.write(f"=== {st}: temp={temp}, top_k={tk} ===\n")
            for j, s in enumerate(sampling_results[st]["samples"]):
                f.write(f"[段{j+1}] {s}\n\n")
    print("已保存 generated_sampling.txt")

    # ========== 汇总保存 ==========
    out = dict(
        vocab_size=V,
        ln_V=math.log(V),
        baseline=results["baseline"],
        experiments={k: v for k, v in results.items() if k != "baseline"},
        sampling=sampling_results,
    )
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n全部实验完成！结果已保存到 experiment_results.json")
    print(f"基线: params={results['baseline']['n_params']:,} "
          f"loss={results['baseline']['final_loss']:.4f} "
          f"time={results['baseline']['train_time']/60:.1f}min")


if __name__ == "__main__":
    main()
