# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA 微调 + DPO 偏好对齐 —— 实验作业五
对应教材「参数高效微调」与「偏好对齐 / DPO」相关章节
单文件、开箱即跑：内置 PIL 合成「几何图形」图像 15 张、问答 15 题、
SFT 样本 15 条、偏好对 12 组 + 自拟 5 组。

常用命令：
  python hw5_dpo.py --mode make_data   # 生成 15 张合成图像
  python hw5_dpo.py --mode infer       # 基线推理 15 题
  python hw5_dpo.py --mode sft         # QLoRA SFT（产出 ./sft_adapter）
  python hw5_dpo.py --mode dpo         # 在 SFT 之上做 DPO（产出 ./dpo_adapter）
  python hw5_dpo.py --mode eval        # 三阶段（基线/SFT/SFT+DPO）× 15 题对比
  python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1  # DPO loss 手算复核
"""
import argparse, math, os, random, json, time
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR = "hw5_imgs"
SFT_DIR, DPO_DIR = "sft_adapter", "dpo_adapter"
RESULTS_FILE = "experiment_results.json"

# ============================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],
    "img12": [("红", "方", 1), ("蓝", "方", 1)],
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}

def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")

def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")

# ============================================================
# 2. 问题集：3 类 × 5 题
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

SFT_DATA = [{"img": q["img"], "prompt": q["q"],
             "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
            for q in QUESTIONS]

# 偏好对（12 组内置 + 5 组自拟 = 17 组）
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
    # ===== 自拟 5 组偏好对（任务三）=====
    {"img": "img04", "prompt": "图中有几个红色方形？",
     "chosen": "3 个。", "rejected": "2 个。图中有两个红色方形。", "reason": "数错（off-by-one）"},
    {"img": "img07", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。蓝色的图形是圆形。", "reason": "形状幻觉（把方说成圆）"},
    {"img": "img12", "prompt": "图里有没有圆形？",
     "chosen": "没有。", "rejected": "有，图中有圆形。", "reason": "对不存在的物体硬答「有」"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "嗯，让我仔细看看，图中似乎有一些黄色的图形，可能是圆形也可能是方形，我不太确定具体数量，大概有几个吧。", "reason": "冗长废话型（回避作答）"},
    {"img": "img14", "prompt": "图里有没有蓝色方形？",
     "chosen": "有，3 个蓝色方形。", "rejected": "有。", "reason": "信息不足型（答对但过于简略，缺乏事实细节）"},
]

# ============================================================
# 3. 模型加载：QLoRA（NF4 量化 + LoRA 低秩适配器）
# ============================================================
def load_model(mode="infer"):
    """mode: infer=原精度推理 / train=4bit 量化+LoRA"""
    from transformers import SmolVLMForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if mode == "train":
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
            print("[load] 使用 4-bit NF4 量化")
        except Exception as e:
            print(f"[load] bitsandbytes 不可用，回退 fp16/fp32 + LoRA: {e}")
    model = SmolVLMForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if mode == "train":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model

def build_prompt(processor, q):
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)

@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()

def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对（「没有」须整词出现）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p
    return g in p

# ============================================================
# 4. SFT：只对答案 token 计损失
# ============================================================
def sft_step(model, processor, opt, steps=60, log_every=10):
    dev = next(model.parameters()).device
    model.train()
    losses = []
    step = 0
    ep = 0
    while step < steps:
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA)))
        rnd.shuffle(order)
        for idx in order:
            if step >= steps:
                break
            s = SFT_DATA[idx]
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[load_image(s["img"])], return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[load_image(s["img"])],
                                       return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % log_every == 0 or step == 1:
                print(f"[sft] step {step}/{steps} loss={loss.item():.4f}")
        ep += 1
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}")
    return losses

# ============================================================
# 5. DPO
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)], images=[img],
                     return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()

def dpo_step(model, processor, opt, beta=0.1, steps=40, log_every=5):
    dev = next(model.parameters()).device
    model.train()
    records = []
    step = 0
    ep = 0
    while step < steps:
        rnd = random.Random(ep)
        order = list(range(len(PRE_PAIRS)))
        rnd.shuffle(order)
        for idx in order:
            if step >= steps:
                break
            p = PRE_PAIRS[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with torch.no_grad(), model.disable_adapter():
                ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            records.append({"step": step, "loss": loss.item(), "margin": margin.item()})
            if step % log_every == 0 or step == 1:
                print(f"[dpo] step {step}/{steps} loss={loss.item():.4f} margin={margin.item():.3f}")
        ep += 1
    model.save_pretrained(DPO_DIR)
    print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")
    return records

def dpo_check(arg):
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    loss = math.log(1 + math.exp(-margin)) if margin >= 0 else -margin + math.log(1 + math.exp(margin))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {beta}·[({pw}-{rw})-({pl}-{rl})] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}")

# ============================================================
# 6. 评测
# ============================================================
def evaluate(processor, model, tag, verbose=True):
    if verbose:
        print(f"\n===== {tag} × 15 题 =====")
    results = {}
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        correct = 0
        per_q = []
        for q in qs:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            ok = judge(q, pred)
            if ok:
                correct += 1
            per_q.append({"id": q["id"], "q": q["q"], "gold": q["gold"],
                          "pred": pred, "correct": ok})
            if verbose:
                mark = "√" if ok else "×"
                print(f"  [{q['id']}] {mark} gold={q['gold']:4} pred={pred[:40]}")
        results[tp] = {"correct": correct, "total": len(qs), "per_q": per_q}
        if verbose:
            print(f"{tag:<10}{tp:<8}正确 {correct}/{len(qs)}")
    total = sum(v["correct"] for v in results.values())
    if verbose:
        print(f"{tag:<10}总计 正确 {total}/15")
    results["total"] = total
    return results

# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval", "all"], default="infer")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dpo_check", default=None)
    args = ap.parse_args()

    if args.dpo_check:
        dpo_check(args.dpo_check)
        return

    results = {}
    t0 = time.time()

    if args.mode == "make_data":
        make_images()
        return

    if args.mode == "infer":
        make_images()
        processor, model = load_model("infer")
        for q in QUESTIONS:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            mark = "√" if judge(q, pred) else "×"
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']} A: {pred}（gold: {q['gold']}）")
        return

    if args.mode == "sft":
        make_images()
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        steps = args.steps or 60
        losses = sft_step(model, processor, opt, steps=steps)
        results["sft_losses"] = losses
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return

    if args.mode == "dpo":
        if not os.path.exists(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        make_images()
        processor, model = load_model("train")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model.get_base_model(), SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        steps = args.steps or 40
        records = dpo_step(model, processor, opt, beta=args.beta, steps=steps)
        results["dpo_records"] = records
        existing = {}
        if os.path.exists(RESULTS_FILE):
            try:
                existing = json.load(open(RESULTS_FILE, encoding="utf-8"))
            except Exception:
                pass
        existing.update(results)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return

    if args.mode == "eval":
        make_images()
        processor, model = load_model("infer")
        results["baseline"] = evaluate(processor, model, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
            results["sft"] = evaluate(processor, model, "SFT")
        if os.path.exists(DPO_DIR):
            model = PeftModel.from_pretrained(model, DPO_DIR)
            results["dpo"] = evaluate(processor, model, "SFT+DPO")
        # 合并已有结果（保留 sft_losses / dpo_records）
        existing = {}
        if os.path.exists(RESULTS_FILE):
            try:
                existing = json.load(open(RESULTS_FILE, encoding="utf-8"))
            except Exception:
                pass
        existing.update(results)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] 结果已保存到 {RESULTS_FILE}")
        return

    if args.mode == "all":
        make_images()
        # 基线推理
        processor, model = load_model("infer")
        results["baseline"] = evaluate(processor, model, "基线")
        # SFT
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sft_steps = args.steps or 60
        results["sft_losses"] = sft_step(model, processor, opt, steps=sft_steps)
        # 加载 SFT 适配器评测
        processor, model = load_model("infer")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
        results["sft"] = evaluate(processor, model, "SFT")
        # DPO
        processor, model = load_model("train")
        model = PeftModel.from_pretrained(
            model.get_base_model(), SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        dpo_steps = max(2, (args.steps or 40) // 2)
        results["dpo_records"] = dpo_step(model, processor, opt, beta=args.beta, steps=dpo_steps)
        # 加载 DPO 适配器评测
        processor, model = load_model("infer")
        model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
        model = PeftModel.from_pretrained(model, DPO_DIR)
        results["dpo"] = evaluate(processor, model, "SFT+DPO")
        results["elapsed_sec"] = round(time.time() - t0)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[all] 全部完成，耗时 {results['elapsed_sec']} 秒，结果保存到 {RESULTS_FILE}")
        return

if __name__ == "__main__":
    main()
