# -*- coding: utf-8 -*-
"""
诊断脚本：验证 SmolVLM-256M 在本环境下「到底能不能看见图像」。
思路：给它一张只含单一纯色大块的图，问「图的主要颜色是什么」。
如果视觉通路正常，它应能报出颜色；如果连这个都答不对，
说明问题出在视觉张量/处理器，而不是 SFT 学不会定位。
"""
import torch
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}


def make_big(color_name, path):
    """画一张目标色占据大面积区域的图，降低『看不清』的干扰"""
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 216, 216], outline=COLORS[color_name], width=16)
    im.save(path)
    return im


def main():
    from transformers import SmolVLMForConditionalGeneration, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SmolVLMForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(dev)

    for name in COLORS:
        img = make_big(name, f"diag_{name}.png")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "图中矩形边框是什么颜色？只答颜色名词。"}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        for k in ("pixel_values", "pixel_attention_mask"):
            if k in inputs and torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(torch.float32)
        inputs = inputs.to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        ans = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[视觉诊断] 图={name}  模型答={ans!r}  {'√' if name in ans else '×'}")

    # 再加一题：位置题，确认它能否感知「左上/右下」
    img = make_big("红", "diag_pos.png")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "图中彩色方框在图像的左上角还是右下角？"}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    for k in ("pixel_values", "pixel_attention_mask"):
        if k in inputs and torch.is_floating_point(inputs[k]):
            inputs[k] = inputs[k].to(torch.float32)
    inputs = inputs.to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    print("[视觉诊断] 位置题答=",
          processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
