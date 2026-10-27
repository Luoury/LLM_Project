"""Build labelled contact sheets for the report."""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def font(size):
    for f in FONT_CANDIDATES:
        if os.path.isfile(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def label(img, text, pad=6, size=28):
    f = font(size)
    bar = Image.new("RGB", (img.width, size + 2 * pad), "white")
    d = ImageDraw.Draw(bar)
    tw = d.textlength(text, font=f)
    d.text(((img.width - tw) / 2, pad / 2), text, fill="black", font=f)
    out = Image.new("RGB", (img.width, bar.height + img.height), "white")
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar.height))
    return out


def sheet(items, out, gap=14, size=28, cols=None, cell=None):
    imgs = []
    for path, text in items:
        im = Image.open(path).convert("RGB")
        if cell:
            im = im.resize((cell, cell), Image.LANCZOS)
        imgs.append(label(im, text, size=size))
    cols = cols or len(imgs)
    rows = (len(imgs) + cols - 1) // cols
    cw = max(i.width for i in imgs)
    ch = max(i.height for i in imgs)
    W = cols * cw + (cols - 1) * gap
    H = rows * ch + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), "white")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * (cw + gap) + (cw - im.width) // 2,
                          r * (ch + gap) + (ch - im.height) // 2))
    canvas.save(out)
    print("saved", out, canvas.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "output"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    root = args.root
    out = args.outdir or root
    os.makedirs(out, exist_ok=True)

    p = lambda *a: os.path.join(root, *a)

    # 7.1 control maps
    items = []
    for name, text in (("demo_depth_0001.png", "depth（反相，近白远黑）"),
                       ("demo_normal_0001.png", "normal（仅检查朝向）"),
                       ("demo_clay.png", "clay 白模渲染")):
        if os.path.isfile(p(name)):
            items.append((p(name), text))
    if items:
        sheet(items, os.path.join(out, "fig_control.png"), size=26)

    # 7.2 runs
    runs = [("a512_s4", "A：512²/4步/CFG1.2/ctrl0.9"),
            ("c512_s8", "C：512²/8步"),
            ("b768_s4", "B：768²/4步")]
    items = [(p(f"{n}_photoreal.png"), t) for n, t in runs
             if os.path.isfile(p(f"{n}_photoreal.png"))]
    if items:
        sheet(items, os.path.join(out, "fig_runs.png"), size=26, cols=min(3, len(items)), cell=576)

    # strength / cfg comparison
    items = [(p(f"{n}_photoreal.png"), t) for n, t in
             (("a512_s4", "A：ctrl=0.9（基准）"), ("d512_ctrl05", "D：ctrl=0.5"),
              ("e512_cfg3", "E：CFG=3.0")) if os.path.isfile(p(f"{n}_photoreal.png"))]
    if os.path.isfile(p("_nocontrol_baseline.png")):
        items.append((p("_nocontrol_baseline.png"), "对照：不加深度控制"))
    if items:
        sheet(items, os.path.join(out, "fig_ablation.png"), size=24, cols=2)

    # extension
    items = [(p(f"{n}.png"), t) for n, t in
             (("head_photoreal", "① 生成的半身像"),
              ("face_src", "② 融合用的正面头像"),
              ("head_fused_raw", "③ alpha 贴图（未融合）"),
              ("head_fused_photoreal", "④ img2img 精修后")) if os.path.isfile(p(f"{n}.png"))]
    if items:
        sheet(items, os.path.join(out, "fig_ext.png"), size=24, cols=2)


if __name__ == "__main__":
    main()
