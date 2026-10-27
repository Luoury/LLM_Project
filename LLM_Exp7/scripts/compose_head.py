"""Portrait fusion (extension task): paste a front-facing head shot into the
generated mannequin bust and make it look like one photo.

Chain implemented here (each step is timed and saved for the report):
  1. locate the head box from the depth map (connected component of the
     foreground mask inside the upper half of the frame);
  2. soft matte the head photo against its light background, erode 1 px and
     feather 1 px;
  3. premultiply alpha, scale the head so its own box fills the face box,
     paste it;
  4. match the mean/std luminance of the pasted patch to the base image;
  5. multiply in a contact shadow under the chin;
  6. gradient-domain (Poisson) clone blend, mixed with the plain paste by
     --clone;
  7. match the film grain / noise level of the base image.

Usage:
  python scripts/compose_head.py --base output/head_photoreal.png \
      --head output/face_photoreal.png --depth output/head_depth_0001.png \
      --out output/head_fused.png
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def head_box_from_depth(depth, upper_frac=0.75):
    """Largest foreground component whose centroid sits in the upper part."""
    mask = depth > 0.02
    h, w = mask.shape
    mask[int(h * upper_frac):, :] = False
    lab, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError("depth map has no foreground")
    best, best_area = None, 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = len(ys)
        if area > best_area:
            best_area, best = area, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return tuple(int(v) for v in best)


def ellipse_alpha(h, w, cx=0.5, cy=0.50, rx=0.46, ry=0.50, feather=0.10):
    """Soft elliptical matte: keeps the central face, drops hair/background."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - cx) / rx) ** 2 + ((yy / h - cy) / ry) ** 2)
    a = np.clip((1.0 - r) / feather, 0.0, 1.0)
    return a ** 0.9


def white_matting(rgb, tol=0.11):
    """Background-key matte: flood the uniform border colour, keep the subject.

    Runs of near-background pixels not connected to the border (e.g. a bright
    specular spot on the forehead) are kept, then interior holes are filled.
    """
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    near = dist < tol
    lab, n = ndimage.label(near)
    edge = set(lab[0].tolist()) | set(lab[-1].tolist()) | \
        set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    edge.discard(0)
    bgmask = np.isin(lab, list(edge)) if edge else np.zeros_like(near)
    alpha = (~bgmask).astype(np.float32)
    alpha = ndimage.binary_fill_holes(alpha > 0.5).astype(np.float32)
    alpha = ndimage.gaussian_filter(alpha, 1.0)
    return np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)


def alpha_bbox(alpha, thr=0.5):
    ys, xs = np.where(alpha > thr)
    if len(ys) == 0:
        raise RuntimeError("empty matte")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_rgba(color, alpha, size):
    rgba = np.dstack([color, alpha])
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., :3], arr[..., 3]


def paste_premultiplied(base, color, alpha, x, y):
    """alpha-composite a premultiplied patch onto base, return composite+mask."""
    h, w = alpha.shape
    H, W = base.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    sx0, sy0 = x0 - x, y0 - y
    a = alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][..., None]
    c = color[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    out = base.copy()
    out[y0:y1, x0:x1] = c * a + base[y0:y1, x0:x1] * (1.0 - a)
    return out, (y0, y1, x0, x1), a


def luminance(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def poisson_blend(base, src, mask, max_iter=800, tol=1e-4):
    """Solve Laplace(f) = Laplace(src) inside mask, f = base on the border."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return base
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    s = src[y0:y1, x0:x1]
    b = base[y0:y1, x0:x1]
    H, W = m.shape
    idx = -np.ones((H, W), dtype=np.int64)
    idx[m] = np.arange(m.sum())
    n = int(m.sum())

    # right-hand side: laplacian of src, minus boundary values of base
    lap = np.zeros_like(s)
    lap[1:-1, 1:-1] = (4 * s[1:-1, 1:-1] - s[:-2, 1:-1] - s[2:, 1:-1]
                       - s[1:-1, :-2] - s[1:-1, 2:])
    rows, cols, vals = [], [], []
    rhs = lap[m].copy()
    hf, wf = base.shape[:2]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        gy, gx = ys + dy, xs + dx              # global coordinates
        in_img = (gy >= 0) & (gy < hf) & (gx >= 0) & (gx < wf)
        yy, xx = gy - y0, gx - x0              # crop coordinates
        inside = in_img & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        jj = np.full(len(ys), -1, dtype=np.int64)
        jj[inside] = idx[yy[inside], xx[inside]]
        is_in = jj >= 0
        # neighbour inside the mask -> unknown, otherwise it's a Dirichlet edge
        rows.append(np.arange(n)[is_in])
        cols.append(jj[is_in])
        vals.append(np.ones(is_in.sum()))
        out = (~is_in) & in_img            # neighbours outside the image: Neumann
        rhs[out] += base[gy[out], gx[out]]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    diag = -np.asarray(A.sum(axis=1)).ravel()
    A = A + sparse.diags(diag + 1e-6)
    sol = spsolve(A.tocsc(), rhs)
    out = base.copy()
    patch = out[y0:y1, x0:x1]
    patch[m] = np.clip(sol, 0.0, 1.0)
    out[y0:y1, x0:x1] = patch
    return out


def estimate_grain(img):
    gray = luminance(img)
    hp = gray - ndimage.gaussian_filter(gray, 1.0)
    return float(np.std(hp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="generated mannequin bust photo")
    ap.add_argument("--head", required=True, help="head shot on a light background")
    ap.add_argument("--depth", required=True, help="depth map of the bust")
    ap.add_argument("--out", required=True)
    ap.add_argument("--face", default="0.16,0.02,0.85,0.71",
                    help="face box as ratios (left,top,right,chin) of the head box")
    ap.add_argument("--shadow", default="0.13,0.14,0.13,0.06",
                    help="contact-shadow ellipse (cx,cy,rx,ry) as ratios")
    ap.add_argument("--grain", type=float, default=1.0)
    ap.add_argument("--clone", type=float, default=0.6)
    ap.add_argument("--warp-fill", action="store_true")
    ap.add_argument("--head-mask", default=None,
                    help="depth map of the head source; when given its foreground mask "
                         "replaces the white-background matte (alpha source)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    base = load_rgb(args.base)
    depth = load_gray(args.depth)
    head_rgb = load_rgb(args.head)
    H, W = base.shape[:2]
    if head_rgb.shape[:2] != (H, W):
        head_rgb = np.asarray(Image.fromarray((head_rgb * 255).astype(np.uint8))
                              .resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0

    # 1. head box from depth
    hb = head_box_from_depth(depth)
    hx0, hy0, hx1, hy1 = hb
    hw, hh = hx1 - hx0, hy1 - hy0
    fx = [float(v) for v in args.face.split(",")]
    face = (hx0 + fx[0] * hw, hy0 + fx[1] * hh, hx0 + fx[2] * hw, hy0 + fx[3] * hh)
    target_w = face[2] - face[0]
    target_h = face[3] - face[1]
    print(f"[compose] head box {hb}  face box {tuple(round(v,1) for v in face)}")

    # 2. matte + erode + feather
    if args.head_mask:
        m = load_gray(args.head_mask)
        if m.shape != head_rgb.shape[:2]:
            m = np.asarray(Image.open(args.head_mask).convert("L")
                           .resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
        mask = (m > 0.02).astype(np.float32)
        mask = ndimage.binary_fill_holes(mask > 0.5).astype(np.float32)
        mask = ndimage.gaussian_filter(mask, 1.0)
        alpha = mask
        color = np.clip(head_rgb, 0.0, 1.0)
        print("[compose] alpha from depth mask")
    else:
        # a tight portrait has hair/background in its corners, so keying is
        # unreliable: use a feathered ellipse over the central face instead
        color = np.clip(head_rgb, 0.0, 1.0)
        alpha = ellipse_alpha(H, W)
        print("[compose] alpha from feathered ellipse")
    alpha = ndimage.binary_erosion(alpha > 0.45, iterations=1).astype(np.float32)
    alpha = ndimage.gaussian_filter(alpha, 1.0)
    sb = alpha_bbox(alpha)
    sx0, sy0, sx1, sy1 = sb

    # 3. scale head so its own box fills the face box, premultiply, paste
    scale = min(target_w / (sx1 - sx0), target_h / (sy1 - sy0))
    new_w = max(1, int(round((sx1 - sx0) * scale)))
    new_h = max(1, int(round((sy1 - sy0) * scale)))
    if args.warp_fill:
        new_w, new_h = int(round(target_w)), int(round(target_h))
    crop_c = color[sy0:sy1, sx0:sx1]
    crop_a = alpha[sy0:sy1, sx0:sx1]
    color_s, alpha_s = resize_rgba(crop_c, crop_a, (new_w, new_h))

    px = int(round(face[0] - (new_w - target_w) / 2.0))
    py = int(round(face[1] - (new_h - target_h) / 2.0))
    comp, (y0, y1, x0, x1), pa = paste_premultiplied(base, color_s, alpha_s, px, py)

    # 4. brightness / contrast alignment on the pasted patch
    patch = comp[y0:y1, x0:x1]
    m = pa[..., 0] > 0.5
    if m.any():
        lb = luminance(base[y0:y1, x0:x1][m])
        lp = luminance(patch[m])
        gain = np.clip(lb.std() / (lp.std() + 1e-4), 0.75, 1.35)
        off = lb.mean() - gain * lp.mean()
        adj = np.clip(patch * gain + off, 0.0, 1.0)
        # only the pasted pixels are re-graded, otherwise the patch rectangle shows
        a = pa[..., 0:1]
        comp[y0:y1, x0:x1] = adj * a + patch * (1.0 - a)

    # 5. contact shadow under the chin
    sh = [float(v) for v in args.shadow.split(",")]
    cx, cy = hx0 + sh[0] * hw, hy0 + sh[1] * hh
    rx, ry = sh[2] * hw, sh[3] * hh
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    shadow = (1.0 - 0.55 * g)[..., None]
    shade_mask = np.zeros((H, W), dtype=bool)
    shade_mask[int(cy - 1.6 * ry):int(cy + 1.6 * ry), int(cx - 1.6 * rx):int(cx + 1.6 * rx)] = True
    comp[shade_mask] *= shadow[shade_mask]

    # 6. Poisson blend inside the pasted alpha region
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = pa[..., 0] > 0.5
    mask = ndimage.binary_erosion(mask, iterations=2)
    poisson = poisson_blend(base, comp, mask)
    blended = (1.0 - args.clone) * comp + args.clone * poisson
    out = np.where(mask[..., None], blended, comp)

    # 7. grain matching
    g_base, g_out = estimate_grain(base), estimate_grain(out)
    noise = rng.normal(0.0, 1.0, (H, W, 1)).astype(np.float32) * 0.85 + \
        rng.normal(0.0, 1.0, (H, W, 1)).astype(np.float32) * 0.15
    out = np.clip(out + noise * (g_base - g_out) * args.grain * mask[..., None], 0.0, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray((out * 255).astype(np.uint8)).save(args.out)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(
        args.out.replace(".png", "_pasteonly.png"))
    Image.fromarray((alpha * 255).astype(np.uint8)).save(args.out.replace(".png", "_matte.png"))
    meta = {"head_box": hb, "face_box": [round(v, 1) for v in face],
            "paste_box": [x0, y0, x1, y1], "scale": round(float(scale), 4),
            "grain_base": round(g_base, 5), "grain_out": round(g_out, 5),
            "clone": args.clone, "seconds": round(time.time() - t0, 2)}
    with open(args.out.replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
