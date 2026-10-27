"""One-click pipeline: Blender control-map export -> stable-diffusion.cpp (CPU).

Example:
    python scripts/run_pipeline.py --name test512 --res 512 --steps 4 --cfg 1.2

The script records wall-clock time for the Blender export and for the sampler,
parses sd-cli's own timing lines (sampling / VAE decode) and writes a JSON
record next to the generated image: output/<name>_run.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROMPT = ("photorealistic render, studio soft lighting, PBR material, "
                  "preserve original 3d geometry, sharp")
DEFAULT_NEGATIVE = "deformed geometry, perspective changed"


def find_blender():
    env = os.environ.get("BLENDER_EXE")
    if env and os.path.isfile(env):
        return env
    for cand in (
        r"E:\blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    ):
        if os.path.isfile(cand):
            return cand
    exe = shutil.which("blender")
    if exe:
        return exe
    raise SystemExit("Blender not found; set BLENDER_EXE")


def run(cmd, cwd=None, log_path=None):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n\n--- stdout ---\n" + proc.stdout +
                    "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc, dt


def parse_timings(log_text):
    """Pull sd.cpp's own timing lines out of the log."""
    out = {}
    m = re.findall(r"decode_first_stage completed,\s*taking\s*([\d.]+)\s*s", log_text)
    if m:
        out["vae_decode_s"] = float(m[-1])
    m = re.findall(r"encode_first_stage completed,\s*taking\s*([\d.]+)\s*s", log_text)
    if m:
        out["vae_encode_s"] = float(m[-1])
    m = re.findall(r"sampling completed,\s*taking\s*([\d.]+)\s*s", log_text)
    if m:
        out["sampling_s"] = float(m[-1])
    m = re.findall(r"generate_image completed in\s*([\d.]+)\s*s", log_text)
    if m:
        out["generate_image_s"] = float(m[-1])
    m = re.findall(r"step\s+(\d+)/(\d+)\s*-\s*([\d.]+)s", log_text)
    if m:
        out["per_step_s"] = [float(x[2]) for x in m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--control-strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--sampler", default="euler_a")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    ap.add_argument("--init-img", default=None, help="img2img init image (harmonise step)")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--control-image", default=None,
                    help="use an existing depth map instead of rendering a new one")
    ap.add_argument("--blend", default=os.path.join(ROOT, "assets", "demo_white_model.blend"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "sd_turbo-q4_0.gguf"))
    ap.add_argument("--controlnet", default=os.path.join(ROOT, "models", "control_depth_sd21-fp16.gguf"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--skip-blender", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)
    sdcpp = os.path.join(ROOT, "vendor", "sdcpp")
    cli = os.path.join(sdcpp, "sd-cli.exe")
    if not os.path.isfile(cli):
        raise SystemExit(f"sd-cli not found at {cli}")

    record = {"name": args.name, "params": vars(args).copy()}

    # ---------- step 1: Blender control maps ----------
    if args.control_image:
        depth_png = os.path.abspath(args.control_image)
        record["depth_source"] = "reused"
        record["blender_s"] = 0.0
    else:
        blender = find_blender()
        cmd = [blender, "-b", os.path.abspath(args.blend),
               "-P", os.path.join(ROOT, "blender", "export_control_maps.py"),
               "--", "--outdir", outdir, "--name", args.name, "--res", str(args.res)]
        proc, dt = run(cmd, log_path=os.path.join(outdir, f"{args.name}_blender.log"))
        record["blender_s"] = round(dt, 2)
        depth_png = os.path.join(outdir, f"{args.name}_depth_0001.png")
        if not os.path.isfile(depth_png):
            raise SystemExit("depth map missing after Blender export")
        print(f"[pipeline] blender export {dt:.1f}s -> {depth_png}")

    # ---------- step 2: sd-cli ----------
    out_png = os.path.join(outdir, f"{args.name}_photoreal.png")
    cmd = [
        cli,
        "-m", os.path.abspath(args.model),
        "--control-net", os.path.abspath(args.controlnet),
        "--control-image", os.path.abspath(depth_png),
        "--control-strength", str(args.control_strength),
        "-p", args.prompt,
        "-n", args.negative_prompt,
        "-W", str(args.res), "-H", str(args.res),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg),
        "--sampling-method", args.sampler,
        "-s", str(args.seed),
        "-t", str(args.threads),
        "-o", out_png,
        "-v",
    ]
    if args.init_img:
        cmd[-1:-1] = ["-i", os.path.abspath(args.init_img), "--strength", str(args.strength)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=sdcpp, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    total = time.time() - t0
    log_path = os.path.join(outdir, f"{args.name}_sdcli.log")
    log_text = " ".join(cmd) + "\n\n--- stdout ---\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    record["sdcli_total_s"] = round(total, 2)
    record.update(parse_timings(log_text))
    record["output"] = out_png

    if proc.returncode != 0 or not os.path.isfile(out_png):
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
        raise SystemExit(f"sd-cli failed ({proc.returncode})")

    print(f"[pipeline] sd-cli {total:.1f}s -> {out_png}")
    with open(os.path.join(outdir, f"{args.name}_run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
