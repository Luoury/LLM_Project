# -*- coding: utf-8 -*-
"""
零依赖演示服务：标准库 http.server + 单页前端。
  cd ~/LLM_Project/LLM_Exp8 && source ~/LLM_Project/LLM_Venv/bin/activate
  python demo/server.py            # 默认 0.0.0.0:8600
浏览器打开 http://localhost:8600

同一提示词与随机种子同时喂给 教师 / 蒸馏学生 / 剪枝+int8 三个模型，
页面同时展示线上实测的延迟与离线评测指标。
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)      # 允许从 demo/ 目录直接启动

import distill_lm as D  # noqa: E402
import pipeline as P    # noqa: E402

RUNS = D.OUT_DIR
BLOCK = D.TEACHER_CFG["block_size"]
STATE = {}


def load_models():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    tok, train_ids, val_ids = D.make_data(BLOCK)

    teacher = D.GPT(tok.vocab_size, D.TEACHER_CFG)
    teacher.load_state_dict(torch.load(os.path.join(RUNS, "teacher.pt")))

    student = D.GPT(tok.vocab_size, D.STUDENT_CFG)
    student.load_state_dict(torch.load(os.path.join(RUNS, "student_kd.pt")))

    tiny_path = os.path.join(RUNS, "student_prune20_int8.pt")
    if os.path.exists(tiny_path):
        tiny = torch.load(tiny_path, weights_only=False)  # 整个量化模块，需要反序列化自定义类
    else:
        pruned = D.GPT(tok.vocab_size, D.STUDENT_CFG)
        pruned.load_state_dict(torch.load(os.path.join(RUNS, "student_kd.pt")))
        masks = P.make_masks(pruned, 0.2)
        P.apply_masks(pruned, masks)
        P.finetune_ce(pruned, train_ids, 120, 24, BLOCK, 0, masks=masks)
        P.apply_masks(pruned, masks)
        tiny = torch.ao.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
        torch.save(tiny, tiny_path)
    for m in (teacher, student, tiny):
        m.eval()

    bench = {}
    bpath = os.path.join(RUNS, "results_bench.json")
    if os.path.exists(bpath):
        with open(bpath, encoding="utf-8") as f:
            bench = json.load(f).get("models", {})

    STATE.update(tok=tok, val_ids=val_ids, bench=bench,
                 teacher=teacher, student=student, tiny=tiny)
    print("[demo] models loaded; vocab=%d" % tok.vocab_size, flush=True)


def quick_latency(model, reps=20):
    """页面上的在线延迟：单线程、20 次取中位数"""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randint(0, 300, (1, BLOCK))
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    return round(statistics.median(ts), 2)


def generate(prompt, temperature, top_k, n_new, seed):
    tok = STATE["tok"]
    ids = tok.encode(prompt)
    if not ids:
        ids = tok.encode("月")
    out = {}
    for key, model in (("teacher", STATE["teacher"]),
                       ("student", STATE["student"]),
                       ("tiny", STATE["tiny"])):
        torch.manual_seed(int(seed))
        t0 = time.perf_counter()
        idx = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            seq = model.generate(idx, max_new_tokens=int(n_new),
                                 temperature=float(temperature), top_k=int(top_k))
        out[key] = dict(text=tok.decode(seq[0].tolist()),
                        ms=round((time.perf_counter() - t0) * 1000, 1),
                        latency=quick_latency(model))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._send(200, json.dumps(dict(
                block=BLOCK, models=STATE["bench"],
                sample=STATE.get("sample", "月")), ensure_ascii=False))
        else:
            self._send(404, json.dumps(dict(error="not found")))

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, json.dumps(dict(error="not found")))
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            res = generate(req.get("prompt", "月"), req.get("temperature", 0.9),
                           req.get("top_k", 20), req.get("n", 60), req.get("seed", 0))
        except Exception as e:  # 演示服务，出错也要把原因回给前端
            self._send(500, json.dumps(dict(error=repr(e)), ensure_ascii=False))
            return
        self._send(200, json.dumps(res, ensure_ascii=False))

    def log_message(self, fmt, *args):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), fmt % args), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    load_models()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[demo] serving on http://%s:%d" % (args.host, args.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
