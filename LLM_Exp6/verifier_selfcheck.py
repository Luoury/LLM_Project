# -*- coding: utf-8 -*-
"""
验证器自检：用 GT 框反推的正确网格去问 verify_cell，
它必须判「通过」；再用明显错的网格问，它应判「不通过」。
这是报告里「验证器是否可信」的直接证据。
"""
import json
from PIL import Image

import hw6_rsi_localize as H


def main():
    gt = H.load_gt()
    QUESTIONS = H.build_questions(gt)
    locs = [q for q in QUESTIONS if q["type"] == "loc"]
    tp = tn = fp = fn = 0
    rows = []
    for q in locs:
        img = Image.open(f"{H.IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        r, c = q["cell"]
        ok, msg = H.verify_cell(img, q, (r, c))
        tp += ok
        rows.append(f"[{q['id']}] GT格({r+1},{c+1}) → 验证器={'通过' if ok else '误判不通过'} | {msg}")
        # 反例：选一个与 GT 不同的格
        wrong = ((r + 1) % H.GRID, (c + 1) % H.GRID)
        bad_ok, bad_msg = H.verify_cell(img, q, wrong)
        if bad_ok:
            fp += 1
            rows.append(f"    反例：错格({wrong[0]+1},{wrong[1]+1}) → 验证器误判通过 ✗ | {bad_msg}")
        else:
            tn += 1
    print("\n".join(rows))
    print(f"\n[验证器自检] GT 网格判定通过 {tp}/{len(locs)}（假阴性 {len(locs)-tp}）；"
          f"错格判定不通过 {tn}/{len(locs)}（假阳性 {fp}）")
    json.dump({"gt_pass": tp, "gt_total": len(locs), "wrong_reject": tn, "wrong_total": len(locs),
               "detail": rows},
              open("verifier_selfcheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
