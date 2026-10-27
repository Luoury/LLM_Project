#!/usr/bin/env bash
# Extension task: generate a mannequin bust + a face, fuse them, then harmonise.
set -e
cd "$(dirname "$0")/.."

BUST_PROMPT="portrait photograph of a classical mannequin bust, smooth stone sculpture, studio soft lighting, neutral grey background, front view, sharp"
FACE_PROMPT="studio portrait headshot of a young person facing the camera, neutral expression, soft key light, plain light grey background, photorealistic, sharp"

echo "=== 1. bust photoreal (depth-controlled)"
python scripts/run_pipeline.py --name head --blend assets/head_bust.blend --res 512 \
    --steps 4 --cfg 1.2 --control-strength 0.9 --seed 42 --prompt "$BUST_PROMPT"

echo "=== 2. head-only photoreal (source for the face)"
python scripts/run_pipeline.py --name face --blend assets/headonly.blend --res 512 \
    --steps 4 --cfg 1.2 --control-strength 0.9 --seed 7 --prompt "$FACE_PROMPT"

echo "=== 3. matte + paste + Poisson + grain"
python scripts/compose_head.py --base output/head_photoreal.png \
    --head output/face_photoreal.png --depth output/head_depth_0001.png \
    --head-mask output/face_depth_0001.png --out output/head_fused_raw.png \
    --clone 0.6 --grain 1.0

echo "=== 4. img2img harmonisation under the same depth control"
python scripts/run_pipeline.py --name head_fused --skip-blender \
    --control-image output/head_depth_0001.png --init-img output/head_fused_raw.png \
    --strength 0.35 --steps 6 --cfg 1.3 --control-strength 0.7 --seed 42 \
    --prompt "$BUST_PROMPT"

echo "=== done"
