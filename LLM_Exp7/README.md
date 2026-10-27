# 实验七 白模转写实：纯 CPU 深度控制生成流水线

用 Blender 从白模（clay）场景导出 depth / normal 控制图，再用 stable-diffusion.cpp
在**纯 CPU**上以 SD-Turbo（Q4_0 GGUF）+ SD2.1-depth ControlNet 生成写实照片，
并做分辨率/步数/CFG/ControlNet 强度的受控对比与人像融合拓展。

## 环境

| 项目 | 本机实测 |
| --- | --- |
| CPU / 内存 | Intel Core i7-14650HX（16 核 24 线程）/ 15.6 GB DDR5 |
| 系统 | Windows 11 专业版 10.0.26200（x64，无独显参与） |
| Blender | 5.1.0（Cycles CPU，Standard 视图变换，64 spp + OIDN） |
| 推理后端 | stable-diffusion.cpp Windows x64 CPU 预编译版，commit be48649 / tag master-1017 |
| 主模型 | `sd_turbo-q4_0.gguf`（1 847 909 248 B，SD2.1 架构，1024 维 cross-attention） |
| ControlNet | `control_v11p_sd21_depth.safetensors` → `-M convert --type f16` → `control_depth_sd21-fp16.gguf`（728 482 784 B） |
| Python | 3.11 + numpy 2.4.6 / Pillow 13.0.0 / scipy 1.17.1（控制图统计与人像融合） |

模型权重与预编译二进制不入库（体积原因），按下面第 1 步自行准备。

## 目录

```
blender/make_demo_scene.py        静物白模场景（球/环/圆柱/倒角立方 + 三点布光）
blender/export_control_maps.py    导出 depth（反相近亮远黑）/ normal / clay
blender/make_head_scene.py        拓展用半身像场景（--variant bust|headonly）
scripts/run_pipeline.py           Blender 导出 → sd-cli 生成，记录分段耗时到 *_run.json
scripts/compose_head.py           人像融合：羽化椭圆 alpha → 预乘 → Poisson → 颗粒匹配
scripts/make_figures.py           拼带中文标题的对比图（fig_control/fig_runs/fig_ablation/fig_ext）
scripts/fill_report.py            读取 *_run.json 填写报告模板并导出 docx
scripts/docx_to_pdf.py            Word COM 转 PDF（仅用于排版检查）
scripts/run_extension.sh          拓展链路一键复现
assets/                           生成的 .blend 场景
output/                           生成图、控制图、*_run.json、拼图
```

## 复现

```bash
# 1. 准备权重与后端（放到 models/ 与 vendor/sdcpp/）
#    主模型  https://hf-mirror.com/samoudi/sd-turbo-gguf/resolve/main/sd_turbo_q4_0.gguf
#    ControlNet https://hf-mirror.com/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_depth.safetensors
#    sd-cli  https://github.com/leejet/stable-diffusion.cpp/releases（Windows x64 CPU 包）
sd-cli.exe -M convert -m models/control_v11p_sd21_depth.safetensors \
    -o models/control_depth_sd21-fp16.gguf --type f16

# 2. 白模场景与控制图
blender -b --factory-startup -P blender/make_demo_scene.py -- --out assets/demo_white_model.blend --res 512
blender -b assets/demo_white_model.blend -P blender/export_control_maps.py -- \
    --outdir output --name demo --res 512 --samples 64

# 3. 生成（A 组基准）
python scripts/run_pipeline.py --name a512_s4 --res 512 --steps 4 --cfg 1.2 \
    --control-strength 0.9 --seed 42

# 4. 对比实验：B 768²/4 步、C 512²/8 步、D ctrl 0.5、E CFG 3.0
python scripts/run_pipeline.py --name b768_s4 --res 768 --steps 4 --seed 42

# 5. 拓展：半身像 + 正脸融合
bash scripts/run_extension.sh

# 6. 拼图与报告
python scripts/make_figures.py
python scripts/fill_report.py
```

## 实测结果（512²，16 线程）

| 运行 | 分辨率 | 步数 | CFG | ctrl | 采样 (s) | VAE (s) | 总耗时 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 512² | 4 | 1.2 | 0.9 | 37.3 | 14.5 | 60.6 |
| B | 768² | 4 | 1.2 | 0.9 | 93.0 | 33.8 | 144.2 |
| C | 512² | 8 | 1.2 | 0.9 | 73.0 | 14.6 | 96.7 |
| D | 512² | 4 | 1.2 | 0.5 | 36.7 | 14.6 | 59.9 |
| E | 512² | 4 | 3.0 | 0.9 | 36.2 | 14.3 | 59.2 |

- 单步耗时：512² 约 9.2 s，768² 约 23.3 s（像素 2.25 倍 → 采样 2.49 倍，注意力随 token 数超线性）。
- ControlNet 开销：无控制基线 5.35 s/步 → 9.33 s/步，**+74%**（远高于指南的 10~20%，因 ControlNet 未量化、以 f16 在 CPU 上跑）。
- 几何：512²/4 步会把圆柱补成开口杯、把圆环孔洞填成灰盘；768² 或 8 步可修正；ControlNet 强度降到 0.5 直接几何崩塌，CFG 调到 3.0 严重过饱和。
- 人像融合：羽化椭圆 alpha + 预乘 + Poisson（clone 0.25）+ 颗粒匹配 → img2img 精修（strength 0.30 / 8 步）后接缝基本消失。
