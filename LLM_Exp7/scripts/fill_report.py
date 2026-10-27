# -*- coding: utf-8 -*-
"""Fill the experiment-7 report template with measured data and figures."""
import glob
import json
import os

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
HW7 = os.path.dirname(HERE)
OUT = os.path.join(HW7, "output")
TEMPLATE = os.path.join(
    os.path.dirname(HW7), "作业七_白膜转写实纯CPU深度控制流水线_实验报告（空白模板）.docx")
CJK = "宋体"
LATIN = "Times New Roman"


def load_runs():
    runs = {}
    for f in glob.glob(os.path.join(OUT, "*_run.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        runs[d["name"]] = d
    return runs


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------
def style_run(run, size=10.5, bold=False):
    run.font.name = LATIN
    run.font.size = Pt(size)
    run.bold = bold
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CJK)


def cell_text(cell, text, size=10.5, bold=False, align=None):
    cell.text = ""
    lines = str(text).split("\n")
    p = cell.paragraphs[0]
    if align:
        p.alignment = align
    for i, line in enumerate(lines):
        if i:
            p = cell.add_paragraph()
            if align:
                p.alignment = align
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(line)
        style_run(r, size, bold)


def add_row(table):
    """Append a row that mirrors the last one (same cell widths and borders)."""
    tbl = table._tbl
    import copy
    new = copy.deepcopy(tbl.tr_lst[-1])
    tbl.append(new)
    return table.rows[-1]


def replace_table_with_text(doc, table, texts, size=10.5):
    """Swap a single-column scratch table for ordinary body paragraphs."""
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph

    tbl = table._tbl
    for t in texts:
        p_el = OxmlElement("w:p")
        tbl.addprevious(p_el)
        p = Paragraph(p_el, doc._body)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.first_line_indent = Cm(0.74)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        style_run(p.add_run(t), size)
    tbl.getparent().remove(tbl)
    tbl = table._tbl
    import copy
    new = copy.deepcopy(tbl.tr_lst[-1])
    tbl.append(new)
    return table.rows[-1]


def put_image(cell_or_para, path, width_cm=16.0):
    if hasattr(cell_or_para, "paragraphs"):
        cell_or_para.text = ""
        p = cell_or_para.paragraphs[0]
    else:
        p = cell_or_para
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))


def para_before(anchor, text, size=10.5, bold=False, align=None, indent=True,
                space_before=3, space_after=3):
    p = anchor.insert_paragraph_before()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align:
        p.alignment = align
    r = p.add_run(text)
    style_run(r, size, bold)
    return p


# --------------------------------------------------------------------------
def main():
    runs = load_runs()
    doc = docx.Document(TEMPLATE)
    T = doc.tables

    # ---------------- header ----------------
    hdr = T[0]
    pairs = [("人工智能——深度学习大模型智能体", "2026 年 10 月 27 日"),
             ("黄思涵", "2025311574"),
             ("—", "个人计算机（Windows 11，纯 CPU）")]
    for r, (left, right) in enumerate(pairs):
        cell_text(hdr.rows[r].cells[1], left, align=WD_ALIGN_PARAGRAPH.CENTER)
        cell_text(hdr.rows[r].cells[3], right, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ---------------- 一、实验目的 ----------------
    purpose = [
        "1. 掌握在 Blender 中搭建白模（clay）场景并用 Cycles 输出可复现深度/法线控制图的完整流程，理解深度归一化区间与反相处理对 ControlNet 输入质量的影响。",
        "2. 掌握 stable-diffusion.cpp 的 GGUF 量化模型部署与纯 CPU 推理调用方法，能独立完成“权重准备（safetensors→GGUF）→ txt2img + ControlNet 控制生成”的流水线搭建。",
        "3. 学会用受控变量法记录分辨率、采样步数、CFG、ControlNet 强度等参数对生成质量与耗时的影响，并对结果做定量对比与几何一致性评价。",
        "4. 能够定位生成失败与画面缺陷（几何漂移、材质错误、深度图失效等）的成因并提出改进方案；初步掌握人像融合中的抠图、预乘 alpha 与梯度域融合方法。",
    ]
    t1 = T[1]
    while len(t1.rows) < len(purpose):
        add_row(t1)
    for i, text in enumerate(purpose):
        cell_text(t1.rows[i].cells[0], text)

    # ---------------- 二、实验环境 ----------------
    env = [
        ("CPU / 内存", "Intel Core i7-14650HX（16 核 24 线程，8P+8E），15.6 GB DDR5；无独立显卡参与计算"),
        ("操作系统", "Windows 11 专业版 10.0.26200（x64）"),
        ("Blender 版本", "Blender 5.1.0（Cycles，CPU 设备，Standard 视图变换，64 spp + OIDN 降噪）"),
        ("推理后端与版本", "stable-diffusion.cpp（Windows x64 CPU 预编译版，commit be48649 / master-1017），ggml CPU 后端，AVX2/AVX-VNNI/F16C，16 线程"),
        ("主模型文件", "models/sd_turbo-q4_0.gguf（SD-Turbo，Q4_0 量化，1 847 909 248 B ≈ 1.85 GB；SD2.1 架构，1024 维 cross-attention）"),
        ("ControlNet 文件", "models/control_v11p_sd21_depth.safetensors（SD 2.1 depth，728 494 536 B）→ sd-cli -M convert --type f16 → control_depth_sd21-fp16.gguf（728 482 784 B）"),
        ("其他依赖", "Python 3.11.12 + numpy 2.4.6 / Pillow 13.0.0 / scipy 1.17.1（控制图统计与人像融合）；bpy 5.1（场景与导出脚本）"),
    ]
    for i, (k, v) in enumerate(env):
        cell_text(T[2].rows[i + 1].cells[1], v)

    # ---------------- 三、实验原理 ----------------
    principle = [
        "1）深度控制图的生成。Cycles 渲染时用 view_layer.material_override 把全部物体替换为“纯发射”材质：Light Path 的 Ray Length（相机空间光线长度 d）经 MapRange 由 [near, far] 线性映射到 [1, 0]，即反相输出——近处白、远处黑，符合 ControlNet depth 的“近亮远暗”约定。near/far 不取场景极值，而取所有可见顶点投影到相机空间后距离的 2%/98% 分位数（本场景 near=7.520 m、far=11.256 m），避免 40 m 见方的地面把有效深度压缩到十几个灰阶。",
        "2）法线图。把世界空间法线 n 映射为 n*0.5+0.5 存成 RGB：R、G、B 对应 x、y、z 分量，正对相机处接近 (0.5,0.5,1)（淡紫色）。法线图在本实验中不参与条件控制，只用于人工核对物体朝向与遮挡关系。",
        "3）量化与纯 CPU 推理。SD-Turbo 是 SD2.1 的蒸馏版本，把 20~50 步的采样轨迹压缩到 1~4 步；蒸馏时教师工作在无分类器引导（CFG=1）条件，因此推理的 CFG 必须贴近 1.2。权重用 Q4_0 量化：每 32 个权重共享一个 f16 尺度因子，主模型由 f16 的约 2.4 GB 降到 1.85 GB，解码时按块反量化；纯 CPU 上瓶颈是内存带宽而非算力。",
        "4）控制生成流程。sd-cli 同时加载 SD2.1 UNet、VAE 与 SD2.1-depth ControlNet，把深度图编码成与 UNet 各层同分辨率的多尺度特征，在每一步去噪中以 --control-strength 加权注入；文本提示词给出材质与光照先验，深度条件给出几何骨架。SD1.5 的 depth ControlNet（768 维 cross-attention）与本模型（1024 维）形状不匹配，会直接报 got [768,1280] expected [1024,1280]。",
    ]
    for i, text in enumerate(principle):
        cell_text(T[3].rows[i].cells[0], text)

    # ---------------- 四、步骤 ----------------
    steps = [
        ("1", "搭建白模场景", "blender -b --factory-startup -P blender/make_demo_scene.py -- --out assets/demo_white_model.blend --res 512",
         "assets/demo_white_model.blend"),
        ("2", "导出 depth / normal / clay", "blender -b assets/demo_white_model.blend -P blender/export_control_maps.py -- --outdir output --name demo --res 512 --samples 64",
         "output/demo_depth_0001.png、_normal_0001.png、_clay.png"),
        ("3", "下载并转换 ControlNet 权重", "sd-cli.exe -M convert -m models/control_v11p_sd21_depth.safetensors -o models/control_depth_sd21-fp16.gguf --type f16",
         "models/control_depth_sd21-fp16.gguf"),
        ("4", "A 组：512² / 4 步基准生成", "python scripts/run_pipeline.py --name a512_s4 --res 512 --steps 4 --cfg 1.2 --control-strength 0.9 --seed 42",
         "output/a512_s4_photoreal.png、_run.json"),
        ("5", "B 组：768² / 4 步分辨率对比", "python scripts/run_pipeline.py --name b768_s4 --res 768 --steps 4",
         "output/b768_s4_photoreal.png"),
        ("6", "C/D/E 组消融 + 拓展人像融合", "python scripts/run_pipeline.py --name c512_s8 --steps 8 …；--name d512_ctrl05 --control-strength 0.5 …；--name e512_cfg3 --cfg 3.0 …；bash scripts/run_extension.sh",
         "output/c512_s8 / d512_ctrl05 / e512_cfg3 / head_fused*"),
    ]
    for i, row in enumerate(steps):
        for j, val in enumerate(row):
            cell_text(T[4].rows[i + 1].cells[j], val, size=9.0)

    # ---------------- 五、参数表 ----------------
    params = [("A", "512×512", "4", "1.2", "0.9", "euler_a", "42"),
              ("B", "768×768", "4", "1.2", "0.9", "euler_a", "42"),
              ("C", "512×512", "8", "1.2", "0.9", "euler_a", "42"),
              ("D", "512×512", "4", "1.2", "0.5", "euler_a", "42"),
              ("E", "512×512", "4", "3.0", "0.9", "euler_a", "42")]
    t5 = T[5]
    while len(t5.rows) < len(params) + 1:
        add_row(t5)
    for i, row in enumerate(params):
        for j, v in enumerate(row):
            cell_text(t5.rows[i + 1].cells[j], v, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ---------------- 提示词 ----------------
    prompt = ("photorealistic render, studio soft lighting, PBR material, "
              "preserve original 3d geometry, sharp")
    cell_text(T[6].rows[0].cells[0], prompt)
    cell_text(T[7].rows[0].cells[0], "deformed geometry, perspective changed, blurry")

    # ---------------- 六、数据表 ----------------
    notes = {"a512_s4": "基准配置；进程常驻 3.6 GB",
             "b768_s4": "像素数为 A 的 2.25 倍，单步 23.3 s",
             "c512_s8": "步数翻倍，单步与 A 基本一致（9.1 s）",
             "d512_ctrl05": "ControlNet 强度减半，条件失效",
             "e512_cfg3": "CFG 超出蒸馏适用范围，过饱和"}
    order = ["a512_s4", "b768_s4", "c512_s8", "d512_ctrl05", "e512_cfg3"]
    for i, name in enumerate(order):
        r = runs[name]
        row = [name[0].upper(), fmt(r.get("sampling_s")), fmt(r.get("vae_decode_s")),
               fmt(r.get("sdcli_total_s")), notes[name]]
        for j, v in enumerate(row):
            cell_text(T[8].rows[i + 1].cells[j], v,
                      align=WD_ALIGN_PARAGRAPH.CENTER if j else None)

    # ---------------- 7.1 / 7.2 ----------------
    cell_text(T[9].rows[0].cells[0],
              "图 7-1  白模导出的控制图（左：depth 反相近亮远黑；中：normal 仅用于核对朝向；右：clay 白模渲染）")
    put_image(T[9].rows[1].cells[0], os.path.join(OUT, "fig_control.png"), 16.0)
    cell_text(T[10].rows[0].cells[0],
              "图 7-2  同一种子下的生成结果（A：512²/4 步；C：512²/8 步；B：768²/4 步）")
    put_image(T[10].rows[1].cells[0], os.path.join(OUT, "fig_runs.png"), 16.0)

    # ---------------- 7.3 几何一致性 ----------------
    geo = [
        "以白模（图 7-1 右）为参照，A 组四个几何体的数量、位置、相对大小与遮挡关系全部正确：球与圆环在后、圆柱与立方体在前，圆环被立方体遮挡的部分在生成图中也确实被遮挡，物体与桌面的接触阴影方向一致，说明深度条件把相机空间中的相对距离正确地传递给了 UNet。为验证“轮廓正确来自深度而非提示词”，在同一提示词、同一种子下去掉 ControlNet 又跑了一次（图 7-3 右下）：模型生成的是一个与白模毫无关系的室内场景（窗、砖柱、桌椅），证明没有深度条件时 SD-Turbo 完全不受几何约束。",
        "不一致之处同样明显：① A 组圆柱被“改造”成一只开口杯——深度图上圆柱顶面灰度约 0.86，与其身后的地面几乎同色，ControlNet 无法区分“水平顶面”与“更远的地面”，模型遂按“容器”的语义先验补出了杯口内壁；② A 组圆环的孔洞被一块灰白圆盘填满——孔洞处本应透出更远的黑色背景，但深度图在该区域取到的是地面中灰，模型把它解释成“透过孔洞看到的地面”，镂空感丢失；③ A 组立方体的可见棱线略有软化，右下角出现 1~2 px 倒圆。",
        "提高分辨率或步数都能修正几何：B 组（768²）中圆柱保持完整圆柱且顶面平坦、圆环孔洞真正镂空、立方体三条棱笔直；C 组（8 步）同样修正了“圆柱变杯”和“孔洞被填”两处错误，只是立方体侧面出现了过强的木纹层板。说明 SD-Turbo 在 512²/4 步时尚未完全收敛，几何细节依赖模型对深度灰阶连续性的推理精度，而分辨率与步数都在提高这种精度——代价是 B 组耗时达到 A 的 2.38 倍（144.3 s 对 60.6 s）。",
    ]
    replace_table_with_text(doc, T[11], geo)

    # ---------------- 7.4 画面质量与缺陷 ----------------
    qual = [
        "材质与光照整体可信：A 组中球呈深色粗陶、圆环是高光釉面、圆柱为白瓷、立方体是木纹，桌面有木纹与柔和高光；三点布光方向与白模一致（左上主光、右侧补光、后方轮廓光），接触阴影紧贴物体底部，没有悬浮感。B 组材质颗粒更细（砂岩/混凝土感），但整体偏灰、缺乏高光，雾感更重；C 组木纹过强，立方体侧面出现平行的“层板”条纹，属于步数增加后细节过度锐化的副作用。",
        "至少三处可复现的失败：① 强度实验 D（ControlNet=0.5，图 7-3 右上）：圆环与立方体完全消失，被替换成一堆陶罐与三脚架，说明条件强度不足时模型会退回语义先验，“像什么”压倒了“在哪里”；② 过引导实验 E（CFG=3.0，图 7-3 左下）：画面严重过饱和，桌面变成暗红色、圆环孔洞出现黄色高光球、暗部细节被压死，这是蒸馏模型在训练分布之外的引导强度下把速度场线性外推的结果；③ 深度图自身的前景饱和：near=7.520 m 以内的地面已被裁剪为纯白，生成结果在画面底部出现一条突兀的“桌沿”横线（A 组最明显），说明深度图的动态范围裁剪会直接变成画面的结构伪影。此外 B 组右上角有一个约 20 px 的浅色圆形伪影，属于平坦区域采样噪声的残留。",
    ]
    replace_table_with_text(doc, T[12], qual)

    # ---------------- 7.3 前的消融图（插在 7.4 标题前） ----------------
    anchor74 = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("7.4"):
            anchor74 = p
            break
    fig_abl = os.path.join(OUT, "fig_ablation.png")
    if anchor74 is not None and os.path.isfile(fig_abl):
        cap = anchor74.insert_paragraph_before(
            "图 7-3  参数消融（左上：A 基准，ctrl=0.9；右上：D，ctrl=0.5；左下：E，CFG=3.0；右下：去掉 ControlNet 的对照）")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.keep_with_next = True
        for r in cap.runs:
            style_run(r, 9.0)
        pic = anchor74.insert_paragraph_before()
        pic.paragraph_format.keep_together = True
        put_image(pic, fig_abl, 13.0)

    # ---------------- 7.5 拓展：人像融合 ----------------
    anchor = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("八、问题与解决"):
            anchor = p
            break

    ext_meta = os.path.join(OUT, "head_fused_raw.json")
    ext_json = {}
    if os.path.isfile(ext_meta):
        with open(ext_meta, encoding="utf-8") as f:
            ext_json = json.load(f)

    if anchor is not None:
        h = anchor.insert_paragraph_before("7.5 拓展：人像融合（半身像 + 正脸）")
        h.style = "Heading 2"
        para_before(anchor, "拓展目标：把一张正脸合成到 7.2 节生成的半身像上，使其看起来像同一次拍摄。整条链路共 6 步：")
        steps_ext = [
            "（1）场景：make_head_scene.py --variant bust 生成“头+颈+肩+胸”白模（相机 63 mm，(0.34,-3.95,1.78)→(0,0,1.40)，三点布光），export_control_maps.py 导出 depth/clay（near=3.416 m、far=4.508 m）。与 7.1 的静物场景共用同一套导出代码，保证控制图约定一致。",
            "（2）底图生成：run_pipeline.py 以 bust 深度图为条件生成 head_photoreal.png（512²、4 步、CFG 1.2、ctrl 0.9、种子 42）；模型把“蛋形头+圆角肩条+胸台”的白模解释成一尊石雕胸像，这一步决定了几何与材质基底。",
            "（3）头像来源：融合只需要“一张脸”，不要求与白模同姿态，因此直接文生图（不加 ControlNet）得到 512² 正脸 face_src.png（4 步、CFG 1.3、种子 7，提示词强调 pure white background）。先试过“用 headonly 场景的深度图当 alpha”，结果模型在蛋形轮廓内自动补出针织兜帽，连兜帽一起贴了进去——融合源的生成方式会直接影响抠图难度。",
            "（4）抠图与对齐：正脸图的边框被头发与背景混占，按边框颜色做 flood key 会在发丝与额头高光处失败，故改用“羽化椭圆”alpha（rx=0.46、ry=0.50、羽化 0.10）只保留中央脸区；再从 bust 深度图取前景外接框得头部框 (67,47,444,384)，按 --face 0.28,0.25,0.78,0.87 推出面部框 (172.6,131.2,361.1,340.2)，用 --warp-fill 把脸拉伸到该框（缩放比 0.4198）。",
            "（5）预乘与梯度域融合：颜色先乘 alpha 再合成，避免缩放/羽化后边缘混入背景色形成白边；亮度均值/方差匹配只作用在 alpha>0 的像素上（早期版本对整个贴图矩形做匹配，矩形边界肉眼可见）；随后在贴图区解 Poisson 方程 ∇²f=∇²(src)、边界取底图，与直接贴图按 --clone 0.25 混合，最后按底图高频噪声强度补回颗粒。融合耗时 1.48 s。",
            "（6）img2img 精修：以融合图为初始图、同一张 bust 深度图为条件，--strength 0.30 / 8 步 / CFG 1.2 / --control-strength 0.6 再生成一次，让模型在低噪声水平下重绘接缝与光照。精修强度必须拿捏：0.22 时接缝仍在，0.35 且提示词写“石雕”时会把脸重新画回石头五官（本实验两版都跑过），0.30 + “young person”才既保住五官又统一了光影。",
        ]
        for s in steps_ext:
            para_before(anchor, s, size=10.0)

        if ext_json:
            para_before(anchor,
                        f"实测：头部框 {ext_json.get('head_box')}，面部框 {ext_json.get('face_box')}，"
                        f"缩放比 {ext_json.get('scale')}，贴图区 {ext_json.get('paste_box')}，"
                        f"融合耗时 {ext_json.get('seconds')} s；底图颗粒 σ={ext_json.get('grain_base')}，"
                        f"融合后 σ={ext_json.get('grain_out')}。", size=10.0)

        ext_fig = os.path.join(OUT, "fig_ext.png")
        if os.path.isfile(ext_fig):
            cap = anchor.insert_paragraph_before(
                "图 7-4  人像融合链路（左上：深度控制生成的半身像；右上：文生图得到的正脸；左下：羽化椭圆 alpha + 预乘 + Poisson 贴图（未精修）；右下：img2img 精修后的最终结果）")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.keep_with_next = True
            for r in cap.runs:
                style_run(r, 9.0)
            pic2 = anchor.insert_paragraph_before()
            pic2.paragraph_format.keep_together = True
            put_image(pic2, ext_fig, 13.0)

    # ---------------- 八、问题与解决 ----------------
    probs = [
        ("Blender 5.1 运行指南中的 Mist 通道脚本报错 AttributeError: 'Scene' object has no attribute 'node_tree'",
         "Blender 5.x 把合成器节点树迁移到 scene.compositing_node_group，旧的 scene.node_tree / use_nodes 接口被移除，指南脚本的添加节点写法整体失效。",
         "放弃合成器 Mist，改用 view_layer.material_override 覆盖为纯发射的距离着色（Light Path.Ray Length → MapRange 反相），直接渲染出与 Mist 等价的深度图；near/far 由 numpy 在相机空间对可见顶点取 2%/98% 分位。"),
        ("加载 SD1.5 depth ControlNet 时报 got [768,1280] expected [1024,1280]（形状不匹配）",
         "SD-Turbo 是 SD2.1 架构，cross-attention 宽度 1024；SD1.5 的 ControlNet 输出 768 维，二者不能混用。",
         "改用 thibaud/controlnet-sd21 的 SD2.1 depth 模型，并用 sd-cli -M convert --type f16 转成 GGUF（728 MB → 728 482 784 B）。"),
        ("256²/2 步的冒烟测试输出纯噪声",
         "SD2.1/SD-Turbo 在 512² 上训练，256² 与 2 步同时偏离训练分布（位置编码与步数嵌入都超出适用范围）。",
         "固定 512²/4 步后正常出图（单张 40.2 s）；正式实验只使用 512²/768² 且步数 ≥4。"),
        ("同时加载主模型、ControlNet 与 VAE 时进程常驻 3.6 GB，机器仅剩约 3.5 GB 空闲",
         "本机内存 15.6 GB（低于指南的 32 GB），且 DSH/QQ/Word 等已占用约 5 GB；VAE 解码时还有 416 MB 计算缓冲峰值。",
         "按 512² 优先跑单组实验、跑前关闭其它大内存程序；必要时加 --vae-tiling / --mmap 降低峰值。"),
    ]
    t13 = T[13]
    while len(t13.rows) < len(probs) + 1:
        add_row(t13)
    for i, row in enumerate(probs):
        for j, val in enumerate(row):
            cell_text(t13.rows[i + 1].cells[j], val, size=9.5)

    # ---------------- 九、结论 ----------------
    concl = [
        "1. 完成了从 Blender 白模到纯 CPU 深度控制生成的完整流水线并全部跑通：make_demo_scene.py 建场景、export_control_maps.py 出 depth/normal/clay、run_pipeline.py 调 sd-cli 完成 txt2img+ControlNet，5 组参数对比与 1 组无控制基线的数据均已记录、可复现。",
        "2. 本机（i7-14650HX / 15.6 GB / 无独显）上 512²/4 步单张 60.6 s（采样 37.3 s + VAE 14.5 s + Blender 6.7 s），768²/4 步 144.3 s，明显快于指南参考值（275 s / 348 s）：少步蒸馏 + Q4_0 量化让 SD2.1 在纯 CPU 上具备可用性。",
        "3. 参数规律：分辨率提高使采样时间按像素数的约 1.1 次方增长（2.25 倍像素 → 2.49 倍采样时间，注意力开销超线性）；步数近似线性（4→8 步，37.3→73.0 s）；ControlNet 使单步耗时从 5.35 s 增至 9.33 s（+74%）；强度 0.9 是几何与材质自由度的较优折中，0.5 会导致几何崩塌；CFG 必须贴近蒸馏值 1.2，3.0 会过饱和。",
        "4. 几何保持度：512²/4 步已能保证位置与遮挡正确，但会把圆柱误判为杯子、填平圆环孔洞；768² 或 8 步可修正这两处错误，代价是耗时翻倍与材质单一化。",
        "5. 可改进方向：用“法线+深度”双条件、对平坦顶面单独做归一化、以 MiDaS 生成更平滑的相对深度，或对结果做 img2img 低强度精修，在保持几何的同时恢复材质多样性；人像融合可引入 IP-Adapter 进一步保证身份一致。",
    ]
    t14 = T[14]
    while len(t14.rows) < len(concl):
        add_row(t14)
    for i, text in enumerate(concl):
        cell_text(t14.rows[i].cells[0], text)

    # ---------------- 十、思考题 ----------------
    qa = [
        ["SD-Turbo 由 SD2.1 蒸馏而来，蒸馏时教师模型工作在无分类器引导（CFG=1）条件下，学生只学到“无条件预测与条件预测相同”的速度场。CFG 的公式 ε=ε_neg+w(ε_pos-ε_neg) 在 w>1 时把两者的差放大，这是模型训练时从未见过的外推方向；在 1~4 步、每步位移极大的情况下，误差会逐步累积，表现为过饱和、彩色描边、塑料感与几何漂移。本实验 E 组把 CFG 从 1.2 提到 3.0，桌面立刻变成暗红色、圆环孔洞出现黄色高光球，即为过引导的直接证据。",
         "CFG 与步数是补偿关系：常规模型靠 20~50 步逐步修正，每步位移小，引导误差可以被后续步骤部分纠正；蒸馏模型把总位移压到 4 步走完，任何方向的放大都会被完整保留。因此步数越少，对 CFG 越敏感，CFG 必须贴近 1.2（本实验取 1.2，出图正常）。",
         "若一定要用较高 CFG，就必须同时增加步数并引入 CFG-rescale、动态引导等手段，使每步放大不超出训练分布；对 SD-Turbo 而言更实用的做法是保持 CFG≈1.2，用提示词与 ControlNet 强度控制风格。"],
        ["ControlNet 的输入是 8 bit 灰度，可用动态范围仅 0~255。若 near/far 由场景极值决定（例如沿用 0.05~40 m），物体所在的 2~3 m 深度区间只占十几阶灰，编码器无法区分前后关系，生成结果会“无视”几何、把多个物体压在同一层。反之若区间过窄，深度图出现大面积纯黑/纯白饱和块，模型把饱和区当作平面或直接压平，表现为物体被“拍扁”、轮廓粘连。",
         "本实验用可见顶点 2%/98% 分位（near=7.520 m、far=11.256 m）代替极值，使有效深度落在中间灰阶，这是 7.3 节几何基本正确的前提。若把 mist 起始距离再设小（例如 near=4 m），前景地面会更早饱和为纯白，近处桌面与圆柱底部将连成一片，模型会把它们解释成一块巨大的近景平面——本实验 A 组底部那条突兀的“桌沿”横线就是这种饱和的轻量版本。",
         "实践建议：先用 2%/98% 分位自动定区间，再检查直方图两端是否堆积；若前景占比过大，可抬高相机、减少地面入画，或对深度图做直方图拉伸/伽马校正，把主体灰阶拉开。"],
        ["强度 0.9 时深度条件几乎主导，轮廓、遮挡与透视与白模高度一致（A/C/B 三组均如此），但材质与光照自由度被压缩，模型只能在“贴着几何”的前提下上色，画面偏硬、纹理容易过强（C 组的层板木纹）。强度 0.5 时条件变弱（D 组），模型有更大创作空间，代价是几何崩塌：圆环与立方体被替换成陶罐与三脚架，只有整体明暗分布还残留白模的影子。",
         "与初始图信息的权衡：在 img2img 场景下，初始图本身携带几何信息，ControlNet 强度可以适当降低（本实验拓展用 0.7 配 strength 0.35），让模型主要修复光照与接缝；在 txt2img 场景下没有任何几何先验，强度低于 0.7 就开始丢形状，0.85~0.95 是安全区。",
         "经验规则：几何由谁提供，谁就应该被强约束。纯 txt2img + ControlNet → 0.9；img2img 精修（初始图已对齐）→ 0.6~0.75；只想“轻微引导构图”→ 0.4~0.5，并接受形状漂移。"],
        ["纯 CPU 推理的瓶颈是内存带宽与矩阵乘吞吐：每步要对 UNet（约 0.86 G 参数，Q4_0 后约 1.7 GB）做一次前向，权重需反复从内存搬运；ControlNet 再叠加约 340 M 参数的编码器，使每步权重流量增加约 40%。实测单步耗时 9.33 s（开 ControlNet）对 5.35 s（关 ControlNet），增加 74%，远高于指南给出的 10~20%，原因是本机 ControlNet 以 f16 全精度在 CPU 上运行、未做量化。",
         "耗时估算式：T ≈ 步数 × k × (W·H)/512² + T_VAE，其中 T_VAE 也与像素数近似成正比。本机 k≈9.3 s（512²、开 ControlNet）；实测 768²（像素 2.25 倍）单步 23.3 s，比值 2.49，说明 k 随分辨率略超线性（注意力随 token 数增长）。",
         "据此：512²/4 步 ≈ 4×9.3 + 14.5 ≈ 51.7 s（实测 37.3+14.5=51.8 s）；768²/4 步 ≈ 4×23.3 + 33.8 ≈ 127 s（实测 93.0+33.8=126.8 s），估算误差 <5%，可作为排期依据。"],
        ["预乘 alpha 把颜色先乘上不透明度（C'=αC），使颜色与 alpha 处于同一线性空间。若不做预乘，半透明边缘像素的 RGB 仍是“前景色与旧背景色的混合”，缩放/羽化后这层旧背景色会被当作真实颜色保留，合成到新背景上就形成一圈亮边（白边）或暗边。",
         "预乘后，颜色插值与高斯羽化都在线性空间进行，边缘像素对最终颜色的贡献正比于其 alpha，不会混入旧背景；同时全透明像素颜色为 0，重采样时不会把“看不见的颜色”平均进来造成渗色。本实验的 compose_head.py 因此在 resize 与 paste 之前就完成预乘，并以 alpha 区域作为 Poisson 融合的求解域。",
         "代价是预乘图像不能再直接当普通 RGB 使用（颜色被压暗），必须与 alpha 一起存储处理；这也是 PNG 的 straight alpha 与渲染器常用的 premultiplied alpha 需要显式区分的原因。"],
        ["路线一（推理期注入身份）：用 IP-Adapter / PhotoMaker / InstantID 把人脸编码成 ID 向量注入 cross-attention，种子变化时 ID 向量不变，脸就稳定；再配 ControlNet OpenPose + depth 约束姿态，实现“身份”与“姿态”解耦。",
         "路线二（权重内化）：用同一人的 20~50 张照片训练 LoRA 或做 DreamBooth 微调，把身份写进权重，之后任意种子、任意提示词都带同一张脸；缺点是每个身份都要单独训练。",
         "路线三（生成后筛选与重绘）：用 ArcFace 计算候选与参考图的身份相似度，做 rejection sampling 或重排序；再对选中样本用 inpaint 局部重绘微调表情与光照。工程上常把三条路线组合：ID 注入保证初筛通过率、LoRA 保证稳定性、ArcFace 做最终质检。"],
    ]
    for qi, rows in enumerate(qa):
        t = T[15 + qi]
        while len(t.rows) < len(rows):
            add_row(t)
        for i, text in enumerate(rows):
            cell_text(t.rows[i].cells[0], text)

    # ---------------- 删除模板提示语 ----------------
    for p in list(doc.paragraphs):
        txt = p.text.strip()
        if txt.startswith("（请用") or txt.startswith("（简述") or txt.startswith("（总结"):
            p._element.getparent().remove(p._element)

    out = os.path.join(os.path.dirname(HW7), "2025311574_黄思涵_实验作业七.docx")
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
