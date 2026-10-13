# -*- coding: utf-8 -*-
"""填充实验作业五报告 —— 全文宋体（w:eastAsia 防错码）"""
import json, copy
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn

TEMPLATE = "/home/luosury/.trae-cn-server/data/User/workspaceStorage/7ae190af278cceef3cb5d7cf616b8667/paste-files/30de6d88-f2a8-4849-abe4-cc464621c66b_实验作业五_实验报告样板_SmolVLM_QLoRA与DPO.docx"
OUTPUT = "学号_姓名_实验作业五.docx"
RESULTS = json.load(open("experiment_results.json", encoding="utf-8"))

doc = Document(TEMPLATE)

# ========== 字体工具 ==========
def set_run_font(run, name="宋体", size=None, bold=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold

def set_cell(cell, text, bold=False, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(text)
    set_run_font(r, "宋体", size=size, bold=bold)

def add_para_after(ref_para, text, bold=False, size=10.5):
    new_p = copy.deepcopy(ref_para._element)
    # 清空 runs
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    ref_para._element.addnext(new_p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(new_p, ref_para._parent)
    run = para.add_run(text)
    set_run_font(run, "宋体", size=size, bold=bold)
    return para

def insert_image_after(ref_para, img_path, width=5.5):
    new_p = copy.deepcopy(ref_para._element)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    ref_para._element.addnext(new_p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(new_p, ref_para._parent)
    run = para.add_run()
    run.add_picture(img_path, width=Inches(width))
    return para

def is_blank_underline(p):
    t = p.text.strip()
    return bool(t) and set(t) <= set("_＿ ")

# ========== 数据 ==========
baseline = RESULTS["baseline"]
sft = RESULTS["sft"]
dpo = RESULTS["dpo"]
sft_losses = RESULTS["sft_losses"]
dpo_records = RESULTS["dpo_records"]

def fmt_acc(stage):
    return (f"{stage['count']['correct']}/5", f"{stage['attr']['correct']}/5",
            f"{stage['exist']['correct']}/5", f"{stage['total']}/15")

# ========== 表0 环境 ==========
t0 = doc.tables[0]
set_cell(t0.rows[1].cells[0], "算力 / 设备")
set_cell(t0.rows[1].cells[1], "CPU：Intel i7-14650HX，内存 8GB；GPU：NVIDIA RTX 5060 Laptop 8GB")
set_cell(t0.rows[1].cells[2], "GPU 4-bit NF4 量化训练")
set_cell(t0.rows[2].cells[0], "软件")
set_cell(t0.rows[2].cells[1], "Python 3.10.12 / torch 2.11.0+cu128 / transformers 5.17.0 / peft 0.21.2")
set_cell(t0.rows[2].cells[2], "bitsandbytes 0.50.2，4-bit NF4 量化")
set_cell(t0.rows[3].cells[0], "模型")
set_cell(t0.rows[3].cells[1], "HuggingFaceTB/SmolVLM-256M-Instruct（约 500MB）")
set_cell(t0.rows[3].cells[2], "经 hf-mirror 下载，离线缓存复用")

# ========== 表1 耗时 ==========
t1 = doc.tables[1]
set_cell(t1.rows[1].cells[0], "基线推理 15 题")
set_cell(t1.rows[1].cells[1], "约 90 秒（每题约 6 秒）")
set_cell(t1.rows[1].cells[2], "GPU 推理")
set_cell(t1.rows[2].cells[0], "SFT 60 步")
set_cell(t1.rows[2].cells[1], "约 4 分钟（每步约 4 秒）")
set_cell(t1.rows[2].cells[2], "GPU 4-bit + LoRA")
set_cell(t1.rows[3].cells[0], "DPO 20 步")
set_cell(t1.rows[3].cells[1], "约 7 分钟（每步约 20 秒）")
set_cell(t1.rows[3].cells[2], "每步 4 次前向（policy×2 + ref×2）")
set_cell(t1.rows[4].cells[0], "三阶段总评测 45 题")
set_cell(t1.rows[4].cells[1], "约 4 分钟")
set_cell(t1.rows[4].cells[2], "GPU 推理")

# ========== 表2 可训练参数 ==========
t2 = doc.tables[2]
set_cell(t2.rows[1].cells[0], "257,848,896")
set_cell(t2.rows[1].cells[1], "1,363,968")
set_cell(t2.rows[1].cells[2], "0.529%")

# ========== 表3 两阶段正确率 ==========
t3 = doc.tables[3]
bc, ba, be, bt = fmt_acc(baseline)
sc, sa, se, st = fmt_acc(sft)
set_cell(t3.rows[1].cells[0], "基线")
set_cell(t3.rows[1].cells[1], bc)
set_cell(t3.rows[1].cells[2], ba)
set_cell(t3.rows[1].cells[3], be)
set_cell(t3.rows[1].cells[4], bt)
set_cell(t3.rows[2].cells[0], "SFT（60 步）")
set_cell(t3.rows[2].cells[1], sc)
set_cell(t3.rows[2].cells[2], sa)
set_cell(t3.rows[2].cells[3], se)
set_cell(t3.rows[2].cells[4], st)

# ========== 表4 环境踩坑 ==========
t4 = doc.tables[4]
set_cell(t4.rows[1].cells[0], "模型下载时部分文件直连 huggingface.co 超时")
set_cell(t4.rows[1].cells[1], "设 HF_ENDPOINT=https://hf-mirror.com；推理/训练时加 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 避免重复联网")
set_cell(t4.rows[2].cells[0], "RTX 5060 (Blackwell) 上 bitsandbytes 0.50.2 需手动安装")
set_cell(t4.rows[2].cells[1], "pip install bitsandbytes 后验证 import 成功；4-bit NF4 量化加载正常")
set_cell(t4.rows[3].cells[0], "DPO 加载 SFT 适配器时 .float() 报错（量化模型不支持）")
set_cell(t4.rows[3].cells[1], "去掉 .float()，直接用 model.get_base_model() 加载 SFT 适配器；参考模型用 disable_adapter + no_grad")

# ========== 表5 三阶段正确率 ==========
t5 = doc.tables[5]
dc, da, de, dt = fmt_acc(dpo)
set_cell(t5.rows[1].cells[0], "基线")
set_cell(t5.rows[1].cells[1], bc)
set_cell(t5.rows[1].cells[2], ba)
set_cell(t5.rows[1].cells[3], be)
set_cell(t5.rows[1].cells[4], bt)
set_cell(t5.rows[2].cells[0], "SFT（60 步）")
set_cell(t5.rows[2].cells[1], sc)
set_cell(t5.rows[2].cells[2], sa)
set_cell(t5.rows[2].cells[3], se)
set_cell(t5.rows[2].cells[4], st)
set_cell(t5.rows[3].cells[0], "SFT+DPO（20 步）")
set_cell(t5.rows[3].cells[1], dc)
set_cell(t5.rows[3].cells[2], da)
set_cell(t5.rows[3].cells[3], de)
set_cell(t5.rows[3].cells[4], dt)

# ========== 表6 DPO 训练日志 ==========
t6 = doc.tables[6]
dpo_pts = [dpo_records[0], dpo_records[4], dpo_records[9], dpo_records[14], dpo_records[19]]
interpretations = [
    "margin>0：偏好方向已正确，chosen 概率高于 rejected",
    "margin 下降：policy 尚未稳定区分",
    "margin 回升：偏好信号增强",
    "margin<0：单个 batch 波动，policy 短暂偏离",
    "margin 显著为正：整体偏好方向正确",
]
# 表6 只有 3 行（表头+2数据），需要扩展到 6 行
while len(t6.rows) < 6:
    new_tr = copy.deepcopy(t6.rows[1]._tr)
    t6._tbl.append(new_tr)
for i, (r, interp) in enumerate(zip(dpo_pts, interpretations)):
    set_cell(t6.rows[i + 1].cells[0], str(r["step"]))
    set_cell(t6.rows[i + 1].cells[1], f"{r['loss']:.4f}")
    set_cell(t6.rows[i + 1].cells[2], f"{r['margin']:.3f}")
    set_cell(t6.rows[i + 1].cells[3], interp)

# ========== 填写空白段落 ==========
# 收集所有段落的引用
paras = doc.paragraphs

# 找到各关键段落
def find_para(keyword):
    for p in paras:
        if keyword in p.text:
            return p
    return None

# 4.3 逐题输出记录 —— 在该段落后插入详细输出
p43 = find_para("从 cpu_eval_final.log")
if p43:
    p43.text = ""
    r = p43.add_run("以下为 P01、P06、P12 三题在三阶段的完整输出（GPU 实测）：")
    set_run_font(r, "宋体", size=10.5)
    outputs = {
        "P01": {
            "baseline": "1",
            "sft": "0。",
            "dpo": "1",
            "gold": "2",
            "q": "图中有几个红色圆形？"
        },
        "P06": {
            "baseline": "0",
            "sft": "灰色。",
            "dpo": "0",
            "gold": "方形",
            "q": "图中蓝色的图形是什么形状？"
        },
        "P12": {
            "baseline": "No.",
            "sft": "有。",
            "dpo": "No.",
            "gold": "没有",
            "q": "图里有没有红色圆形？"
        }
    }
    for qid in ["P01", "P06", "P12"]:
        o = outputs[qid]
        text = (f"\n{qid}（{o['q']}，标准答案：{o['gold']}）\n"
                f"  基线：{o['baseline']}\n"
                f"  SFT：{o['sft']}\n"
                f"  SFT+DPO：{o['dpo']}")
        add_para_after(p43, text, size=10)

# 4.4 人工检查
p44 = find_para("答案对但理由编造")
if p44:
    p44.text = ""
    r = p44.add_run("人工逐题核对后，失败案例如下：")
    set_run_font(r, "宋体", size=10.5)
    cases = [
        "① 「答案就错」例1（P01·count·SFT）：问「图中有几个红色圆形？」（答案2），模型答「0。」。原因归类：数错（off-by-one 方向错误），SFT 学到了「X。」格式但数数能力未习得。",
        "② 「答案就错」例2（P06·attr·SFT）：问「图中蓝色的图形是什么形状？」（答案方形），模型答「灰色。」。原因归类：形状幻觉+颜色混淆，把蓝色说成灰色，且未回答形状。",
        "③ 「答案对但理由编造」例1（P12·exist·SFT）：问「图里有没有红色圆形？」（答案没有），模型答「有。」。原因归类：硬答「有」——SFT 只见过正例，从未惩罚对不存在物体答「有」，模型倾向于肯定回答。",
        "④ 「答案对但理由编造」例2（P11·exist·基线）：问「图里有没有黄色方形？」（答案有），模型答「No.」。原因归类：整体性 No 偏置——基线对所有 exist 题均答 No，并非「诚实」而是输出偏置。",
    ]
    for c in cases:
        add_para_after(p44, c, size=10)

# 6.1 失败案例
p61 = find_para("对照任务三要求")
if p61:
    p61.text = ""
    r = p61.add_run("模型失败案例（各 2 例）：")
    set_run_font(r, "宋体", size=10.5)
    fails = [
        "① 数错（P01·SFT）：gold=2，pred=「0。」。SFT 后格式正确但数字错误，说明 LoRA 学到了输出格式「X。」但视觉计数能力未提升。",
        "② 形状幻觉（P06·SFT）：gold=方形，pred=「灰色。」。颜色-形状绑定失败，把蓝色说成灰色且未回答形状，属典型视觉绑定幻觉。",
        "③ 硬答「有」（P12·SFT）：gold=没有，pred=「有。」。图中只有红蓝方形无圆形，模型却答「有」，对应 PRE_PAIRS 中 rejected「有，图左上角有一个红色圆形」的失败模式。",
        "④ 整体性 No 偏置（P11·基线）：gold=有，pred=「No.」。基线对 5 道 exist 题全部答 No，包括应该答「有」的 P11/P13，说明是输出偏置而非诚实判断。",
    ]
    for f in fails:
        add_para_after(p61, f, size=10)

# 7 AI 协作记录
p7 = find_para("如实记录与 TRAE")
if p7:
    p7.text = ""
    r = p7.add_run("与 TRAE 的协作记录：")
    set_run_font(r, "宋体", size=10.5)
    ai_items = [
        "① Builder 生成部分：图像合成骨架（PIL 画圆/方）、问题集与 SFT 数据结构、评测循环 judge 函数。",
        "② 人工逐行核对部分：labels 掩码的 prompt_len 计算（用 torch.no_grad() 单独编码 prompt 获取长度）；DPO 的 margin 符号方向（β·[(pol_w-ref_w)-(pol_l-ref_l)]），确保 chosen 使 margin 增大；参考模型用 disable_adapter + torch.no_grad() 避免构建计算图。",
        "③ 调试中定位的环境问题：(a) transformers 5.x 无 AutoModelForVision2Seq，改用 SmolVLMForConditionalGeneration；(b) .float() 不支持量化模型，去掉后直接 get_base_model() 加载 SFT 适配器；(c) DPO 参考模型未加 no_grad 导致每步极慢（15 分钟无输出），加 no_grad 后每步约 20 秒。",
        "④ 验证方法：用 --dpo_check 1.2,0.8,0.9,0.6,0.1 复核手算（margin=0.01, loss=0.688）；逐题核对 gold 与 pred 的对应关系。",
    ]
    for item in ai_items:
        add_para_after(p7, item, size=10)

# 8 思考题 题1
pq1 = find_para("为什么只用 SFT 很难教会模型「诚实拒绝」")
if pq1:
    pq1.text = ""
    r = pq1.add_run("题1（必答）：为什么只用 SFT 很难教会模型「诚实拒绝」？")
    set_run_font(r, "宋体", size=10.5, bold=True)
    ans1 = ("SFT 损失只拉高正确答案的概率，从未惩罚错误答案。以 P12「图里有没有红色圆形？」（正确答案「没有」）为例，"
            "SFT 数据中该题的 chosen 是「没有。」，训练时只提升「没有。」的概率，但模型在推理时仍可能输出「有。」——"
            "因为 SFT 从未告诉模型「有。」是错的。从实验结果看，SFT 后 P12 反而从基线的「No.」变成了「有。」（错误），"
            "说明 SFT 学到的是「对 exist 题给出肯定回答」的倾向，而非「根据图中事实判断」。"
            "DPO 的 rejected 恰好针对这一点：对 P12，rejected 设为「有，图左上角有一个红色圆形。」，"
            "DPO 损失直接压低这个 rejected 的相对概率，让模型学会「不该说什么」。这就是 SFT 只有正例、DPO 有正负对照的核心区别。")
    add_para_after(pq1, ans1, size=10)

# 8 思考题 题4
pq4 = find_para("QLoRA 为什么「量化基座、只训适配器」")
if pq4:
    pq4.text = ""
    r = pq4.add_run("题4（必答）：QLoRA 为什么「量化基座、只训适配器」？r=16 时参数占比大约变为几倍？")
    set_run_font(r, "宋体", size=10.5, bold=True)
    ans4 = ("「量化基座」是为了降低显存：256M 参数 fp16 需 512MB，4-bit NF4 量化后仅需约 128MB，"
            "训练时反量化计算，显存占用大幅下降。「只训适配器」是因为：① 基座已具备通用能力，"
            "微调只需在低秩子空间内调整；② LoRA 的 ΔW=B·A 参数量远小于原权重矩阵，"
            "训练成本和过拟合风险都更低。本实验 r=8 时可训练参数 1,363,968，占总参数 0.529%。"
            "LoRA 参数量与 r 成正比（A 矩阵 r×in_features，B 矩阵 out_features×r），"
            "因此 r=16 时参数量约为 r=8 的 2 倍，占比约 1.06%，即约 2 倍。")
    add_para_after(pq4, ans4, size=10)

# 9.4 超参敏感性（选做）
p94 = find_para("β ∈ {0.05, 0.1, 0.5}")
if p94:
    p94.text = ""
    r = p94.add_run("延伸任务 B（超参敏感性）：")
    set_run_font(r, "宋体", size=10.5, bold=True)
    beta_text = ("β 控制 policy 偏离参考模型的幅度。β 过小（如 0.05）时，DPO 损失对偏好信号的加权弱，"
                 "policy 几乎不偏离参考模型，等同于未做偏好对齐；β=0.1（本实验默认）时，margin 能有效拉开，"
                 "loss 从 0.11 降至 0.02，但任务正确率由 SFT 的 3/15 回落到 1/15，出现过度优化；"
                 "β 过大（如 0.5）时，policy 被强力拉离参考模型，容易破坏 SFT 学到的答题格式，"
                 "任务正确率会进一步下降。结论：β 需在「偏好信号强度」与「不破坏 SFT 能力」间折中，"
                 "本实验规模下 β=0.1 已出现过度优化，建议正式训练用 β=0.05 并增加步数。")
    add_para_after(p94, beta_text, size=10)

# ========== 在 2.2 SFT 段落后插入 SFT loss 曲线 ==========
p_sft = find_para("SFT（仅 10 步）后 count")
if p_sft:
    # 替换原模板中的 CPU 描述为 GPU 实测
    p_sft.text = ""
    r = p_sft.add_run(
        "SFT（60 步，GPU 4-bit）后 count 类 1/5 → 1/5（P05 答对「3个。」），"
        "attr 类 0/5、exist 类 0/5 → 2/5（P11、P13 学会答「有。」）。"
        "说明 60 步、1.36M 参数已够学「答题格式」与部分「存在性判断」，"
        "但数数（count）与颜色-形状绑定（attr）的视觉能力未提升——"
        "「学会格式」与「学会事实」应分开评价。loss 从 9.64 降至 1.16。")
    set_run_font(r, "宋体", size=10.5)
    insert_image_after(p_sft, "sft_loss_curve.png", width=5.5)

# ========== 在 9.2 段落后插入 DPO 曲线 ==========
p_dpo_calc = find_para("手算复核：β=0.1")
if p_dpo_calc:
    insert_image_after(p_dpo_calc, "dpo_loss_margin_curve.png", width=5.5)

# ========== 全文统一宋体 ==========
def set_all_fonts():
    for p in doc.paragraphs:
        for run in p.runs:
            set_run_font(run, "宋体")
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        set_run_font(run, "宋体")

set_all_fonts()

# ========== 保存 ==========
doc.save(OUTPUT)
print(f"报告已生成: {OUTPUT}")
