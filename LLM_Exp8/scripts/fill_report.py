# -*- coding: utf-8 -*-
"""把实验八的实测数据与图片填进空白报告模板。

用法（在 LLM_Exp8 目录下）：
    REPORT_NAME=... REPORT_ID=... python scripts/fill_report.py

姓名/学号/班级/日期都从环境变量读，仓库里不出现任何个人信息。
"""
import copy
import json
import os

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
HW = os.path.dirname(HERE)
RUNS = os.path.join(HW, "runs_distill")
DOCS = os.path.join(HW, "docs")
TEMPLATE = os.path.join(DOCS, "template_exp8.docx")
TARGET = os.path.join(DOCS, "实验八_知识蒸馏_实验报告.docx")

CJK = "宋体"
LATIN = "Times New Roman"
SIZE = 10.5

BODY = None  # 文档 body，供 add_text_after_element 构造段落时取 part

NAME = os.environ.get("REPORT_NAME", "")
SID = os.environ.get("REPORT_ID", "")
CLASS = os.environ.get("REPORT_CLASS", "人工智能（大模型方向）2025 级")
DATE = os.environ.get("REPORT_DATE", "2026 年 11 月 6 日")


# ---------------------------------------------------------------- 基础样式
def style_run(run, size=SIZE, bold=False):
    run.font.name = LATIN
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), CJK)


def cell_text(cell, text, size=9.5, bold=False, align=None):
    cell.text = ""
    lines = str(text).split("\n")
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    for i, line in enumerate(lines):
        if i:
            p = cell.add_paragraph()
            if align is not None:
                p.alignment = align
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        style_run(p.add_run(line), size, bold)


def add_row(table):
    tbl = table._tbl
    tbl.append(copy.deepcopy(tbl.tr_lst[-1]))
    return table.rows[-1]


def _blank_after(anchor_p, align=None, indent=False):
    """在 anchor_p 后面插一个空段落（沿用它的 pPr，再覆盖对齐/缩进）。"""
    el = copy.deepcopy(anchor_p._p)
    for child in list(el):
        if child.tag != qn("w:pPr"):
            el.remove(child)
    anchor_p._p.addnext(el)
    p = Paragraph(el, anchor_p._parent)
    p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
    p.paragraph_format.keep_with_next = False
    if align is not None:
        p.alignment = align
    return p


def add_text(anchor_p, text, indent=True, bold=False, align=None, size=SIZE):
    p = _blank_after(anchor_p, align=AL.JUSTIFY if indent and align is None else align,
                     indent=indent)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    style_run(p.add_run(text), size, bold)
    return p


def add_block(anchor_p, texts, indent=True):
    last = anchor_p
    for t in texts:
        last = add_text(last, t, indent=indent)
    return last


def add_picture(anchor_p, path, width_cm=15.5):
    p = _blank_after(anchor_p, align=AL.CENTER)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Cm(width_cm))
    p.paragraph_format.keep_with_next = True
    return p


def add_caption(anchor_p, text):
    p = _blank_after(anchor_p, align=AL.CENTER)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    style_run(p.add_run(text), SIZE, True)
    return p


def add_text_after_element(el, text, indent=True):
    new_p = OxmlElement("w:p")
    el.addnext(new_p)
    p = Paragraph(new_p, BODY)
    p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    if indent:
        p.alignment = AL.JUSTIFY
    style_run(p.add_run(text), SIZE)
    return p


def set_para(p, text, indent=True, size=SIZE):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    if indent:
        p.alignment = AL.JUSTIFY
    style_run(p.add_run(text), size)
    return p


def put_image(cell, path, width_cm=15.0):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = AL.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))


def replace_table_with_picture(doc, table, path, width_cm=15.0):
    """把 1×1 的图片占位表格换成普通段落，避免表格边框框住图片。"""
    p = doc.add_paragraph()
    p.alignment = AL.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    tbl_el = table._tbl
    tbl_el.addprevious(p._p)
    tbl_el.getparent().remove(tbl_el)
    return p


def set_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Cm(w)


def clear_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def no_split(table):
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


# ---------------------------------------------------------------- 数据
def load(name):
    with open(os.path.join(RUNS, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    global BODY
    allres = load("results_all.json")
    abl = load("results_ablate.json")["ablation"]
    steps = load("results_steps.json")["steps"]
    ext = load("results_ext.json")["ext"]

    th, s0, s1 = allres["teacher"], allres["scratch"], allres["kd"]
    q = ext["quant"]
    scale = ext["teacher_scale"]
    selfd = ext["self_distill"]
    feat = {r["feat"]: r for r in ext["feat"]}
    best = min(abl, key=lambda r: r["val_loss"])
    worst = max(abl, key=lambda r: r["val_loss"])
    kd_best_step = max(steps, key=lambda r: r["gain_rel"])
    worst_step = min(steps, key=lambda r: r["gain_rel"])

    doc = docx.Document(TEMPLATE)
    BODY = doc._body
    T = doc.tables
    P = doc.paragraphs
    # 章节标题/小节标题与后文同页，避免标题落在页面最底部
    for idx in (3, 5, 6, 8, 9, 12, 16, 17, 20, 22, 25, 27, 28, 29, 31):
        P[idx].paragraph_format.keep_with_next = True

    # ------------------------------------------------ 封面
    hdr = T[0]
    cell_text(hdr.rows[2].cells[1], "%s / %s" % (NAME, SID) if NAME else "—",
              size=10.5, align=AL.CENTER)
    cell_text(hdr.rows[3].cells[1], CLASS, size=10.5, align=AL.CENTER)
    cell_text(hdr.rows[4].cells[1], DATE, size=10.5, align=AL.CENTER)
    cell_text(hdr.rows[5].cells[1],
              "Ubuntu 22.04（WSL2）/ Python 3.10.12 / PyTorch 2.11.0+cu128（CPU 模式）",
              size=10.5, align=AL.CENTER)

    # ------------------------------------------------ 一、实验目的
    set_para(P[4], "1. 理解知识蒸馏中“教师软标签”相对 one-hot 硬标签多提供的类间相似性信息，"
                   "掌握带温度 T 的 softmax、KL 散度与 T² 缩放的实现细节，能独立复现一份可用的蒸馏"
                   "训练代码。")
    add_block(P[4], [
        "2. 掌握在纯 CPU 上从零训练小型字符级 GPT（教师 0.737 M、学生 0.147 M）的完整流程，"
        "并会用“相同初始化 + 相同数据顺序 + 相同步数”的受控变量法做公平对照。",
        "3. 学会用二维网格（T × α）做超参数消融，用留出集 loss、困惑度 PPL、与教师 top-1 一致率"
        "三个指标评价蒸馏效果，并把结果整理成表格与图。",
        "4. 掌握特征蒸馏（FitNet）、自蒸馏、动态量化、教师规模扫描等拓展手段，理解“先蒸馏再量化”"
        "这条压缩流水线的组合方式与适用边界。",
        "5. 培养用受控实验回答机制性问题（收益来自“更多信息”还是“更强正则”）的能力，"
        "并能把观察到的现象与公式推导对应起来。",
    ])

    # ------------------------------------------------ 二、实验环境
    env = [
        "Ubuntu 22.04.5 LTS，运行在 WSL2 中（内核 6.6.87.2-microsoft-standard-WSL2，宿主 Windows 11）",
        "Python 3.10.12（LLM_Venv 虚拟环境）",
        "PyTorch 2.11.0+cu128，仅使用 CPU（torch.set_num_threads(12)）；matplotlib 3.10.9",
        "Intel Core i7-14650HX，12 核 24 线程；WSL2 分配 24 vCPU，训练时固定 12 线程",
        "7 GB（WSL2 默认上限，本实验峰值占用 < 0.6 GB）",
        "否。全程 CPU，单次 200 步教师训练仅 6.7 s，因此不需要 GPU",
    ]
    for i, v in enumerate(env):
        cell_text(T[1].rows[i + 1].cells[1], v)
    r = add_row(T[1])
    cell_text(r.cells[0], "其他依赖", size=9.5)
    cell_text(r.cells[1], "numpy 2.x（热力图）、python-docx（报告生成）；语料为脚本内置，无网络下载",
              size=9.5)

    # ------------------------------------------------ 三、原理简述
    set_para(P[7], "1）基本思想。知识蒸馏把已经学好“输入→输出”映射的教师模型的输出分布当作监督信号，"
                   "训练一个容量更小的学生。与 one-hot 标签相比，学生看到的不再只是“正确的下一个字是哪个”，"
                   "而是“给定上下文时词表里每个字分别有多大概率”，相当于把教师对数据的“理解方式”也迁移过来。")
    add_block(P[7], [
        "2）软标签多提供了什么。以本实验语料为例，教师在某个位置可能给出“月”0.62、“夜”0.11、“风”0.08、"
        "“雨”0.05 的分布：哪些字可以互相替换、哪些字几乎不可能出现，这种类间相似性就是所谓的 dark knowledge。"
        "硬标签把 322 维分布压成 1 个 1 和 321 个 0，学生必须在有限语料上自己重新发现这些结构；软标签把结构"
        "直接交给学生，对容量只有教师 1/5 的学生来说省下了一大笔“重新发现”的成本。",
        "3）带温度的 softmax。p_i(T) = exp(z_i/T) / Σ_j exp(z_j/T)。T = 1 时就是普通 softmax；T 越大指数被"
        "压平，原本接近 0 的负类概率被抬到数值上可见的范围，类间相对大小被放大。T → ∞ 时分布趋于均匀，"
        "信息反而丢失，所以 T 存在甜点区间（本实验实测为 4~8）。",
        "4）损失函数与 T²。L = (1−α)·CE(y, p_s) + α·T²·KL(p_t(T) ‖ p_s(T))。softmax 的雅可比随 1/T 缩放，"
        "因此 KL 对 logits 的梯度幅度约按 1/T² 衰减；乘 T² 后软标签项与 CE 项回到同一量级。这带来两个好处："
        "改变 T 时不必重新调学习率，α 的含义（硬标签与软标签的相对权重）也保持稳定。",
        "5）实现上的坑。F.kl_div(reduction=\"batchmean\") 只对 batch 维求平均，若把 (B, T, V) 的张量直接传进去，"
        "序列维会被求和，等价于把 KL 放大 block_size 倍（本实验 block=96，数值会大到完全压过 CE）。"
        "正确做法是先 logits.reshape(-1, V) 再算 KL，最后乘 T²。",
    ])

    # ------------------------------------------------ 四、步骤
    ok, t = "成功", [
        "约 5 s", "6.7 s", "约 16 s", "约 78 s（16 组）", "约 42 s（4 个步数 × 2 组）"]
    set_widths(T[2], [3.4, 7.0, 2.0, 3.2])
    no_split(T[2])
    for i in range(5):
        cell_text(T[2].rows[i + 1].cells[2], ok, align=AL.CENTER)
        cell_text(T[2].rows[i + 1].cells[3], t[i], align=AL.CENTER)
    cell_text(T[2].rows[5].cells[0], "5. 学生步数扫描（分析用）", size=9.0)
    cell_text(T[2].rows[5].cells[1], "python distill_lm.py --mode steps", size=9.0)
    cell_text(T[2].rows[5].cells[3], "约 42 s", align=AL.CENTER)
    r = add_row(T[2])
    cell_text(r.cells[0], "6. 拓展实验 A~E（选做）", size=9.0)
    cell_text(r.cells[1], "python distill_lm.py --mode ext", size=9.0)
    cell_text(r.cells[2], "成功", size=9.0, align=AL.CENTER)
    cell_text(r.cells[3], "约 100 s", size=9.0, align=AL.CENTER)

    # ------------------------------------------------ 五、参数
    t3 = T[3]
    for i, (a, b) in enumerate([("2", "2"), ("4", "4"), ("160", "64"), ("96", "96"),
                                ("737 280（0.737 M）", "147 456（0.147 M）")]):
        cell_text(t3.rows[i + 1].cells[1], a, align=AL.CENTER)
        cell_text(t3.rows[i + 1].cells[2], b, align=AL.CENTER)
    cell_text(t3.rows[6].cells[2], "5.0×（0.737 M / 0.147 M）", align=AL.CENTER)

    t4 = T[4]
    hp = [("4", "默认 4；消融扫 1/2/4/8"),
          ("0.5", "默认 0.5；消融扫 0.3/0.5/0.7/0.9"),
          ("3e-3", "AdamW，weight_decay=0.01，梯度裁剪 1.0"),
          ("24", "每个 batch 24 条长度 96 的序列"),
          ("200", "默认 200"),
          ("120", "默认 120（步数扫描另做 60/200/400）"),
          ("0", "默认 0；学生初始化用 seed+7，两个学生完全一致"),
          ("0（A 组 0.1）", "0 表示关闭；拓展 A 用 0.1 做 FitNet 特征蒸馏")]
    for i, (v, note) in enumerate(hp):
        cell_text(t4.rows[i + 1].cells[1], v, align=AL.CENTER)
        cell_text(t4.rows[i + 1].cells[2], note, size=9.0)

    # ------------------------------------------------ 六、表 6/7/8
    t5 = T[5]
    rows5 = [
        ("教师 Teacher", "737,280", "200",
         "%.1f s" % th["time_s"], "%.4f" % th["val_loss"], "%.2f" % th["ppl"], "—"),
        ("学生·从零训练", "147,456", "120",
         "%.1f s" % s0["time_s"], "%.4f" % s0["val_loss"], "%.2f" % s0["ppl"],
         "%.3f" % s0["agree"]),
        ("学生·蒸馏训练", "147,456", "120",
         "%.1f s" % s1["time_s"], "%.4f" % s1["val_loss"], "%.2f" % s1["ppl"],
         "%.3f" % s1["agree"]),
    ]
    set_widths(T[5], [3.2, 2.3, 1.9, 1.8, 2.1, 1.8, 2.2])
    no_split(T[5])
    for i, row in enumerate(rows5):
        for j, v in enumerate(row):
            cell_text(t5.rows[i + 1].cells[j], v, size=9.0, align=AL.CENTER)

    no_split(T[7])

    t6 = T[6]
    for i, Tval in enumerate([1, 2, 4, 8]):
        cell_text(t6.rows[i + 1].cells[0], "T=%d" % Tval, align=AL.CENTER)
        for j, aval in enumerate([0.3, 0.5, 0.7, 0.9]):
            v = [r for r in abl if r["T"] == Tval and abs(r["alpha"] - aval) < 1e-9][0]
            cell_text(t6.rows[i + 1].cells[j + 1], "%.4f" % v["val_loss"], align=AL.CENTER)

    t7 = T[7]
    for i, (label, key) in enumerate([("教师 Teacher", "teacher"),
                                      ("学生·从零训练", "scratch"),
                                      ("学生·蒸馏训练", "kd")]):
        cell_text(t7.rows[i + 1].cells[0], label, align=AL.CENTER)
        cell_text(t7.rows[i + 1].cells[1], allres["sample"][key])

    # ------------------------------------------------ 七、7.1
    replace_table_with_picture(doc, T[8], os.path.join(RUNS, "curves.png"), 15.6)
    set_para(P[19],
             "图 1 左图是三条训练损失曲线。教师（蓝）在前 30 步就从 5.83 降到 1.4 附近，200 步时"
             "训练损失约 0.77；两个学生都从 5.79 起步，下降速度明显更慢，120 步时停在 0.88 附近。"
             "从零（橙）与蒸馏（绿）两条曲线在全程几乎重合，蒸馏曲线只在 40 步之后稳定地低一点点——"
             "这说明 120 步之内两者的“训练集拟合程度”相近，差别主要体现在留出集上。")
    add_block(P[19], [
        "右图的柱形把留出集差距摆出来：教师 0.9986（PPL 2.71），从零学生 1.5060（PPL 4.51），"
        "蒸馏学生 1.4612（PPL 4.31）。参数量只有教师 1/5 的学生追不平教师（仍差 0.46），"
        "但蒸馏把它往教师的方向拉了 0.045。教师曲线末端的抖动来自每个 iter 只记一次 batch 损失，"
        "而学生曲线更平滑，说明小模型在 120 步内还处在欠拟合、损失单调下降的阶段。",
    ])

    # ------------------------------------------------ 七、7.2
    set_para(P[21],
             "同参数量（147 456）、同步数（120）、同初始化（seed+7）、同数据顺序下，蒸馏把留出集 loss "
             "从 1.5060 降到 1.4612（−0.0448，相对下降 3.0%），PPL 从 4.51 降到 4.31（−4.3%），"
             "与教师的 top-1 一致率从 0.619 升到 0.684（+6.5 个百分点）。三个指标同向改善，说明学生"
             "确实学到了教师分布里的结构，而不仅仅是“碰巧在某些 batch 上拟合得更好”。")
    anchor = add_block(P[21], [
        "为了看清这份收益什么时候出现，额外做了学生步数扫描（60/120/200/400 步，每个步数各跑一组"
        "从零、一组蒸馏，其余设置完全相同）：60 步时 1.8021 → 1.7814（+1.1%）；120 步时 "
        "1.5060 → 1.4612（+3.0%）；200 步时 1.2513 → 1.2410（+0.8%）；400 步时 1.0200 → 1.0947"
        "（−7.3%，蒸馏反而更差）。收益在 120 步左右的欠拟合区间最大，之后迅速收窄，训练到 400 步时"
        "由正转负。",
        "这一反转很能说明问题：教师自身的留出集 loss 只有 0.9986，400 步的从零学生已经追到 1.0200，"
        "此时软标签项仍在每一步把学生的输出往教师分布上拽，而教师的分布本身带 0.9986 的误差，"
        "学生的“自由度”被这份约束吃掉，反而学不过自己直接拟合数据。所以蒸馏的收益不是单调的，"
        "它同时携带“信息”和“约束”两种成分：欠拟合时信息占优，接近教师水平后约束占优。",
    ])
    pic = add_picture(anchor, os.path.join(RUNS, "steps.png"), width_cm=12.5)
    add_caption(pic, "图 2　蒸馏收益随学生训练步数的变化（标注为相对从零的降幅）")

    # ------------------------------------------------ 七、7.3
    set_para(P[24],
             "表 7 与图 3 给出 16 组（T × α）的结果：(T=8, α=0.5) 最好，留出集 loss 1.4133，比从零学生"
             "低 6.2%，比默认的 (4, 0.5) 还低 0.048；(T=1, α=0.9) 最差，1.4989。整张表的形状很有规律："
             "固定 α 时，T=4~8 明显优于 T=1~2；固定 T 时，α=0.3~0.7 是一个平台，α=0.9 在所有 T 上都变差。")
    anchor = add_block(P[24], [
        "T 过小的现象：T=1 的四组几乎挤在 1.486~1.499，与从零学生（1.5060）差别很小。T=1 时软标签退化成"
        "接近 one-hot 的分布（教师最大概率已经在 0.6 以上），学生能拿到的类间信息被压掉了，等价于用"
        "“教师预测的硬标签”训练，信息量大幅缩水，收益自然消失。",
        "T 过大的现象：T=8 略好于 T=4，但代价是 KL 的绝对尺度变小、训练更“软”。本实验只扫到 8，"
        "从趋势看 α=0.5 时 T=8 已接近饱和，再往上（思考题 5 讨论的 T=100）分布趋于均匀，信息会被"
        "数值精度淹没。",
        "α 接近 1 变差的原因：α 越大，硬标签 CE 的权重 (1−α) 越小，学生失去“绝对正确”的锚点，"
        "只能复现教师分布，而教师自身 loss 是 0.9986 而不是 0，这份误差会被完整继承；同时梯度里的"
        "“纠错”成分（把错误类压到 0）几乎消失。实验里 α 从 0.5 升到 0.9，四个 T 的 loss 全部上升"
        "（例如 T=4：1.4612 → 1.4891），与这个解释一致。",
        "还有一个值得注意的细节：一致率随 α 单调上升（T=1 时 0.671 → 0.743），但 loss 却在 α=0.9 处"
        "变差。这说明“像教师”与“预测得准”不是同一件事：α 越大越像教师，可教师的错误也被学了过去，"
        "所以只有中等 α 才能让两个指标同时改善。",
    ])
    replace_table_with_picture(doc, T[9], os.path.join(RUNS, "heatmap.png"), 10.6)
    set_para(P[23], "图 3　T × α 消融热力图（颜色越亮 loss 越低，红框为最优组合）", indent=False)
    P[23].alignment = AL.CENTER
    pic = add_picture(anchor, os.path.join(RUNS, "ablation.png"), width_cm=11.5)
    add_caption(pic, "图 4　四个温度下留出集 loss 随软标签权重 α 的变化")

    # ------------------------------------------------ 七、7.4
    set_para(P[26],
             "三段文本都用同一提示词“月”、同一种子、temperature=0.9、top_k=20 生成 60 字。"
             "教师（0.737 M）的文本最完整：五言/七言交替，“月了一夜听，一笑而别。少年想起道：“青山可惜。””"
             "之后接“山僧溪桥独对归雁”“渔夫在荒村细雨”等意象，句式与语料里的“某某在某某处 + 动作”模板"
             "吻合，标点也正常闭合，只有开头“月了一夜听”略显生硬。")
    add_block(P[26], [
        "从零学生（0.147 M）的文本语法框架还在：“月，忽忆旧游。渔夫在溪桥孤客送去道：“归客空忆暮云了一夜守，"
        "欲语还休。””但已经出现明显崩坏：“渔夫在溪桥孤客送去”堆了三个名词、“归客空忆暮云了一夜守”把两个"
        "模板缝在一起，并且“忽忆旧游”在 60 字里复读了两次，说明它只记住了高频片段，缺少长程组织能力。",
        "蒸馏学生的文本在“像教师”这一点上更近：它也用“某某道：“……””的对话骨架，也复用了“孤舟、野渡、"
        "寒江、故人、溪桥、泪湿衣襟”这批意象，句式与教师的相似度明显高于从零学生；但仍有两处硬伤——"
        "“月依旧月正好”把两个开头粘在一起，“默然归客野渡口明”缺谓语。总体看，蒸馏学生的文本更接近教师，"
        "而不是更接近从零学生，这与一致率 0.684 > 0.619 的量化结论一致。",
    ])

    # ------------------------------------------------ 八、拓展
    add_text_after_element(P[27]._p,
                           "五项拓展（A~E）全部完成，F 因本机没有一万字以上的离线真实语料而未做。"
                           "每一项都在同一套代码里以 --mode ext 复现，结果写入 runs_distill/results_ext.json。")
    ext_rows = [
        ("A", "特征蒸馏（FitNet）", "完成，feat=0.1",
         "1.4612 → 1.4166（−3.0%，PPL 4.31→4.12）；耗时 3.9 s → 9.9 s", "+5"),
        ("B", "温度 / 权重热力图", "完成，16 组热力图",
         "最优 (T=8, α=0.5)=1.4133；最差 (T=1, α=0.9)=1.4989", "+3"),
        ("C", "自蒸馏", "完成，同构学生 400 步",
         "教师 1.0200 → 120 步蒸馏学生 1.3579（一致率 0.746）", "+4"),
        ("D", "动态量化导出", "完成，int8 动态量化",
         "674 606 B → 324 552 B（2.08×）；loss 1.4612→1.4623；前向 0.77→0.89 ms", "+5"),
        ("E", "教师规模扫描 C∈{96,160,256}", "完成，各 200 步后蒸馏",
         "教师 0.295/0.737/1.769 M，收益 +3.27%/+2.98%/+3.88%", "+4"),
        ("F", "换成真实语料", "未完成", "本机无 1 万字以上离线语料（不联网），不伪造数据", "0"),
    ]
    set_widths(T[10], [1.2, 4.0, 3.3, 5.1, 1.9])
    no_split(T[10])
    for i, row in enumerate(ext_rows):
        for j, v in enumerate(row):
            cell_text(T[10].rows[i + 1].cells[j], v, size=9.0,
                      align=AL.CENTER if j in (0, 4) else None)

    anchor = add_text_after_element(
        T[10]._tbl,
        "A 特征蒸馏：在 KD 损失之外，把学生最后一层隐状态经一个线性投影对齐到教师的隐状态，"
        "加 0.1 的 MSE 权重。留出集 loss 从 1.4612 降到 1.4166，与 B 组最优的 1.4133 相当，"
        "说明“中间层特征”确实比“输出分布”多带了一些可迁移信息；代价是每步要多跑一次学生前向"
        "和一次教师前向，耗时从 3.9 s 涨到 9.9 s。")
    anchor = add_text(
        anchor,
        "C 自蒸馏：把训练 400 步的同构学生（0.147 M，loss 1.0200）当教师，再蒸馏一个 120 步的"
        "同构学生，得到 1.3579。它用 1/3 的步数就恢复了长训学生约 74% 的差距（0.4860 → 0.3379），"
        "说明蒸馏不依赖“更大的教师”，只依赖“更强的教师”。")
    anchor = add_text(
        anchor,
        "D 动态量化：int8 把权重体积压到 2.08×，留出集 loss 只涨 0.0011（1.4612→1.4623），"
        "精度损失几乎可以忽略；但 CPU 上单条序列（1×96）的前向反而从 0.77 ms 变成 0.89 ms，"
        "因为小模型的计算量太小，反量化开销盖过了访存收益。这一条要如实写进结论："
        "量化省的是体积，不是所有场景下的时间。")
    anchor = add_text(
        anchor,
        "E 教师规模：C 从 96 增到 256，教师参数 0.295 M → 1.769 M（6 倍），教师自身 loss "
        "1.1263 → 0.9319，但学生收益只从 3.27% 走到 3.88%。学生容量（0.147 M）才是瓶颈，"
        "教师再强也只能传递有限信息；反过来，0.295 M 的弱教师也能带来 3.27% 的收益，"
        "说明“教师比学生同预算训练更强”就足以产生可蒸馏的信息。")
    pic = add_picture(anchor, os.path.join(RUNS, "teacher_scale.png"), width_cm=12.6)
    add_caption(pic, "图 5　教师规模 vs 学生蒸馏收益（左：教师与学生的留出集 loss；右：相对从零的降幅）")

    # ------------------------------------------------ 九、问题
    add_text_after_element(P[28]._p,
                           "下面四条都是本机真实踩到的坑，按“报错原文 / 原因 / 解决”记录。")
    probs = [
        ("1", "绘图时中文标题全部变成方框，日志反复刷：\n"
              "UserWarning: Glyph 26354 (\\N{CJK UNIFIED IDEOGRAPH-66F2}) missing from font(s) DejaVu Sans.",
         "WSL 里 matplotlib 的字体缓存只登记了 50 个字体（DejaVu / Liberation / Noto Mono 等），"
         "系统里虽然装了 Noto CJK .ttc，但没有被扫进 fontManager，rcParams 里写 \"Noto Serif CJK SC\" 也匹配不到。",
         "用 fm.fontManager.addfont() 直接按 .ttc 文件注册，再用 FontProperties(fname=...).get_name() "
         "取到真实族名（这里解析成 Noto Serif CJK JP）写回 rcParams[\"font.sans-serif\"]，中文恢复正常。"),
        ("2", "第一次直接把 (B, T, V) 的 logits 传给 F.kl_div(reduction=\"batchmean\")，KD 项一开始就是 480 左右，"
              "总损失 400+，loss 完全不收敛。",
         "batchmean 只除以 batch 维，序列维被整段求和，KL 被放大了 block_size=96 倍；T² 一乘更离谱。",
         "先 logits.reshape(-1, V) 再算 KL，最后乘 T²；KD 项回到 5.06 的量级，与 CE(5.79) 可比，"
         "训练立刻正常（见 distill_lm.py 的 train()）。"),
        ("3", "用 --mode all 复用 teacher.pt 后，图 1 里教师那条曲线是空的。",
         "缓存分支只 load_state_dict，训练历史 hist 没有落盘；曲线不参与评估，所以一直没暴露。",
         "教师训练结束时把 curve 写进 results_teacher.json，复用分支读回来补进 res[\"teacher\"][\"curve\"]；"
         "同时把“缓存判定”和 --retrain 分开，避免复用与重训互相干扰。"),
        ("4", "动态量化后前向反而变慢：0.77 ms → 0.89 ms。",
         "int8 Linear 在 CPU 上要先反量化再加权，1.5e5 参数、序列长 96 的模型计算量太小，"
         "省下的访存抵不过反量化的额外开销（32 个线程的并行度也用不上）。",
         "保留量化做体积对比（2.08×），把结论改写成“小模型上动态量化主要省体积、不省时间”，"
         "并说明只有在批量更大或序列更长时才会体现速度收益。"),
    ]
    set_widths(T[11], [1.5, 5.1, 4.3, 4.6])
    no_split(T[11])
    while len(T[11].rows) < len(probs) + 1:
        add_row(T[11])
    for i, row in enumerate(probs):
        for j, v in enumerate(row):
            cell_text(T[11].rows[i + 1].cells[j], v, size=9.0,
                      align=AL.CENTER if j == 0 else None)

    # ------------------------------------------------ 十、结论
    set_para(P[30],
             "本次实验在纯 CPU（WSL2/Ubuntu 22.04）上完成了知识蒸馏的全部必做任务与 A~E 五项拓展："
             "教师 0.737 M 参数、200 步、6.7 s，留出集 loss 0.9986（PPL 2.71）；学生 0.147 M"
             "（压缩 5.0×）、120 步，从零训练 1.5060（PPL 4.51，一致率 0.619），蒸馏训练 1.4612"
             "（PPL 4.31，一致率 0.684），相对下降 3.0%。16 组 T × α 消融里最优是 (T=8, α=0.5)=1.4133"
             "（相对从零 −6.2%）。")
    add_block(P[30], [
        "学到的知识：软标签的价值不在“给出答案”，而在“给出答案之间的关系”；T² 让软标签项的梯度尺度"
        "与 CE 对齐，因此改 T 不用重调 lr；kl_div 的 batchmean 只除 batch 维，必须先 reshape 成 (B·T, V)。"
        "这三条都直接来自动手复现，比看公式印象深得多。",
        "观察到的现象：蒸馏收益随训练步数先增后减（60/120/200/400 步为 +1.1%/+3.0%/+0.8%/−7.3%），"
        "说明它同时是“信息”和“约束”；一致率随 α 单调上升而 loss 在 α=0.9 处变差，说明“像教师”不等于"
        "“预测得准”；教师规模从 0.295 M 加到 1.769 M，学生收益只从 3.27% 走到 3.88%，说明瓶颈在学生容量。",
        "可改进的方向：① 把学生训练到 200~300 步并配余弦退火，观察蒸馏与从零的交叉点；② 用温度退火"
        "（前期大 T 学结构、后期小 T 学细节）替代固定 T；③ 拓展 A 已证明中间层特征有效，可以再加注意力"
        "图的蒸馏；④ 当前语料是模板生成的仿古诗句，若换成真实语料，教师能提供的信息量会更大，"
        "但需要先解决离线语料的问题（本机无网络，故 F 未做）。",
    ])

    # ------------------------------------------------ 十一、思考题
    answers = [
        "α = 1 时损失只剩 T²·KL(p_t‖p_s)，学生的梯度是 T(q_s − p_t)：它会把学生的分布往教师分布上拉，"
        "但教师的分布本身带误差（本实验教师留出集 loss 0.9986，不是 0），这份误差会被学生完整继承，"
        "学生最多只能“复现教师”，无法通过数据纠正教师。硬标签提供的是“绝对正确”的锚点：CE 的梯度"
        "（p_s − onehot）会把错误类直接压向 0，与软标签的“按教师比例分配”方向并不完全一致，两者混合"
        "才有“学结构 + 学答案”的效果。α=1 还会让训练后期梯度变小（p_t 的负类概率本来就只有 1e-3 量级，"
        "乘上 T² 后更小），收敛变慢。实验里 α=0.9 的四组在所有 T 上都比 α=0.3~0.7 差，间接印证了"
        "“软标签不能完全替代硬标签”。",
        "不带 T² 时，KL 对 logits 的梯度约为 (1/T)(q_s − p_t)，T 越大梯度越小。设 T 从 4 改成 8，"
        "梯度幅度约减半（1/8 对 1/4），若不乘 T²，等价于把学习率缩小了 2 倍，需要把 lr 乘 2 才能保持"
        "相同的更新幅度；更一般地，漏乘 T² 就要把 lr 乘 T（相对 T=1 的基准）来补偿。本实验全程乘了 T²，"
        "所以 16 组消融可以共用 lr = 3e-3，T 从 1 变到 8 不需要重新调参——这正是 T² 的设计目的："
        "让“温度”只改变软标签的平滑程度，不改变优化器的有效步长。",
        "不算。0.74 M 参数只相当于一个小型字符级语言模型（词表 322、语料 5 万字），GPT-2 small 是 124 M，"
        "差两个数量级。但蒸馏的收益并不要求教师“绝对大”，只要求它“明显强于学生在同等预算下能学到的水平”。"
        "任务五的数据正好说明这一点：教师从 0.295 M（loss 1.1263）加到 1.769 M（loss 0.9319），参数 6 倍、"
        "loss 降 0.19，学生收益只从 3.27% 走到 3.88%；而 0.295 M 的弱教师同样带来 3.27% 的收益。"
        "可见收益的上限由学生容量（0.147 M）决定，教师只要“够强”即可，教师规模与收益是明显的边际递减关系。",
        "两种解释对应两种可观测的预测，可以用三个实验区分。① 硬标签替换：把软标签换成教师的 top-1 硬标签"
        "（相当于 T→0），若收益基本保留，说明起作用的是“更干净的标签”（正则），类间结构贡献不大；"
        "② 随机教师对照：用一个只训练 10 步、几乎随机的同构模型当教师，若仍有收益，说明收益来自"
        "“平滑输出、抑制过拟合”的正则效应；③ 数据量扫描：在 1/4、1/2、1、2 倍训练语料上重复 120 步实验，"
        "若收益随数据增多而缩小，支持“正则”解释，若基本不变则支持“信息”解释。本实验已有的步数扫描"
        "（收益在欠拟合区最大、400 步时转负）更符合“正则”解释的特征，因为正则只在容量/数据受限时有帮助，"
        "过度时就会变成约束；但 α=0.3 与 0.5 在 T=8 下几乎相同（1.4151 vs 1.4133），又说明信息成分确实存在，"
        "所以更可能是两者叠加。",
        "T=100 时 softmax(z/100) ≈ (1 + z_i/100)/V：分布几乎是均匀分布加一个极小的偏置，最大概率约"
        "1/322 ≈ 0.0031，与最小概率的差别不到 1%。此时所有类的“软标签”几乎一样，KL 的数值很小，"
        "梯度在 float32 的有效位数（约 1e-7）附近被淹没，乘 T² 也只是把极小的差值放大成另一个极小的数；"
        "学生学到的信息趋近于“输出均匀分布”，蒸馏基本失效。所以温度不能无限大，它有一个由“信息可见度”"
        "和“数值精度”共同决定的甜点区间，本实验实测为 4~8。",
        "三者压缩的对象不同：蒸馏压缩“参数量与结构”（教师→更小的学生，减少层宽/深度），量化压缩"
        "“数值精度”（fp32→int8，参数量不变、字节数约 4×），剪枝压缩“结构”（删权重/通道/注意力头，"
        "参数量真的变小，但需要稀疏或结构化才能落到实际加速）。三者可以叠加，常见顺序是"
        "“先蒸馏再剪枝再量化”：蒸馏先得到容量合适的小模型，剪枝在结构上继续瘦身，量化最后把数值精度"
        "压到 int8。本实验给出了其中的两段流水线：教师 0.737 M（fp32 约 2.95 MB）→ 蒸馏 120 步得"
        "学生 0.147 M（674 606 B）→ quantize_dynamic(int8) 得 324 552 B，留出集 loss 仅从 1.4612 涨到"
        "1.4623，端到端压缩约 9.1×（参数 5.0× × 字节 2.08×），且全程可在 CPU 上跑。",
    ]
    for i, ans in enumerate(answers):
        qp = P[32 + i]
        add_text(qp, "答：" + ans)

    doc.save(TARGET)
    print("saved", TARGET)


if __name__ == "__main__":
    main()
