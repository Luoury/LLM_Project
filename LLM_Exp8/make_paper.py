# -*- coding: utf-8 -*-
"""
生成期末论文 docx（arXiv 风格，英文正文）。
以 Final-Report-Template-arXiv-style.docx 为样式与页面设置来源，正文全部重写。

个人信息只从环境变量读取，脚本本身不含姓名/学号：
    REPORT_NAME="..." REPORT_ID="..." python make_paper.py
输出：docs/期末报告_知识蒸馏剪枝量化.docx
"""
import json
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
FIGS = os.path.join(HERE, "figs")
RUNS = os.path.join(HERE, "runs_distill")
TEMPLATE = os.path.join(DOCS, "final_template.docx")
OUT = os.path.join(DOCS, "期末报告_知识蒸馏剪枝量化.docx")

NAME = os.environ.get("REPORT_NAME", "Student")
SID = os.environ.get("REPORT_ID", "0000000000")
TODAY = "November 6, 2026"
LATIN = "Times New Roman"
MONO = "Consolas"


# ============================================================
# 0. 基础工具
# ============================================================
def load(name):
    with open(os.path.join(RUNS, name), encoding="utf-8") as f:
        return json.load(f)


def style_setup(doc):
    """统一字体：西文 Times New Roman，中文宋体；标题黑色。"""
    st = doc.styles["Normal"]
    st.font.name = LATIN
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    pf = st.paragraph_format
    pf.line_spacing = 1.18
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)

    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11.5)):
        try:
            hs = doc.styles[name]
        except KeyError:
            continue
        hs.font.name = LATIN
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.color.rgb = RGBColor(0, 0, 0)
        rpr = hs.element.get_or_add_rPr()
        rpr.rFonts.set(qn("w:eastAsia"), "SimHei")
        hs.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        hs.paragraph_format.space_after = Pt(6)
        hs.paragraph_format.keep_with_next = True


def blacken_headings(doc):
    """模板标题样式带主题色，这里强制纯黑（同时清掉 themeColor）。"""
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        rPr = st.element.get_or_add_rPr()
        color = rPr.find(qn("w:color"))
        if color is None:
            color = OxmlElement("w:color")
            rPr.append(color)
        for attr in ("w:themeColor", "w:themeShade", "w:themeTint"):
            if color.get(qn(attr)) is not None:
                del color.attrib[qn(attr)]
        color.set(qn("w:val"), "000000")
        # 模板还塞了 w14:textFill(accent1)，它会盖过 w:color，必须删掉
        for child in list(rPr):
            if child.tag.endswith("}textFill"):
                rPr.remove(child)


def no_split(table):
    """表格行不跨页；首行作为重复表头。"""
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if i == 0:
            hdr = OxmlElement("w:tblHeader")
            trPr.append(hdr)


def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def para(doc, text="", size=None, bold=False, italic=False, align=None,
         space_after=None, indent_first=None, style=None, keep=False):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if indent_first:
        p.paragraph_format.first_line_indent = Cm(indent_first)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def rich(doc, pieces, align=None, space_after=None, indent_first=None, size=None):
    """pieces: [(text, bold, italic)]，用来在正文里排数学符号。"""
    p = doc.add_paragraph()
    for text, b, i in pieces:
        r = p.add_run(text)
        r.bold = b
        r.italic = i
        r.font.size = Pt(size or 11)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if indent_first:
        p.paragraph_format.first_line_indent = Cm(indent_first)
    return p


def equation(doc, text, number):
    """居中公式 + 右对齐编号（用制表位实现）。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    pf.tab_stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run("\t")
    r.font.size = Pt(11)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(11)
    r = p.add_run("\t(%s)" % number)
    r.font.size = Pt(11)
    return p


def equation_lines(doc, lines, number):
    """多行公式：各行居中，编号置于最后一行右侧。"""
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(6 if i == 0 else 0)
        pf.space_after = Pt(6 if i == len(lines) - 1 else 0)
        width = (doc.sections[0].page_width - doc.sections[0].left_margin
                 - doc.sections[0].right_margin)
        pf.tab_stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run("	")
        r.font.size = Pt(11)
        r = p.add_run(line)
        r.italic = True
        r.font.size = Pt(10.5)
        r = p.add_run("	(%s)" % number) if i == len(lines) - 1 else p.add_run("")
        r.font.size = Pt(11)
    return p


def fig(doc, filename, width_cm=15.6, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIGS, filename), width=Cm(width_cm))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        parts = caption.split("  ")
        r = cap.add_run(parts[0] + "  ")
        r.bold = True
        r.font.size = Pt(9.5)
        r = cap.add_run("  ".join(parts[1:]))
        r.font.size = Pt(9.5)
        r.italic = True
    return p


def caption(doc, label, text, above=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8 if above else 2)
    p.paragraph_format.space_after = Pt(4 if above else 10)
    if above:
        p.paragraph_format.keep_with_next = True
    r = p.add_run(label + "  ")
    r.bold = True
    r.font.size = Pt(9.5)
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    return p


def shade(cell, color="F2F5F9"):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tcPr.append(sh)


def table(doc, header, rows, widths=None, font=9.5, align_right_from=1, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < align_right_from else WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font)
        shade(c)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < align_right_from else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            r = p.add_run(str(val))
            r.font.size = Pt(font)
            if i in bold_rows:
                r.bold = True
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    no_split(t)
    return t


def code_block(doc, lines, size=8.6):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    shade(cell, "FBFBFB")
    cell.text = ""
    first = True
    for ln in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ln)
        r.font.name = MONO
        r.font.size = Pt(size)
    no_split(t)
    return t


# ============================================================
# 1. 数据
# ============================================================
ALL = load("results_all.json")
ABL = load("results_ablate.json")
STEPS = load("results_steps.json")
EXT = load("results_ext.json")["ext"]
PIPE = load("results_pipeline.json")
BENCH = load("results_bench.json")["models"]


def grid_rows():
    rows = []
    for T in (1, 2, 4, 8):
        cells = ["$T$=%d" % T]
        for a in (0.3, 0.5, 0.7, 0.9):
            v = next(r["val_loss"] for r in ABL["ablation"] if r["T"] == T and r["alpha"] == a)
            cells.append("%.4f" % v)
        rows.append(cells)
    return rows


# ============================================================
# 2. 正文
# ============================================================
def build():
    doc = Document(TEMPLATE)
    style_setup(doc)
    blacken_headings(doc)
    clear_body(doc)

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)

    # ---------------- 封面 ----------------
    para(doc, "", space_after=60)
    para(doc, "Knowledge Distillation, Pruning and Quantization", size=19, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "for a Sub-Megaparameter Character-Level", size=19, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "Language Model", size=19, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para(doc, "A CPU-only study of the distillation → pruning → int8 pipeline, with an interactive demo",
         size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=48)

    para(doc, "Student: %s        Student ID: %s" % (NAME, SID), size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(doc, "Course: Foundations and Applications of Large Models", size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(doc, "Harbin Institute of Technology, Shenzhen", size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(doc, TODAY, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=36)
    para(doc, "Code, figures and the demonstration system: github.com/Luoury/LLM_Project (LLM_Exp8)",
         size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()

    # ---------------- 标题页信息 ----------------
    para(doc, "Knowledge Distillation, Pruning and Quantization for a "
              "Sub-Megaparameter Character-Level Language Model", size=17, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(doc, "%s, Harbin Institute of Technology, Shenzhen" % NAME, size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "Course paper · Foundations and Applications of Large Models · %s" % TODAY,
         size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Keywords:  ")
    r.bold = True
    r.font.size = Pt(10.5)
    r = p.add_run("knowledge distillation; model compression; magnitude pruning; "
                  "dynamic quantization; character-level language model; CPU inference")
    r.font.size = Pt(10.5)

    # ---------------- Abstract ----------------
    doc.add_heading("Abstract", level=1)
    para(doc,
         "Compressing a language model is usually discussed at the scale of billions of parameters, "
         "where the interesting question is how much accuracy a fixed budget of GPU hours can buy. "
         "This paper asks the mirror question on a scale that fits one CPU: given a 0.74 M-parameter "
         "character-level GPT trained from scratch on a 50 k-character corpus of classical Chinese verse, "
         "how far can it be compressed before the generated text collapses, and which stage of the "
         "pipeline actually pays for itself? We build and measure a four-stage ladder — knowledge "
         "distillation (KD) to a 0.15 M student (5.0x fewer parameters), magnitude pruning with a "
         "matched-budget fine-tuning control, dynamic int8 quantization, and an interactive web demo. "
         "KD lowers held-out cross-entropy from 1.5060 (identical student trained from scratch) to "
         "1.4612, a 3.0% relative gain, and raises top-1 agreement with the teacher from 61.9% to 68.4%. "
         "A 16-cell temperature-weight grid peaks at T=8, alpha=0.5 (1.4133, -6.2%); a training-budget "
         "sweep shows the gain is +1.1% at 60 steps, +3.0% at 120, +0.8% at 200 and -7.3% at 400, "
         "which we read as evidence that soft labels carry both extra information and an implicit "
         "regularizer. Beyond KD, the matched control reveals that most of pruning's apparent gain is "
         "the fine-tuning that follows it: 20% pruning costs only 0.0096 loss over a control trained "
         "for the same extra 120 steps, while 75% pruning costs 0.135. Dynamic int8 keeps the loss "
         "within 0.0007 and shrinks the checkpoint to 0.31 MB (9.4x smaller than the teacher), yet it is "
         "13% slower per forward pass on CPU at batch size 1 — quantization buys memory, not speed, at "
         "this scale. The final artifact generates coherent classical-Chinese pastiche in 0.31 MB. All "
         "code, figures and the zero-dependency demo are released.",
         size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 1 Introduction ----------------
    doc.add_heading("1  Introduction", level=1)
    doc.add_heading("1.1  Research Background & Problem Statement", level=2)
    para(doc,
         "Deployment, not training, is where most language models meet their users. A model that "
         "occupies three megabytes and answers in a millisecond can run inside a browser tab, a "
         "microcontroller or a classroom laptop; the same model at three gigabytes cannot. The course "
         "module on language and generative intelligence gives us the two ingredients needed to study "
         "this trade-off end to end: the Transformer decoder [1], whose cost is dominated by width and "
         "depth rather than by vocabulary, and the statistical view of a language model as a conditional "
         "distribution over the next token, which is exactly the object that knowledge distillation "
         "transfers [2].", indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Existing compression work is overwhelmingly evaluated at the scale of hundreds of millions of "
         "parameters, where a 4x reduction still leaves a usable model. At the sub-megaparameter scale "
         "the situation is different in kind: the student is small enough that its capacity, not its "
         "precision, is the binding constraint, and the usual assumption that a bigger teacher is always "
         "a better teacher stops being obviously true. The concrete question of this paper is therefore: "
         "for a 0.74 M-parameter teacher that reaches 2.71 perplexity on a 5 357-character held-out set, "
         "which compression stage contributes real quality, which contributes only size, and where does "
         "the ladder break?", indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("1.2  Research Objectives & Core Contributions", level=2)
    para(doc, "Our contributions are four-fold:", indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "(1) We implement a complete, reproducible KD pipeline for a character-level GPT (teacher, "
         "from-scratch student, distilled student) and verify it against the reference numbers of the "
         "course experiment, reproducing the from-scratch baseline to four decimal places (1.5060).",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "(2) We map the temperature-weight plane with a 16-cell grid under a protocol identical to the "
         "main experiment, so that the default cell (T=4, alpha=0.5) reproduces the main result exactly "
         "(1.4612) and the grid can be read as one experiment rather than sixteen unrelated ones.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "(3) We extend the ladder beyond distillation with magnitude pruning, a matched-budget control, "
         "dynamic int8 quantization, a student-width sweep and a teacher-size sweep, and we report the "
         "negative results (quantization slower on CPU, pruning's gain mostly attributable to "
         "fine-tuning) rather than hiding them.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "(4) We release the codebase, the exact figures of this paper and a zero-dependency interactive "
         "demo that runs three checkpoints side by side in a browser, so that every number here can be "
         "regenerated on a laptop CPU (Section 5, Section 6).",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Submission status: this manuscript has not been submitted to arXiv or to a venue; it is a "
         "course paper. The repository is public and the demo runs locally (Appendix A).",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("1.3  Paper Organization", level=2)
    para(doc,
         "Section 2 reviews related work and the course knowledge used here. Section 3 formalizes the "
         "task and describes the three compression stages. Section 4 reports the experiments. Section 5 "
         "documents reproducibility and the repository, Section 6 the demonstration system, Section 7 "
         "discusses findings and limitations, and Section 8 concludes.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 2 Related Work ----------------
    doc.add_heading("2  Related Work", level=1)
    doc.add_heading("2.1  Course Fundamental Knowledge Review", level=2)
    para(doc,
         "Three course ideas carry the whole paper. First, a decoder-only Transformer computes, for each "
         "position, a distribution over the vocabulary; the softmax temperature rescales logits before "
         "normalization, and the 1/T factor in the gradient of the cross-entropy is what motivates the "
         "T^2 correction in the distillation loss [2]. Second, the Kullback-Leibler divergence between "
         "two categorical distributions is zero exactly when they coincide, which makes it the natural "
         "supervision signal when the target is a distribution rather than a label. Third, the "
         "bias-variance reading of regularization: any auxiliary signal that keeps the model from "
         "fitting idiosyncrasies of a finite sample trades training fit for held-out fit, and therefore "
         "has an optimum in its strength.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("2.2  State-of-the-Art Research Progress", level=2)
    para(doc,
         "Knowledge distillation was introduced by Hinton et al. [2] as a way to transfer the dark "
         "knowledge encoded in soft targets; FitNet [3] added intermediate-layer supervision, and the "
         "DistilBERT/TinyBERT line [4, 5] scaled the recipe to pretrained Transformers. Pruning has two "
         "families: unstructured magnitude pruning [6], which removes individual weights and therefore "
         "needs sparse kernels to pay off, and structured pruning, which removes channels or heads and "
         "speeds up dense hardware. Quantization [7] reduces the number of bits per weight; dynamic "
         "quantization quantizes weights ahead of time and activations on the fly, which is attractive "
         "for CPU deployment because it needs no calibration set. The combination distilled-then-pruned-"
         "then-quantized is the standard deployment ladder for edge models [8].",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "The gap this paper addresses is evaluative rather than algorithmic: at the sub-megaparameter "
         "scale, published ladders rarely report a matched-budget control for pruning, and rarely report "
         "latency alongside size for int8 on CPU. Both omissions change the conclusion, as Section 4 "
         "shows.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("2.3  Correlation Between This Research and the Course", level=2)
    para(doc,
         "The experiment sits at the intersection of the language-intelligence and generative-"
         "intelligence modules: the model is a language model, the supervision is a distribution over "
         "the vocabulary, and the compression stages are the deployment-side counterpart of generation. "
         "It also exercises the empirical discipline the course emphasizes — identical initialization, "
         "identical batch order, identical step count — without which the 3% effect studied here would "
         "be indistinguishable from seed noise.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 3 Methodology ----------------
    doc.add_heading("3  Methodology / Proposed Method", level=1)
    doc.add_heading("3.1  Problem Definition & Symbol Explanation", level=2)
    para(doc,
         "Let the corpus be a character sequence and let the vocabulary have size V=322. A causal "
         "Transformer with parameters theta maps a context window of B=96 characters to logits z in "
         "R^{B x V}. Writing p_theta(y|x) = softmax(z) for the predicted next-character distribution, "
         "the training objective on a batch of token sequences y is the mean token-level cross-entropy",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    equation(doc, "L_CE(theta) = -(1/B) * sum_b log p_theta(y_b | x_b)", "1")
    para(doc,
         "Let the teacher be theta_t (0.74 M parameters) and the student theta_s (0.15 M). Distillation "
         "replaces the one-hot target by the teacher's tempered distribution "
         "p_t^T = softmax(z_t / T) and trains the student to match it while still predicting the true "
         "character:",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    equation(doc, "L = (1 - alpha) * L_CE(theta_s) + alpha * T^2 * KL(p_t^T || p_s^T)", "2")
    para(doc,
         "The factor T^2 is not cosmetic. Because d softmax(z/T)/dz scales as 1/T, the gradient of the "
         "KL term with respect to the student logits scales as 1/T as well; multiplying by T^2 restores "
         "its magnitude to that of the cross-entropy term, so that alpha keeps a stable meaning and the "
         "learning rate does not have to be retuned when T changes:",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    equation(doc, "dKL/dz_s = (1/T) * (p_s^T - p_t^T),   hence   T^2 * dKL/dz_s = T * (p_s^T - p_t^T)",
             "3")
    para(doc,
         "A practical subtlety that cost us one debugging cycle: PyTorch's kl_div with "
         "reduction='batchmean' averages only over the batch dimension. Passing a (B, T, V) tensor "
         "therefore sums over the sequence dimension and multiplies the KL term by the block size (96 "
         "here, and T on top of that), which swamps the cross-entropy term and prevents convergence. The "
         "correct call reshapes to (B*T, V) first:",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    equation_lines(doc, [
        "KL = F.kl_div( log_softmax(z_s.reshape(-1, V) / T),",
        "                softmax(z_t.reshape(-1, V) / T), reduction='batchmean' ) * T^2"], "4")

    doc.add_heading("3.2  Course Theoretical Preliminaries", level=2)
    para(doc,
         "Two properties of the softmax make distillation work at all. (i) For large T the tempered "
         "distribution converges to the uniform distribution, so the teacher reveals the relative order "
         "of all V logits but no sharp preference; for T = 1 it degenerates to the usual prediction. The "
         "useful range is where the negative logits, invisible at T = 1, are lifted into a numerically "
         "meaningful range — in our grid, T in {4, 8}. (ii) Because KL is asymmetric, matching the "
         "teacher's small probabilities is penalized more than matching its large ones, which is what "
         "encourages the student to reproduce the teacher's ranking of unlikely characters rather than "
         "only its argmax. This is the mechanism behind the observed increase in top-1 agreement at "
         "fixed loss.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Compression is then read through the same lens. Magnitude pruning removes the weights whose "
         "absolute value is smallest, on the hypothesis that the function is locally insensitive to "
         "them; because the surviving weights are unchanged, the pruned model starts from a loss close "
         "to the original and a short fine-tuning recovers most of the damage. Dynamic int8 quantization "
         "replaces each weight by an 8-bit integer plus a per-tensor scale, which is a uniform "
         "quantizer with a rounding error of at most half a step; the resulting perturbation is small "
         "relative to the weight distribution, which is why the loss moves by less than 0.001.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("3.3  Core Model & Algorithm Design", level=2)
    para(doc,
         "Fig. 1 shows the pipeline. A 0.74 M teacher is trained on the corpus; a 0.15 M student with "
         "the same depth but half the width is trained twice — once from scratch and once against the "
         "teacher's soft labels (Eq. 2) — and the distilled student is then pruned and quantized. The "
         "three checkpoints share the tokenizer and the block size, so they can be served by the same "
         "generation code and compared token by token.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    fig(doc, "fig1_arch.png", 15.6,
        "Fig. 1  The compression ladder studied in this paper. Corpus → teacher (0.74 M) and student "
        "(0.15 M) → distillation on soft labels → magnitude pruning → dynamic int8 → interactive demo. "
        "Every stage runs on CPU.")

    para(doc,
         "Algorithm 1 summarizes the training and evaluation protocol. The invariance that matters is "
         "the seed: both students start from the same initialization (torch.manual_seed(7) at "
         "construction) and see the same batch order, so the only difference between the scratch and "
         "distilled runs is the loss function.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Algorithm 1", "Distillation, pruning and quantization on a single CPU.")
    code_block(doc, [
        "Input: corpus C (50k chars), held-out H (5.4k chars), vocab V=322, block B=96",
        "Hyper: T=4, alpha=0.5, lr=3e-3, batch=24, teacher 200 iters, student 120 iters, seed=0",
        "",
        " 1  train teacher  theta_t <- AdamW(argmin L_CE)                      # 200 iters, 6.7 s",
        " 2  for mode in {scratch, distill}:",
        " 3      init student theta_s with seed 7                              # identical init",
        " 4      for it in 1..120:",
        " 5          (x, y) ~ C                                               # identical batches",
        " 6          z_t <- theta_t(x)                                        # teacher logits",
        " 7          loss <- CE(theta_s(x), y)                                # mode = scratch",
        " 8          loss <- (1-alpha)*CE + alpha*T^2*KL_reshape(z_t/T, z_s/T) # mode = distill",
        " 9          theta_s <- AdamW.step(loss)",
        "10  evaluate: held-out CE, PPL, top-1 agreement with the teacher",
        "11  masks <- |w| >= quantile(|w|, ratio);  apply;  fine-tune 120 iters (CE only)",
        "12  q <- quantize_dynamic(theta_s, {Linear}, qint8)                  # 0.31 MB",
        "13  serve theta_t, theta_s, q in the demo; measure latency (batch=1, 1 and 8 threads)",
    ])

    doc.add_heading("3.4  Experimental Implementation Details", level=2)
    para(doc,
         "All experiments run on a single x86 CPU (Intel Core i7-14650HX, 12 cores / 24 threads) inside "
         "WSL2 Ubuntu 22.04.5, with PyTorch 2.11.0 (CPU build) and 12 intra-op threads. The model, the "
         "corpus generator, the tokenizer and the training loop are implemented from scratch in a single "
         "file (distill_lm.py, about 900 lines) with no pretrained weights and no network access. "
         "Table 1 lists the configuration.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 1", "Model and training configuration. Teacher and student share the tokenizer, "
                            "block size and optimizer; they differ in width.")
    table(doc, ["Item", "Teacher", "Student"], [
        ["Layers / heads", "2 / 4", "2 / 4"],
        ["Embedding width d_model", "160", "64"],
        ["Context length", "96", "96"],
        ["Vocabulary |V|", "322", "322"],
        ["Parameters", "737,280", "147,456"],
        ["Checkpoint (fp32)", "2.89 MB", "0.64 MB"],
        ["Training iterations", "200", "120"],
        ["Optimizer / lr", "AdamW / 3e-3", "AdamW / 3e-3"],
        ["Weight decay / grad clip", "0.01 / 1.0", "0.01 / 1.0"],
        ["Batch size (tokens)", "24 x 96", "24 x 96"],
        ["Distillation T / alpha", "—", "4 / 0.5"],
        ["Random seed", "0", "0 (init 7)"],
    ], widths=[5.4, 5.0, 5.0], align_right_from=1)

    # ---------------- 4 Experiments ----------------
    doc.add_heading("4  Experiments and Results", level=1)
    doc.add_heading("4.1  Experimental Setup", level=2)
    para(doc,
         "Dataset. The corpus is generated by the template-based poet of the course experiment: 3 000 "
         "lines of five- and seven-character verse (49 457 characters after tokenization) plus sixteen "
         "Tang poems. The held-out set is 300 lines from the same generator with a different seed "
         "(5 357 characters, seed 999), i.e. in-distribution text that the model never sees — the "
         "correct proxy for generalization on a synthetic corpus, since out-of-distribution Chinese "
         "would test the tokenizer rather than the model.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Metrics. (i) Held-out cross-entropy per character, in nats; (ii) perplexity, exp(CE); "
         "(iii) top-1 agreement, the fraction of held-out positions where the student's argmax equals "
         "the teacher's argmax; (iv) checkpoint size on disk; (v) single-sample forward latency, median "
         "of 100 runs after 20 warm-ups, with 1 and with 8 threads. Agreement is the metric that "
         "separates 'predicts the data well' from 'imitates the teacher', and the two diverge in "
         "Section 4.3.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Baselines and controls. The from-scratch student is the baseline for every distillation claim. "
         "For pruning we add a control trained for the same extra 120 iterations without any mask, "
         "because pruning is always followed by fine-tuning and the fine-tuning alone already improves "
         "the model. Latency is measured with a dense fp32 model on the same machine, so it compares "
         "implementations rather than hardware.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("4.2  Main Results", level=2)
    para(doc,
         "Table 2 reports the ladder. Distillation buys 3.0% relative loss (1.5060 to 1.4612) at "
         "identical parameter count and identical training budget, and 6.5 points of agreement. "
         "Pruning 20% with 120 fine-tuning iterations then reaches 1.3177 — better than the distilled "
         "student, but the matched control without any mask reaches 1.3081, so the honest reading is "
         "that pruning is nearly free (-0.0096 against the control) rather than beneficial. Dynamic "
         "int8 on top changes the loss by +0.0007 and the size by 2.1x. The end artifact is 0.307 MB, "
         "9.4x smaller than the teacher, at 1.3184 held-out loss.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 2", "Main results on the held-out set (5 357 characters). Latency is the median "
                            "of 100 single-sample forward passes at batch size 1; the from-scratch "
                            "student shares the distilled student's architecture and was not re-timed.")
    b = BENCH
    table(doc, ["Model", "Held-out CE", "PPL", "Agreement", "#Params", "Size", "1 thread", "8 threads"], [
        ["Teacher (fp32)", "%.4f" % b["teacher"]["loss"], "%.2f" % b["teacher"]["ppl"], "100.0%",
         "737,280", "%.2f MB" % b["teacher"]["mb"],
         "%.2f ms" % b["teacher"]["lat1"]["median"], "%.2f ms" % b["teacher"]["lat8"]["median"]],
        ["Student, scratch (fp32)", "1.5060", "4.51", "61.9%", "147,456", "0.64 MB", "—", "—"],
        ["Student, KD (fp32)", "%.4f" % b["student_kd"]["loss"], "%.2f" % b["student_kd"]["ppl"],
         "%.1f%%" % (b["student_kd"]["agree"] * 100), "147,456", "%.2f MB" % b["student_kd"]["mb"],
         "%.2f ms" % b["student_kd"]["lat1"]["median"], "%.2f ms" % b["student_kd"]["lat8"]["median"]],
        ["+ prune 20% (fp32)", "%.4f" % b["prune20"]["loss"], "%.2f" % b["prune20"]["ppl"],
         "%.1f%%" % (b["prune20"]["agree"] * 100), "147,456 (20% zero)", "%.2f MB" % b["prune20"]["mb"],
         "%.2f ms" % b["prune20"]["lat1"]["median"], "%.2f ms" % b["prune20"]["lat8"]["median"]],
        ["+ int8 (deployed)", "%.4f" % b["prune20_int8"]["loss"], "%.2f" % b["prune20_int8"]["ppl"],
         "%.1f%%" % (b["prune20_int8"]["agree"] * 100), "147,456", "%.3f MB" % b["prune20_int8"]["mb"],
         "%.2f ms" % b["prune20_int8"]["lat1"]["median"],
         "%.2f ms" % b["prune20_int8"]["lat8"]["median"]],
    ], widths=[3.9, 1.9, 1.2, 1.8, 2.6, 1.5, 1.5, 1.5], font=8.8, bold_rows=(4,))

    doc.add_heading("4.3  Ablation Study", level=2)
    para(doc,
         "Table 3 isolates the components. The temperature-weight grid (Section 4.4, Fig. 3a) is "
         "summarised by its best and worst cells: moving from the default (4, 0.5) to (8, 0.5) is worth "
         "0.0479 loss, more than the entire gain of distillation over the scratch baseline, while (1, 0.9) "
         "is worse than the default by 0.0377. The training-budget row is the most informative: at 400 "
         "steps the distilled student is worse than the scratch student by 7.3%, i.e. the sign of the "
         "effect flips once the student has enough budget to fit the data itself.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 3", "Ablation of the individual stages. Delta is measured against the distilled "
                            "student at the default setting (1.4612); the pruning rows are measured "
                            "against both the distilled student and the matched control.")
    table(doc, ["Variant", "Held-out CE", "Delta", "Note"], [
        ["Student, KD (T=4, alpha=0.5)", "1.4612", "—", "default of Table 2"],
        ["Student, from scratch", "1.5060", "+0.0448", "same init, same batches"],
        ["KD, best cell (T=8, alpha=0.5)", "1.4133", "-0.0479", "16-cell grid, Fig. 3a"],
        ["KD, worst cell (T=1, alpha=0.9)", "1.4989", "+0.0377", "soft labels nearly one-hot"],
        ["KD + FitNet feature loss (0.1)", "1.4166", "-0.0446", "adds one MSE on hidden states"],
        ["Self-distillation (400-step student as teacher)", "1.3579", "-0.1033", "no bigger teacher"],
        ["Teacher C=96 (0.295 M) instead of 160", "1.4568", "-0.0044", "weaker teacher"],
        ["Teacher C=256 (1.769 M) instead of 160", "1.4476", "-0.0136", "stronger teacher"],
        ["Prune 20% + 120 fine-tune iters", "1.3177", "-0.1435", "vs. -0.1531 for the control"],
        ["Control: same 120 extra iters, no mask", "1.3081", "-0.1531", "the pruning-free reference"],
        ["Prune 40% / 60% / 75%", "1.3309 / 1.3581 / 1.4431", "—", "cost grows slowly, then fast"],
        ["Prune 20% + int8", "1.3184", "-0.1428", "+0.0007 over fp32, 2.1x smaller"],
    ], widths=[6.3, 3.2, 2.3, 4.1], font=8.8)

    doc.add_heading("4.4  Visualization Analysis", level=2)
    para(doc,
         "Fig. 2 shows why the loss curves alone cannot settle the question. The teacher descends "
         "fastest and flattens near 0.8 training loss; the two students are almost superposed for the "
         "first 80 iterations, and only afterwards does the distilled curve sit consistently below the "
         "scratch curve. The gap at the end of training is small (0.045 in held-out loss), which is "
         "exactly what a 3% effect looks like on a synthetic corpus.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    fig(doc, "fig2_curves.png", 13.4,
        "Fig. 2  Training loss of the teacher (200 iterations) and of the two students (120 iterations, "
        "identical initialization and batch order).")

    para(doc,
         "Fig. 3a is the temperature-weight plane. Its shape has three readable features: a valley "
         "along alpha in {0.3, 0.5} for T in {4, 8}; a monotone degradation as alpha approaches 0.9, "
         "where the student is asked to imitate a teacher that is itself wrong 0.9986 nats per "
         "character; and a collapse at T = 1, where the soft labels are nearly one-hot and carry almost "
         "no class-similarity information. Fig. 3b makes the budget dependence explicit: the gain is "
         "largest (+3.0%) exactly where the student is still underfitting, shrinks to +0.8% at 200 "
         "iterations, and turns negative (-7.3%) at 400.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    fig(doc, "fig3_ablation.png", 15.8,
        "Fig. 3  (a) Held-out loss over the 4x4 temperature-weight grid; the red box marks the best "
        "cell (T=8, alpha=0.5, 1.4133). (b) Gain of distillation over the from-scratch student as a "
        "function of the training budget.")

    para(doc,
         "Fig. 4 places every checkpoint on the accuracy-size plane. The width sweep traces a clean "
         "Pareto front — 0.27 MB at 1.78 loss up to 0.90 MB at 1.36 — and the canonical 0.64 MB student "
         "sits on it. The pruning points fall on a vertical line: all four pruning ratios have the same "
         "file size, because a dense tensor with zeros is still a dense tensor. Only the int8 point "
         "moves left, to 0.31 MB, at essentially unchanged loss.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    fig(doc, "fig4_tradeoff.png", 12.8,
        "Fig. 4  Accuracy-size trade-off. The width sweep (green) moves along the Pareto front; pruning "
        "(purple) changes loss but not size; int8 (red) is the only stage that shrinks the checkpoint "
        "substantially.")

    doc.add_heading("4.5  Result Discussion", level=2)
    para(doc,
         "The three stages behave like three different kinds of compression, and the measurements say "
         "so. Distillation transfers a function: same architecture, same size, better loss, and a "
         "measurable increase in agreement, which is the fingerprint of imitation rather than of extra "
         "fitting. Pruning removes parameters but, in dense storage, not bytes; its effect on loss is "
         "small at 20-40% and only becomes expensive above 60%, consistent with the redundancy of an "
         "over-parameterized MLP. Quantization removes precision and is nearly lossless here because "
         "the weights are small and roughly Gaussian, so 8 bits per weight still resolve them finely.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Two results deserve emphasis because they contradict a naive reading of the pipeline. First, "
         "pruning looks like a 0.14 improvement over the distilled student, but 0.15 of that comes from "
         "the fine-tuning steps that pruning is always paired with; without the control, one would "
         "credit pruning with a gain it does not produce. Second, int8 is slower, not faster, on CPU at "
         "batch size 1 (0.76 ms versus 0.67 ms, and 1 135 versus 1 669 characters per second): dynamic "
         "quantization inserts quantize/dequantize operations around every linear layer, and with 96 "
         "tokens per pass those fixed costs dominate the saved multiply-adds. The benefit is memory, "
         "and it would turn into a speed benefit only at larger batches or with integer kernels.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Finally, the budget sweep reframes what distillation is doing at this scale. If soft labels "
         "were pure extra information, more training would not hurt; if they were pure regularization, "
         "the gain would grow with overfitting. What we see — a peak in the underfitting regime and a "
         "reversal once the student approaches the teacher's loss — is the signature of both effects at "
         "once: the teacher's distribution regularizes early and misleads late, when its own 0.9986-nat "
         "error becomes the binding term.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 5 Reproducibility ----------------
    doc.add_heading("5  Reproducibility and Reusable Code", level=1)
    para(doc,
         "Every number in this paper is produced by four scripts in a single directory, with fixed "
         "seeds and no external downloads. The repository layout is:",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    code_block(doc, [
        "LLM_Exp8/",
        "├── distill_lm.py      # model, corpus, tokenizer, KD training, ablations, plots",
        "├── pipeline.py        # magnitude pruning + int8 + width sweep -> Table 2/3",
        "├── make_deploy.py     # exports the two deployable checkpoints, latency benchmark",
        "├── paper_figs.py      # Fig. 1-4 (300 dpi, English labels)",
        "├── demo/server.py     # zero-dependency demo backend (stdlib http.server)",
        "├── demo/index.html    # single-page front end (Fig. 5)",
        "├── runs_distill/      # checkpoints (.pt) and results_*.json",
        "└── docs/              # this report and the course experiment report",
    ])
    para(doc,
         "One-command reproduction (about three minutes on 12 CPU threads):",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    code_block(doc, [
        "git clone git@github.com:Luoury/LLM_Project.git && cd LLM_Project/LLM_Exp8",
        "python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt",
        "python distill_lm.py --mode teacher --retrain   # 7 s   -> teacher.pt",
        "python distill_lm.py --mode all                 # 16 s  -> Table 2 (KD vs scratch)",
        "python distill_lm.py --mode ablate              # 78 s  -> Fig. 3a (16 cells)",
        "python distill_lm.py --mode steps               # 42 s  -> Fig. 3b",
        "python distill_lm.py --mode ext                 # 100 s -> self-distill, C-sweep, int8",
        "python pipeline.py && python make_deploy.py     # 90 s  -> prune/int8, Fig. 4",
        "python paper_figs.py                            # Fig. 1-4",
        "python demo/server.py                           # http://localhost:8600  (Fig. 5)",
    ])
    para(doc,
         "Randomness is fixed at three levels: Python and PyTorch seeds for weight initialization and "
         "batch order, a fixed corpus generator seed (0 for training, 999 for held-out), and a fixed "
         "torch.Generator for the ablation batches. The ablation protocol reuses the same train() "
         "function and the same teacher checkpoint as the main experiment, which is why the default "
         "cell reproduces 1.4612 exactly; we consider this a correctness check rather than a "
         "convenience. Total compute for the whole paper is under four CPU-minutes of training plus "
         "about one minute of benchmarking, so the results can be re-verified on any laptop.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 6 Demo ----------------
    doc.add_heading("6  System Demonstration", level=1)
    para(doc,
         "To show that the compressed model is usable rather than merely measurable, we built a small "
         "interactive system that serves the three checkpoints side by side. It deliberately avoids "
         "heavy frameworks: the backend is Python's standard-library HTTP server, the front end is one "
         "HTML file with a few hundred lines of JavaScript, and the only dependency is PyTorch. This "
         "keeps the demo runnable on the same CPU that produced the results, which is the point of the "
         "compression.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 4", "Demonstration system architecture.")
    table(doc, ["Component", "Technology", "Function"], [
        ["Model runtime", "PyTorch 2.11 (CPU), 8 threads",
         "Loads teacher.pt / student_kd.pt / student_prune20_int8.pt; greedy-top-k sampling"],
        ["Backend", "Python http.server (stdlib)",
         "GET / serves the page; POST /api/generate runs the three models with a shared seed"],
        ["Front end", "HTML + CSS + vanilla JS",
         "Prompt, temperature, top-k and length controls; three output cards; live latency"],
        ["Evaluation", "Pre-computed JSON",
         "Held-out loss, PPL, agreement and checkpoint size shown next to the generated text"],
        ["Deployment", "localhost:8600, no network",
         "Single process, no GPU, no external API; starts in about ten seconds"],
    ], widths=[3.2, 5.4, 7.3], font=9.0)

    para(doc,
         "A walkthrough: the user types a prompt (default 月), optionally moves the temperature and "
         "top-k sliders, and presses Generate. The same seed is applied to all three models, so the "
         "three outputs differ only because the weights differ — a live version of the comparison in "
         "Table 2. The page then reports the measured single-thread forward latency for each model and "
         "the offline metrics from the benchmark file.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    fig(doc, "fig5_demo_2x.png", 16.0,
        "Fig. 5  The demonstration system: identical prompt and seed, three checkpoints, and the "
        "measured metrics underneath (teacher 2.89 MB / 0.9986; KD student 0.64 MB / 1.4612; pruned "
        "20% + int8 0.31 MB / 1.3184).")

    # ---------------- 7 Discussion ----------------
    doc.add_heading("7  Discussion", level=1)
    doc.add_heading("7.1  In-depth Result Interpretation", level=2)
    para(doc,
         "The compression ladder is often drawn as a straight line from a large model to a small one. "
         "Our measurements suggest it is better drawn as three orthogonal moves: distillation changes "
         "the function at fixed size, pruning changes the parameter count at fixed size (in dense "
         "storage), and quantization changes the size at fixed function. Only the third move reduced "
         "the artifact on disk; only the first improved the loss at equal budget. This decomposition "
         "explains the otherwise puzzling numbers in Table 2, where the smallest model is not the "
         "worst one and the pruned model beats the model it was pruned from.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "The agreement column adds a second axis. At alpha = 0.9 the student imitates the teacher more "
         "closely (agreement rises with alpha, from 67% at 0.3 to 70% at 0.9 at T=4) while its loss gets "
         "worse — the student is learning the teacher's errors along with its structure. This is the "
         "cleanest evidence in our data that 'predicting the data' and 'imitating the teacher' are "
         "different objectives, and that a good distillation recipe has to balance them.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("7.2  Advantages & Limitations", level=2)
    para(doc,
         "Advantages. The study is fully reproducible on commodity hardware, it reports negative "
         "results (int8 slower, pruning's gain mostly fine-tuning), and it separates the three "
         "compression axes with matched controls. The artifacts are small enough to be inspected: a "
         "0.31 MB checkpoint that still produces classical-Chinese pastiche is a tangible demonstration "
         "of what each stage contributes.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para(doc,
         "Limitations. (i) The corpus is synthetic and small (50 k characters, 322 symbols), so the "
         "absolute losses are not comparable to natural-language results and the 3% effect is of the "
         "same order as seed noise for some configurations (the width sweep shows standard deviations "
         "up to 0.03). (ii) Unstructured pruning was evaluated with dense storage, so its size benefit "
         "is zero; a sparse format would only pay off above roughly 50% sparsity, where 8 bytes per "
         "non-zero (value plus index) fall below 4 bytes per dense element. (iii) The latency numbers "
         "come from one machine and exclude tokenization and HTTP overhead; they are indicative, not "
         "benchmarks. (iv) We did not evaluate structured pruning, quantization-aware training or "
         "knowledge distillation from a pretrained model, each of which would change the picture at "
         "larger scale.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("7.3  Course Learning Reflection", level=2)
    para(doc,
         "The most valuable part of this project was not the final number but the debugging that "
         "produced it. Three failures were instructive: the kl_div reduction trap, which taught us to "
         "read a library's reduction semantics instead of trusting its name; the empty teacher curve "
         "after reusing a checkpoint, which taught us that caching a model and caching its training "
         "history are different things; and the matched-budget control, which taught us that an "
         "improvement must be attributed to the change that caused it, not to the procedure that "
         "surrounded it. All three are course lessons — numerical care, experimental hygiene, and the "
         "bias-variance view of regularization — encountered as bugs rather than as slides.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("7.4  Future Work", level=2)
    para(doc,
         "Four directions seem natural. (i) Replace magnitude pruning with structured pruning (heads or "
         "channels) so that the parameter reduction becomes a latency reduction. (ii) Add quantization-"
         "aware fine-tuning, which typically recovers the small int8 gap and may also make the int8 "
         "path faster with integer kernels. (iii) Scale the teacher to a pretrained model and check "
         "whether the budget-sweep reversal persists, which would test the 'information plus "
         "regularizer' reading on a stronger teacher. (iv) Turn the demo into a small agent that "
         "composes verses under user constraints (rhyme, tone pattern), which would connect the "
         "compression work to the agent-intelligence module of the course.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- 8 Conclusion ----------------
    doc.add_heading("8  Conclusion", level=1)
    para(doc,
         "We compressed a 0.74 M-parameter character-level GPT into a 0.31 MB artifact — 9.4x smaller — "
         "while keeping held-out loss within 0.32 nats of the teacher, and we measured where each stage "
         "of the ladder contributes. Distillation is the only stage that improves quality at equal "
         "budget (1.5060 to 1.4612, +6.5 points of teacher agreement); pruning at 20% is nearly free "
         "but, in dense storage, saves no bytes; dynamic int8 saves 2.1x of memory at essentially no "
         "loss but does not speed up single-sample CPU inference. A budget sweep showing the "
         "distillation gain peaking at 120 steps and reversing at 400 suggests that soft labels act as "
         "information and as a regularizer at the same time, with the balance set by how far the student "
         "still is from the teacher. The full pipeline, the figures and an interactive demo are "
         "released, and every number can be regenerated in minutes on a laptop CPU.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # ---------------- References ----------------
    doc.add_heading("References", level=1)
    refs = [
        "[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need [C]//Advances in Neural "
        "Information Processing Systems (NeurIPS), 2017: 5998-6008.",
        "[2] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network [J]. arXiv "
        "preprint arXiv:1503.02531, 2015.",
        "[3] Romero A, Ballas N, Kahou S E, et al. FitNets: hints for thin deep nets [C]//ICLR, 2015.",
        "[4] Sanh V, Debut L, Chaumond J, Wolf T. DistilBERT, a distilled version of BERT: smaller, "
        "faster, cheaper and lighter [J]. arXiv preprint arXiv:1910.01108, 2019.",
        "[5] Jiao X, Yin Y, Shang L, et al. TinyBERT: distilling BERT for natural language "
        "understanding [C]//Findings of EMNLP, 2020: 4163-4174.",
        "[6] Han S, Pool J, Tran J, Dally W. Learning both weights and connections for efficient neural "
        "networks [C]//NeurIPS, 2015: 1135-1143.",
        "[7] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for "
        "efficient integer-arithmetic-only inference [C]//CVPR, 2018: 2704-2713.",
        "[8] Gholami A, Kim S, Dong Z, et al. A survey of quantization methods for efficient neural "
        "network inference [M]//Low-Power Computer Vision. Chapman and Hall/CRC, 2022: 291-326.",
        "[9] Course Teaching Group. Foundations and Applications of Large Models: course materials "
        "[Z]. Harbin Institute of Technology, Shenzhen, 2026.",
        "[10] Goodfellow I, Bengio Y, Courville A. Deep Learning [M]. MIT Press, 2016.",
    ]
    for r in refs:
        p = para(doc, r, size=9.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.paragraph_format.space_after = Pt(3)

    # ---------------- Appendix ----------------
    doc.add_heading("Appendix", level=1)
    doc.add_heading("A. Submission Status", level=2)
    para(doc,
         "This paper has not been submitted to arXiv, a conference or a journal. It is the final report "
         "of the course Foundations and Applications of Large Models. The code is public at "
         "github.com/Luoury/LLM_Project (directory LLM_Exp8) and the demonstration system runs locally "
         "with no network access; the repository history was rewritten to remove personal documents "
         "before publication.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    doc.add_heading("B. Core Code Snippet", level=2)
    para(doc, "The distillation loss, including the reshape that makes kl_div's batchmean correct:",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    code_block(doc, [
        "# distill_lm.py  (training step, mode = 'kd')",
        "logits_s, _ = model(xb)                       # (B, T, V)",
        "with torch.no_grad():",
        "    logits_t, _ = teacher(xb)                 # (B, T, V)",
        "ce = F.cross_entropy(logits_s.reshape(-1, V), yb.reshape(-1))",
        "kd = F.kl_div(",
        "        F.log_softmax(logits_s.reshape(-1, V) / T, dim=-1),",
        "        F.softmax(logits_t.reshape(-1, V) / T, dim=-1),",
        "        reduction='batchmean') * (T ** 2)     # reshape first, then T^2",
        "loss = (1 - alpha) * ce + alpha * kd",
    ])

    doc.add_heading("C. Supplementary Results", level=2)
    para(doc, "The full 4x4 grid behind Fig. 3a. Every cell is an independent 120-iteration run with the "
              "same teacher, initialization and batch order; the default cell (T=4, alpha=0.5) "
              "reproduces the main result 1.4612, which validates the protocol.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 5", "Held-out cross-entropy for all 16 temperature-weight combinations.")
    table(doc, ["Temperature T", "alpha = 0.3", "alpha = 0.5", "alpha = 0.7", "alpha = 0.9"],
          [["T = %d" % T] + row[1:] for T, row in zip((1, 2, 4, 8), grid_rows())],
          widths=[3.4, 3.2, 3.2, 3.2, 3.2], font=9.2)
    para(doc, "Student width sweep (KD, three seeds each):",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    caption(doc, "Table 6", "Capacity sweep: student width versus quality, size and measured latency.")
    table(doc, ["d_model", "#Params", "Held-out CE (mean +- std)", "PPL", "Size", "1 thread"], [
        ["%d" % w["width"], "{:,}".format(w["params"]),
         "%.4f +- %.4f" % (w["loss"], w["loss_std"]), "%.2f" % w["ppl"],
         "%.2f MB" % w["size_mb"], "%.2f ms" % w["latency_ms"]] for w in PIPE["width"]
    ], widths=[2.2, 2.8, 5.0, 1.8, 2.2, 2.2], font=9.0)
    para(doc, "All runs are on the same CPU with 12 threads; the checkpoints sum to less than 15 MB, "
              "which is why the entire study fits in a repository.",
         indent_first=0.6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    os.makedirs(DOCS, exist_ok=True)
    doc.save(OUT)
    print("[saved]", OUT)
    print("[stats] paragraphs=%d tables=%d" % (len(doc.paragraphs), len(doc.tables)))


if __name__ == "__main__":
    build()
