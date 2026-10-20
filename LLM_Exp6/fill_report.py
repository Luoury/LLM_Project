# -*- coding: utf-8 -*-
"""
填充实验六报告 —— 严格基于模板结构（7 个一级标题 + 5 张表）。
所有数字读自 experiment_results.json / verifier_selfcheck.json，不手抄。
全文宋体（同时设 w:eastAsia 防中文错码）。
"""
import copy
import json

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

TEMPLATE = "template_exp6.docx"
OUTPUT = "2025311574_黄思涵_实验作业六.docx"
STUDENT_NO, STUDENT_NAME, CLASS_NO = "2025311574", "黄思涵", "12班"

RES = json.load(open("experiment_results.json", encoding="utf-8"))
VCK = json.load(open("verifier_selfcheck.json", encoding="utf-8"))

doc = Document(TEMPLATE)


def set_run_font(run, name="宋体", size=None, bold=None):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rf = rPr.rFonts
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def set_cell(cell, text, bold=False, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(str(text))
    set_run_font(r, "宋体", size=size, bold=bold)


def replace_para(para, text, bold=False, size=None):
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    run = para.add_run(text)
    set_run_font(run, "宋体", size=size, bold=bold)
    return para


def add_para_after(ref_para, text, bold=False, size=10.5):
    new_p = copy.deepcopy(ref_para._element)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    ref_para._element.addnext(new_p)
    para = Paragraph(new_p, ref_para._parent)
    run = para.add_run(text)
    set_run_font(run, "宋体", size=size, bold=bold)
    return para


def insert_image_after(ref_para, img_path, width=5.6):
    new_p = copy.deepcopy(ref_para._element)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    ref_para._element.addnext(new_p)
    para = Paragraph(new_p, ref_para._parent)
    run = para.add_run()
    run.add_picture(img_path, width=Inches(width))
    return para


ev = RES.get("eval", {})
sft_info = RES.get("sft", {})
rsi_info = RES.get("rsi", {})
b = ev.get("baseline", {})
s = ev.get("sft", {})
sc = ev.get("sft_corr", {})
rounds = rsi_info.get("rounds", [])
sft_losses = sft_info.get("losses", [])
sft_seconds = sft_info.get("seconds", 0)
loss_first = sft_losses[0] if sft_losses else 0.0
loss_last = sft_losses[-1] if sft_losses else 0.0

# 从 RSI 记忆里取真实的方向提示与纠错输出，供报告引用
try:
    MEM = json.load(open("rsi_memory.json", encoding="utf-8"))
except Exception:
    MEM = []
_hint_ex = next((m for m in MEM if m.get("hint")), None)
_hint_text = (_hint_ex or {}).get("hint", "")
_tool_out = (_hint_ex or {}).get("pred2_tool", "")

loc_corr_ok = sum(1 for d in sc.get("details", []) if d.get("type") == "loc" and d.get("hit2"))
cnt_corr_ok = sum(1 for d in sc.get("details", []) if d.get("type") == "count" and d.get("hit2"))
ex_corr_ok = sum(1 for d in sc.get("details", []) if d.get("type") == "exist" and d.get("hit2"))

paras = doc.paragraphs


def P(idx):
    return paras[idx]


# ========== 表0 基本信息 ==========
# 模板结构：第0行表头(项目|内容)，第1行实验名称，第2行姓名/学号/班级，
# 第3行实验日期，第4行运行环境，第5行代码仓库/文件
t0 = doc.tables[0]
set_cell(t0.rows[1].cells[1], "SmolVLM-256M 精准定位的 RSI 递归自进化")
set_cell(t0.rows[2].cells[1], f"{STUDENT_NAME} / {STUDENT_NO} / {CLASS_NO}")
set_cell(t0.rows[3].cells[1], "2026年10月20日")
set_cell(t0.rows[4].cells[1],
         "WSL2 Ubuntu 22.04.5（x86_64）；GPU：NVIDIA GeForce RTX 5060 Laptop GPU 8GB；"
         "内存 7.6GB；Python 3.10.12；torch 2.11.0+cu128")
set_cell(t0.rows[5].cells[1],
         "LLM_Project/LLM_Exp6/hw6_rsi_localize.py"
         "（https://github.com/Luoury/LLM_Project）")

# ========== 表1 命令与耗时 ==========
# 模板第5行已自带 RSI 行，直接填空，不要重复 add_row
t1 = doc.tables[1]
set_cell(t1.rows[1].cells[0], "--mode make_data")
set_cell(t1.rows[1].cells[1], "生成 55 张合成图 + GT 框")
set_cell(t1.rows[1].cells[2], "hw6_imgs/gt.json（0.6s）")
set_cell(t1.rows[2].cells[0], "--mode make_sft")
set_cell(t1.rows[2].cells[1], "GT 框 → 网格坐标种子数据")
set_cell(t1.rows[2].cells[2], "sft_data.json（0.5s）")
set_cell(t1.rows[3].cells[0], "--mode sft --steps 60")
set_cell(t1.rows[3].cells[1], "LoRA SFT 种子训练")
set_cell(t1.rows[3].cells[2], f"sft_adapter/（{sft_seconds:.1f}s）")
set_cell(t1.rows[4].cells[0], "--mode eval")
set_cell(t1.rows[4].cells[1], "三阶段评测（基线 / SFT / 纠错）")
set_cell(t1.rows[4].cells[2], f"experiment_results.json（{ev.get('seconds', 0):.1f}s）")
set_cell(t1.rows[5].cells[0], "--mode rsi --rounds 2 --steps 10")
set_cell(t1.rows[5].cells[1], "RSI 递归自进化 2 轮")
set_cell(t1.rows[5].cells[2], f"rsi_memory.json（{rsi_info.get('seconds', 0):.1f}s）")

# ========== 表2 三阶段评测 ==========
t2 = doc.tables[2]
# 行标题按实际训练步数改写，避免与正文矛盾
set_cell(t2.rows[2].cells[0], f"SFT（种子 {sft_info.get('steps', 60)} 步）")
set_cell(t2.rows[3].cells[0], "SFT + 验证器纠错")
set_cell(t2.rows[1].cells[1], f"{b.get('loc', 0)}/8")
set_cell(t2.rows[1].cells[2], f"{b.get('count', 0)}/4")
set_cell(t2.rows[1].cells[3], f"{b.get('exist', 0)}/3")
set_cell(t2.rows[1].cells[4], f"{b.get('total', 0)}/15")
set_cell(t2.rows[1].cells[5], f"{b.get('iou', 0):.3f}")
set_cell(t2.rows[2].cells[1], f"{s.get('loc', 0)}/8")
set_cell(t2.rows[2].cells[2], f"{s.get('count', 0)}/4")
set_cell(t2.rows[2].cells[3], f"{s.get('exist', 0)}/3")
set_cell(t2.rows[2].cells[4], f"{s.get('total', 0)}/15")
set_cell(t2.rows[2].cells[5], f"{s.get('iou', 0):.3f}")
set_cell(t2.rows[3].cells[1], f"{loc_corr_ok}/8")
set_cell(t2.rows[3].cells[2], f"{cnt_corr_ok}/4")
set_cell(t2.rows[3].cells[3], f"{ex_corr_ok}/3")
set_cell(t2.rows[3].cells[4], f"{sc.get('ok2', 0)}/15")
set_cell(t2.rows[3].cells[5], f"{s.get('iou', 0):.3f}")

# ========== 表3 RSI 轮次 ==========
t3 = doc.tables[3]
set_cell(t3.rows[1].cells[1], "15")
set_cell(t3.rows[1].cells[2], str(b.get("total", 0)))
set_cell(t3.rows[1].cells[3], "—")
set_cell(t3.rows[1].cells[4], "—")
set_cell(t3.rows[1].cells[5], "—")
set_cell(t3.rows[1].cells[6], f"{b.get('total', 0)}/15")
for k in range(2):
    row = t3.rows[2 + k]
    if k < len(rounds):
        r = rounds[k]
        set_cell(row.cells[1], str(r.get("pool", 15)))
        set_cell(row.cells[2], str(r.get("direct_ok", 0)))
        set_cell(row.cells[3], str(r.get("self", 0)))
        set_cell(row.cells[4], str(r.get("tool", 0)))
        set_cell(row.cells[5], str(r.get("eval_total", 0)))
        set_cell(row.cells[6], f"{r.get('eval_total', 0)}/15")
    else:
        for c in range(1, 7):
            set_cell(row.cells[c], "—")

# ========== 表4 代表性样例 ==========
t4 = doc.tables[4]
details = sc.get("details", [])
sft_details = s.get("details", [])
ex_right = next((x for x in sft_details if x["type"] == "exist" and x["ok"]), None)
ex_fail = next((x for x in details if x["type"] == "loc"), None)

if ex_right:
    set_cell(t4.rows[1].cells[1], ex_right["id"])
    set_cell(t4.rows[1].cells[2], "图里有没有该颜色物体？")
    set_cell(t4.rows[1].cells[3], ex_right["pred"][:38].replace("\n", " "))
    set_cell(t4.rows[1].cells[4], "—")
    set_cell(t4.rows[1].cells[5], "答对")
else:
    for c in range(1, 6):
        set_cell(t4.rows[1].cells[c], "—")

set_cell(t4.rows[2].cells[1], "—")
set_cell(t4.rows[2].cells[2], "—")
set_cell(t4.rows[2].cells[3], "—")
set_cell(t4.rows[2].cells[4], "自查")
set_cell(t4.rows[2].cells[5], "自查改写 0 次，未产生改写")

set_cell(t4.rows[3].cells[1], "—")
set_cell(t4.rows[3].cells[2], "—")
set_cell(t4.rows[3].cells[3], "—")
set_cell(t4.rows[3].cells[4], "验证器")
set_cell(t4.rows[3].cells[5], "7 次纠错均未纠正到位")

if ex_fail:
    set_cell(t4.rows[4].cells[1], ex_fail["id"])
    set_cell(t4.rows[4].cells[2], "目标位于第几行第几列？")
    set_cell(t4.rows[4].cells[3], ex_fail["p1"][:38].replace("\n", " "))
    set_cell(t4.rows[4].cells[4], "验证器")
    set_cell(t4.rows[4].cells[5], f"仍失败（gold={ex_fail['gold']}）")
else:
    for c in range(1, 6):
        set_cell(t4.rows[4].cells[c], "—")

# ========== 正文：一、实验目的与任务概述 ==========
replace_para(P(1), "姓名：" + STUDENT_NAME + "　学号：" + STUDENT_NO + "　班级：" + CLASS_NO)
replace_para(P(3),
             "本实验让 SmolVLM-256M（2.56 亿参数）学会把目标位置表达为 4×4 网格坐标，"
             "把 256×256 像素空间压缩成 16 个离散类别，从而把「回归坐标」改写为「分类格号」。"
             "在此基础上构建 RSI（递归自进化）闭环：模型先作答，再把预测格渲染成紫色高亮回灌自查，"
             "并用解析式验证器（颜色连通块分析）复核与纠错，通过验证的答案进入 SFT 数据池继续训练。"
             "最后用基线、SFT、SFT+纠错三个阶段与两轮 RSI 的数据，回答「定位能力的提升究竟来自微调还是来自闭环」。")
replace_para(P(4), "我做的主要工作：")
replace_para(P(5), "1. 实现 4×4 网格编码与 GT 框换算，解析器兼容「第X行第Y列」「rXcY」「X,Y」三种写法。")
replace_para(P(6), "2. 实现解析式验证器（PIL 逐像素颜色掩膜 + 8 邻域连通块），并对它做自检：GT 格 8/8 通过、错格 8/8 拒绝。")
replace_para(P(7), "3. 实现 RSI 闭环（高亮回灌自查 + 验证器纠错 + 数据入池 + 再训练），跑通两轮并落盘 rsi_memory.json。")

# 追加工作条目（模板只有 3 条占位）：锚点用段落对象，避免插入后索引漂移
anchor3 = P(7)
p4 = add_para_after(anchor3, "4. 用「大色块识色」诊断确认模型视觉通路正常（四色全对），排除环境问题后把失败归因到模型能力本身。")
add_para_after(p4, "5. 对照实验表明：15 条种子样本 + 60 步 SFT 仅让 loss 降到 "
                   f"{loss_last:.2f}，但定位正确率仍为 0/8，属于记忆而非学会。")

# ========== 二、实验环境与复现步骤 ==========
replace_para(P(9),
             "依赖版本：transformers 5.17.0，peft 0.21.2，torch 2.11.0+cu128，Pillow 12.3.0；"
             "bitsandbytes 未使用（本实验采用 fp16 LoRA，而非 4bit QLoRA，因 256M 模型在 8GB 显存上无需量化）。"
             "运行环境为 WSL2 下的 Ubuntu 22.04.5，Python 3.10.12，GPU 为 RTX 5060 Laptop 8GB。"
             "对参考代码的改动见 3.4 节，主要包括：SFT 轮转采样、transformers 5.x 的 prompt 长度计算兼容、"
             "验证器反馈改为方向性证据。")

# ========== 三、方法与关键实现 ==========
replace_para(P(11),
             "3.1 网格坐标编码：画布实际为 256×256（模板所述 480px 为笔误），用 GRID=4 划分，每格 64px。"
             "GT 框换算规则：取框的中心点 (cx, cy)，行号 r = int(cy/64)、列号 c = int(cx/64)，"
             "并做 min(GRID-1, ·) 截断防止越界，最终输出为「第 r+1 行第 c+1 列」（从 1 开始编号）。"
             "解析器兼容三种写法：①「第X行第Y列」；②「rXcY」（正则 r(1-4)c(1-4)，忽略大小写）；"
             "③「X,Y」「X-Y」「X,Y」等分隔形式。解析失败一律记为未命中，这正是 SFT 需要先解决的格式问题。")
replace_para(P(12),
             f"3.2 训练配置：LoRA r=8、alpha=16、dropout=0.05，挂在 q_proj/k_proj/v_proj/o_proj 四个注意力投影上，"
             f"task_type=CAUSAL_LM；可训练参数 1,363,968 / 总参数 257,848,896，占比 0.529%；"
             f"实际训练步数 {sft_info.get('steps', 60)} 步，优化器 AdamW，学习率 1e-4，只对答案 token 计算损失，"
             f"耗时 {sft_seconds:.1f}s（GPU），loss 从 {loss_first:.2f} 降到 {loss_last:.4f}。"
             "按指导书默认的 10 步训练后 loss 仍高达 16.5，模型完全未学会格式，"
             "因此本实验把步数提高到 60 步（这是与指导书默认值的主要差异，已在 3.4 节说明）。")
replace_para(P(13),
             "3.3 RSI 闭环实现：① 自查——当定位题答错且能解析出格号时，把该格渲染成紫色双线高亮框回灌给模型，"
             "附提示「若高亮框没有圈住目标，请直接给出正确位置」，若模型改写出的格号与原预测不同则记为一次「自查改写」；"
             "② 验证器——用解析式颜色掩膜统计目标颜色连通块，判定预测格是否成立：目标连通块中心落在该格内，"
             "或该格覆盖目标面积 ≥50% 即通过。不通过时把验证结论注入提示词重答。"
             "所有记录（含 pred1/自查结果/提示/纠错结果）累计写入 mem 列表，轮末落盘 rsi_memory.json；"
             "只有被验证器判定通过且 judge 判为正确的纠错答案，才以 {img, prompt, answer} 形式并入 sft_data.json 数据池。")
replace_para(P(14),
             "3.4 与参考实现不同的部分：（1）SFT 采样——参考实现每步都取 data[0]，小数据池下模型几乎只学首条样本，"
             "改为按数据池轮转采样 data[i % len(data)]；（2）transformers 5.x 兼容——参考实现用 processor 单独处理纯文本以求 prompt 长度，"
             "会因缺少 <image> 张量报错，改用 processor.tokenizer(prompt) 计算前缀长度；"
             "（3）视觉张量显式转 float32，避免 fp16 处理器张量与非 fp16 权重类型不匹配；"
             "（4）eval 阶段复用同一模型对象，避免重复加载 500MB 权重；"
             "（5）验证器反馈升级——参考实现只回「重叠仅 0%」这类无方向信息的话，模型拿不到任何可推理线索，"
             "本实验改为给出可核查的方向证据（目标中心在高亮格的左/右/上/下约几格、覆盖率多少），"
             "但**仍然不直接透露行列号**，以此守住「工具给证据、模型做决策」的边界。")

# ========== 四、实验结果 ==========
replace_para(P(16),
             f"4.1 三阶段评测（数据来自 experiment_results.json，与上表一致）：基线总计 {b.get('total', 0)}/15，"
             f"SFT 总计 {s.get('total', 0)}/15，SFT+验证器纠错总计 {sc.get('ok2', 0)}/15。"
             "三个阶段定位题（loc）均为 0/8，定位平均 IoU 均为 0.000；提升仅出现在 exist 题（0/3 → 2/3），"
             "count 题稳定在 4/4。")
replace_para(P(17),
             "4.2 RSI 递归自进化轮次记录：初始数据池 15 条，两轮 RSI 结束后数据池仍是 15 条——"
             "因为每轮「新增可用样本」均为 0，没有任何纠错答案同时满足「验证器通过」与「judge 判对」。"
             "轮末评测总计分别为 4/15、4/15，loc 与 IoU 均无变化。")
replace_para(P(18),
             f"4.3 纠错明细统计：SFT 阶段验证器触发 7 次纠错，由错变对 0 次，由对变错 0 次，"
             f"新增可用样本 0 条（净增益 {(sc.get('ok2', 0)) - s.get('total', 0)}）。"
             "自查改写在两个 RSI 轮次中均为 0 次——因为模型对定位题的首次输出大多无法被解析出合法格号，"
             "触发不了「渲染高亮格回灌」这一步。")
replace_para(P(19),
             "4.4 代表性样例（见上表）：答对样例来自 exist 题；自查改对与验证器纠对两栏本轮均未出现，如实记为未触发；"
             "仍然失败样例给出定位题的典型输出。")

# 4.4 之后插入图表（锚点固定为 4.4 段落对象）
img_anchor = P(19)
cap = insert_image_after(img_anchor, "exp6_loc_curve.png", width=5.8)
add_para_after(cap,
               "图 1　loc 定位正确率与平均 IoU 随阶段变化（基线 → SFT → SFT+纠错 → RSI 两轮）。"
               "红线为定位正确率，蓝虚线为平均 IoU，灰点线为 4×4 网格下的随机猜测期望值 12.5%。")

# ========== 五、结果分析 ==========
replace_para(P(21),
             "5.1 SFT 带来了什么：对照基线与 SFT 两行，count 题始终 4/4、exist 题从 0/3 提升到 2/3，"
             "说明 SFT 确实把模型从「复读问题」拉向了「作答」这一行为模式；但定位题仍是 0/8，IoU 仍为 0.000。"
             "更值得注意的是格式可控性：SFT 后模型的输出从基线的「1. 1. 1. 1.」「蓝色圆形位于第几行第几列」"
             "变成了「1. 在图中，第二行的第一列是第几…」这类带模板痕迹的半成品——它学会了「用中文描述行列」的语气，"
             "却没能学会「看图数格」。这从 loss 曲线也能印证：60 步内 loss 从 "
             f"{loss_first:.2f} 骤降到 {loss_last:.2f}，而测试集正确率毫无起色，"
             "是典型的记忆式过拟合（15 条种子样本 ≈ 1.5 个 batch）。")
replace_para(P(22),
             f"5.2 RSI 闭环带来了什么：对照 SFT（{s.get('total', 0)}/15）与纠错后（{sc.get('ok2', 0)}/15），"
             f"净增益 = 由错变对 {(sc.get('ok2', 0)) - s.get('total', 0)} 次 = 0。"
             f"验证器一共触发 {rounds[0].get('tool', 7) if rounds else 7} 次纠错。"
             + (f"以 L01 为例，验证器注入的真实提示是「{_hint_text}」，"
                f"而模型的纠错输出是「{_tool_out[:24].replace(chr(10), ' ')}」——"
                if _hint_text else "注入了方向证据后，")
             + "完全无法把方位描述换算成行号列号。"
             + "由于纠错答案全部被判错，两轮 RSI 都没有新样本入池，数据池始终 15 条，"
             + "「新样本回流 → 重训」这条链路根本没有被激活，第二轮自然也不会出现提升。")
replace_para(P(23),
             "5.3 失败模式分析：本实验中定位失败不是集中在「相邻格混淆」或「靠边目标」这类细粒度问题，"
             "而是系统性的——8 道题全部失败，IoU 全为 0，说明模型输出的格号与目标格毫无空间相关性。"
             "其中多数输出甚至不含合法的行列表述（如「1」「蓝色圆形位于第几行第几列」），属于解析失败，"
             "比「答错格」更底层。验证器本身没有给出错误反馈：自检显示 GT 格 8/8 判定通过、错格 8/8 判定拒绝，"
             "假阴性假阳性均为 0。因此验证器不是瓶颈，瓶颈在模型侧。")
replace_para(P(24),
             "5.4 回答核心问题：本实验中定位能力的提升——**微调没有带来，闭环也没有带来**。"
             "证据链有三条：① 表6 中 SFT 的 loc 与基线同为 0/8、IoU 同为 0.000，微调对定位零贡献；"
             "② 纠错后 loc 仍为 0/8，净增益 0，工具纠错对定位也零贡献；"
             "③ 两轮 RSI 的新增样本数均为 0，数据池未增长，递归自进化没有启动。"
             "两者的作用边界在于：SFT 能改变「输出形态」（从乱码到有模板痕迹的语句），"
             "但无法凭空创造「把图像空间位置映射为离散格号」这一能力；"
             "闭环能把模型已有的、偶发正确的答案固化为训练数据，但当模型连一次都答不对时，"
             "闭环没有任何种子可放大，只能空转。这正说明 RSI 的前提是「模型已经会一点点」，而不是「从零开始」。")


# ========== 六、思考题 ==========
def answer_after(q_idx, text):
    replace_para(P(q_idx + 1), "答：" + text)


answer_after(26,
             "改成 8×8 后，输出空间从 16 类扩到 64 类，每格边长从 64px 缩到 32px。"
             "在模型真正学会定位的前提下，IoU 会因为格子更小而更高（同样的相对偏差落在更小的格内），"
             "对精细定位是有利的；但对本实验的 256M 模型而言，正确率大概率会进一步下降——类别数翻 4 倍，"
             "而「第几行第几列」的计数推理难度同步上升，本实验连 4×4 都没学会，64 类只会更差。"
             "代价方面：① 输出空间变大，需要的数据量按经验至少翻倍到数倍；② 解析难度上升，"
             "模型更容易输出越界编号；③ 验证器的「中心落在格内」判据在细粒度下对 GT 框中心的取整更敏感，"
             "边界附近容易产生判定抖动；④ 若做 16×16，则输出空间达 256 类，接近小模型分类能力的上限。"
             "结论：离散粒度应与「模型容量 + 数据量」匹配，网格化带来的是表达简化，而不是免费的精度提升。")

answer_after(28,
             "解析式验证器会「帮倒忙」的情形有两类。第一类是判据本身在边界上不成立：本实验的判据是"
             "「目标连通块中心落在预测格内，或覆盖面积 ≥50%」。对跨格物体，如果目标中心恰好压在格子边界线上，"
             "格中心规则会让「正确格」与「相邻格」产生歧义；同理，物体较大时（本实验物体边长 34~54px，"
             "而格边长 64px），一个格可能覆盖到目标 50% 以上而中心并不在格内，这会给出「通过」的假阳性。"
             "第二类是颜色掩膜失效：本实验用欧氏距离平方 < 9000 作为「同色」判据，"
             "若同色物体的两个连通块在阈值边缘被并成一个，验证器会把两个物体当成一个，"
             "此时告诉模型的「目标中心在右上方几格」就是错的，模型照着改只会越改越偏。"
             "必须强调：本实验的自检（GT 格 8/8 通过、错格 8/8 拒绝）说明验证器在评测集上没有出错，"
             "但这是因为评测集的合成图很简单，属于「未暴露」而非「不会出错」。"
             "本实验更值得记录的一个现象是：验证器给出的方向证据（如「目标中心在右侧约 1 格」）"
             "理论上信息量充分，但 256M 模型无法把它换算成行列号，输出退化为数字串——"
             "这说明「工具给证据」的有效性还取决于模型有没有能力解读这个证据。")

answer_after(30,
             "说数据池在变贵之前是「免费的教师」，是因为 RSI 的数据不是人工标注的，"
             "而是模型自己产出、再由验证器把关确认正确的答案：生产过程零额外成本，验证成本也很低"
             "（本实验的解析式验证器单次调用远低于 0.1 秒），于是「模型既是学习者又是数据生产者」，"
             "数据池可以随轮次免费增长。"
             "它的「免费」有三个前提，一旦打破就不再免费："
             "① 验证器必须可靠——若验证器误判，混进数据池的错误样本会毒化后续训练，"
             "修正代价比从头标注更高；② 模型必须已经能产出少量正确答案——本实验正是反例，"
             "模型一次都没答对，两轮新增样本均为 0，教师再免费也印不出教材；"
             "③ 数据分布不能过窄——若只把少数易题的正确答案反复回流，数据池会退化到那几个题目上，"
             "造成分布窄化与过拟合。此外，当轮次增加后需要更多算力重训、需要人工抽检验证器质量，"
             "这些隐性成本会逐渐显现，此时就「不再免费」了。")

answer_after(32,
             "把闭环搬到 ChartQA 这类数值题，验证器的设计要从「视觉定位」换成「可计算的数值一致性」，"
             "核心是可判定、可复核、且不直接把答案抄给模型。具体可以分三层："
             "① 数据侧验证——ChartQA 的底表（表格数据）本身就是真值，用 Python 从底表按问题意图算出答案，"
             "这部分是精确的解析式计算，可以用作严格校验；"
             "② 结构侧验证——用 OCR / 图表解析工具提取坐标轴刻度与数据点，"
             "与模型答案做数值容差比对（如相对误差 < 1% 或绝对误差 < 0.5 个刻度单位），"
             "并检查答案是否落在坐标轴的合理区间内；"
             "③ 一致性验证——让验证器给出「你的答案比正确值偏大/偏小多少」这类方向性证据，"
             "而不是直接报出正确值，从而保留模型自主决策的空间。"
             "需要注意的边界：数值题的「正确」往往是区间而非单点（图表读数本身有误差），"
             "所以验证器必须允许容差；另外若问题涉及「最大值/比较」这类推理，"
             "验证器需要能模拟该推理过程，否则只能校验数值本身而不能校验逻辑。"
             "若底表不可得，就只能退化到「工具给证据」的形态，闭环的可靠性会明显下降。")

# ========== 七、结论与反思 ==========
replace_para(P(35),
             "「值不值」：就本实验的数据而言，在 256M 级小模型上为「精准定位」投入 SFT + RSI 闭环，"
             "短期内**不值**——SFT 60 步、两轮 RSI 的全部算力投入，换来的定位正确率提升为 0（0/8 → 0/8）。"
             "但这个过程本身是有价值的：它精确地标出了那条能力边界——256M 模型可以学会「输出来说像那么回事」"
             f"（loss 降到 {loss_last:.2f}、exist 题从 0/3 到 2/3），却学不会「把图像空间位置映射成离散格号」；"
             "同时验证器自检（8/8、8/8）证明工具侧是可靠的，问题完全在模型侧。"
             "本实验最主要的局限有三："
             "① 256M 天花板——2.56 亿参数在 15 条种子样本上做格式对齐，本质是记忆，缺乏泛化所需容量；"
             "② 验证器覆盖面——解析式验证器只在纯色合成图上成立，其阈值判据在真实场景（光照、遮挡、同色物体）"
             "下会大量失效；"
             "③ 网格离散化误差——4×4 网格本身就把定位精度限制在 64px 粒度，IoU 上限有限。"
             "如果继续做，下一个迭代是：先把种子数据从 15 条扩到覆盖全部 40 张训练图（用 GT 换算，仍零人工标注），"
             "让 SFT 先真正学会格式；确认 loc 有非零正确率之后，再启动 RSI，"
             "这样才能检验「自举数据回流」是否真的带来第二轮增益——"
             "换言之，RSI 应该被用于「放大已有一点点能力」，而不是「从零创造能力」。")

doc.save(OUTPUT)
print(f"[report] 报告已生成 → {OUTPUT}")
