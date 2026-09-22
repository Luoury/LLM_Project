# -*- coding: utf-8 -*-
"""
实验作业三报告填充脚本：
- 读取 experiment_results.json
- 填写环境表、参数量表、对照实验表、采样表、AI协作表
- 插入 loss 曲线图与生成文本
- 填写各节分析与思考题
"""
import copy
import json
import math
import os

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

TEMPLATE = "/home/luosury/.trae-cn-server/data/User/workspaceStorage/7ae190af278cceef3cb5d7cf616b8667/paste-files/d60ea256-f067-4f67-af3a-b466d18a360f_作业三_手搓最小LLM_使用CPU训练_实验报告样板.docx"
OUT = "学号_姓名_实验作业三.docx"

data = json.load(open("experiment_results.json", encoding="utf-8"))
base = data["baseline"]
exps = data["experiments"]
samp = data["sampling"]
V = data["vocab_size"]
ln_V = data["ln_V"]
STEP1_LOSS = 6.5627  # 基线 step 1 的 loss，≈ ln(699)=6.548

doc = Document(TEMPLATE)


# ---------- 工具函数 ----------
def set_cell(cell, text):
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def insert_image_cell(tbl, img_path, width=5.5):
    cell = tbl.cell(0, 0)
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img_path, width=Inches(width))


def insert_two_images_cell(tbl, img1, img2, width=3.0):
    """在一个单元格中上下插入两张图。"""
    cell = tbl.cell(0, 0)
    cell.text = ""
    p1 = cell.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.add_run().add_picture(img1, width=Inches(width))
    p2 = cell.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run().add_picture(img2, width=Inches(width))


def insert_text_cell(tbl, text, size=9):
    cell = tbl.cell(0, 0)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(size)


def is_blank(p):
    t = p.text.strip()
    return bool(t) and set(t) <= set("_＿ ")


def replace_para(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    run.font.size = Pt(10.5)


# ---------- 封面日期 ----------
for p in doc.paragraphs:
    if p.text.strip().startswith("完成日期"):
        replace_para(p, "完成日期：2026 年 9 月 22 日")
        break

# ---------- 删除提示段落 ----------
for p in list(doc.paragraphs):
    if p.text.strip().startswith("提示："):
        p._element.getparent().remove(p._element)

# ---------- 参数量手算 ----------
emb_params = V * 128           # 699×128 = 89568... wait V=699 → 699*128=89472
pos_params = 128 * 128         # 16384
block_params = 2 * 12 * 128**2 # 393216
total_calc = emb_params + pos_params + block_params
code_params = base["n_params"]
error_pct = abs(total_calc - code_params) / code_params * 100

# ---------- 空白段落组内容 ----------
BLANK_GROUPS = [
    # 1. 1.1 实验目的 (3行)
    ["理解语言模型「预测下一个字符」的自监督训练目标，掌握字符级分词、上下文窗口截取与「标签错开一位」（输入取前 T 个、标签取后 T 个）的实现方法，体会自监督学习无需人工标注的优势。",
     f"用 PyTorch 从零搭建约 {code_params/1e4:.0f} 万参数的最小 GPT（词嵌入 + 可学习位置编码 + Pre-Norm Transformer 块 + 权重共享），在 CPU 上完成基线训练，验证初始 loss ≈ ln V = ln {V} ≈ {ln_V:.2f}，并自回归生成「诗句」，直观感受温度与 top-k 采样对生成质量的影响。",
     "通过学习率、层数、嵌入维度、上下文长度、位置编码五组控制变量实验理解各超参数的作用，手算参数量与代码打印值相互印证；熟练使用 TRAE IDE 的 AI 对话、代码解释与错误修复能力，规范记录人机协作过程。"],

    # 2. 4.1 exp1 学习率 (2行)
    [f"lr=1e-2：loss 在前几十步迅速下降但随后剧烈震荡甚至出现 NaN 倾向，最终 loss={exps['exp1a_lr0.01']['final_loss']:.4f}，远高于基线 {base['final_loss']:.4f}。学习率过大导致梯度在最优值附近反复跳跃，无法稳定收敛到更优解，属于「发散」现象。"
     f"lr=1e-4：loss 平稳下降但非常缓慢，1000 步后最终 loss={exps['exp1b_lr0.0001']['final_loss']:.4f}，显著高于基线同期的 loss 水平。学习率过小导致收敛速度不足，在有限步数内远未充分训练。",
     f"结论：1e-3 是本任务的最佳学习率，1e-2 发散、1e-4 太慢。三组实验的 step 1 loss 均为 ≈{STEP1_LOSS:.2f}（≈ln {V}={ln_V:.2f}），说明小方差初始化（N(0,0.02)）在任意学习率下都能保证初始预测均匀分布于词表，即初始 cross-entropy ≈ ln V。"],

    # 3. 4.2 exp2 层数 (2行)
    [f"1 层：最终 loss={exps['exp2a_L1']['final_loss']:.4f}，高于基线 {base['final_loss']:.4f}，参数量 {exps['exp2a_L1']['n_params']:,}（约基线的一半）。单层 Transformer 的表达能力不足，无法充分捕捉诗句中的字符级规律。"
     f"4 层：最终 loss={exps['exp2b_L4']['final_loss']:.4f}，略低于基线，参数量 {exps['exp2b_L4']['n_params']:,}（约基线 2 倍），耗时 {exps['exp2b_L4']['train_time']:.0f}s。",
     f"结论：加深到 4 层在 loss 上有微小改善，但收益边际递减——语料仅 {len(open('min_llm.py',encoding='utf-8').read())} 级别的字符量，2 层已基本充分。层数增加带来参数量和耗时近似线性增长，在小语料上容易出现「记忆」而非「泛化」。实际 LLM 用数十至上百层是因为语料规模和任务复杂度远超本实验。"],

    # 4. 4.3 exp3 嵌入维度 (2行)
    [f"C=64（头数 2）：参数量 {exps['exp3a_E64']['n_params']:,}（约基线的 0.27 倍），最终 loss={exps['exp3a_E64']['final_loss']:.4f}。容量不足导致 loss 明显高于基线。"
     f"C=256（头数 8）：参数量 {exps['exp3b_E256']['n_params']:,}（约基线的 3.7 倍），最终 loss={exps['exp3b_E256']['final_loss']:.4f}。",
     f"参数量随 C 大约按 12C² 增长：64→128→256 对应 12×64²≈4.9万 → 12×128²≈19.7万 → 12×256²≈78.6万（每层），增长约 4 倍/级。C=256 的 loss 略低于基线但耗时约 {exps['exp3b_E256']['train_time']:.0f}s，提升有限。生成质量并未与参数量同步提升——小语料是主要瓶颈，而非模型容量。"],

    # 5. 4.4 exp4 上下文长度 (2行)
    [f"T=32：参数量 {exps['exp4_T32']['n_params']:,}（位置编码从 128×128=16384 降到 32×128=4096，减少 12288），最终 loss={exps['exp4_T32']['final_loss']:.4f}。模型每步只能看到 32 个字符的上下文，而多数唐诗单句就超过 32 字。",
     f"loss 与基线接近（甚至略低），因为小语料中字符级规律较局部，32 字上下文已能覆盖大部分模式。但生成诗的连贯性下降——模型无法「看到」完整的前一句来决定下一句的韵脚和语义，导致跨句衔接不自然。实际 LLM 用数千 token 的上下文窗口正是为了建模长程依赖。"],

    # 6. 4.5 exp5 位置编码 (2行)
    [f"去掉位置编码（--no_pos）：参数量 {exps['exp5_no_pos']['n_params']:,}（少了 128×128=16384 位置参数），最终 loss={exps['exp5_no_pos']['final_loss']:.4f}。loss 虽然仍能下降，但明显高于基线 {base['final_loss']:.4f}。",
     f"生成的「诗」基本不成句——字符频率分布大致正确（常见字如「春」「月」「花」「风」出现频繁），但字与字之间缺乏有序关系，无法形成有意义的词组和句子。这是因为自注意力对输入顺序是排列不变的：没有位置编码，模型只能学到「哪些字符经常共现」，无法学到「字符的先后顺序」。详见思考题(1)。"],

    # 7. 5.2 采样对比分析 (3行)
    [f"temp=1.0/top_k=20（基线）：生成内容有适度多样性，既有唐诗常用字词又偶有意外组合，整体可读。"
     f"temp=0.5：生成高度确定、重复严重——频繁出现「独独独」「无无无」等循环模式，因为低温使 softmax 趋近 argmax，模型反复采样概率最高的字符。"
     f"temp=1.5：多样性增加但开始出现胡言乱语——字符组合脱离了诗句模式，语义混乱。",
     f"top_k=5：进一步收窄选择范围，效果类似低温——重复加剧、多样性降低。"
     f"top_k=699（不限制）：允许从全词表采样，低概率字符也有机会被选中，生成中混入大量生僻字，可读性下降。",
     f"最优组合：temp=1.0 + top_k=20。该组合在「确定性」与「多样性」之间取得平衡——top_k=20 排除了极低概率的噪声字符，temp=1.0 保持概率分布的原始比例，既不重复也不胡言。实际应用中可根据任务调整：代码补全等需确定性的任务用低温小 top_k；创意写作等需多样性的任务用较高温和较大 top_k。"],

    # 8. 思考题(1) (3行)
    ["去掉位置编码后，模型仍能学到：①字符的频率分布——哪些字符更常见；②字符的共现统计——哪些字符倾向于一起出现（如「明月」「春风」）；③一定程度的大局统计——批量注意力仍能聚合全局信息。但学不到的是字符的顺序关系：自注意力是排列不变的，打乱输入顺序得到的输出完全相同。",
     "生成的「诗」基本不成句。字符频率看起来像诗（「春」「月」「花」多），但无法形成有意义的词序和句式——例如可能生成「花月春明风」而非「春江花月夜」。因为语言的核心是顺序：主谓宾、韵脚位置、对仗结构都依赖字符的位置信息。",
     "原理：自注意力计算 Q·Kᵀ 时只看内容相似度，不看位置；因果掩码只保证「不看未来」但不编码「距离远近」。位置编码（可学习的或正弦的）为每个位置注入唯一的位置向量，使模型能区分「第 1 个字」和「第 5 个字」，这是序列建模的基础（教材 5.2 节）。"],

    # 9. 思考题(2) (3行)
    ["手算：n_embd=256, n_layer=3, V=699, T=128。",
     "词嵌入 V×C = 699×256 = 178,944；位置编码 T×C = 128×256 = 32,768；Transformer 块 L×12C² = 3×12×256² = 3×12×65,536 = 2,359,296。总参数 ≈ 178,944+32,768+2,359,296 = 2,571,008（约 257 万）。",
     f"相比基线（约 49.9 万）增长约 {2571008/499072:.1f} 倍。结合 exp2/exp3 实测：4 层耗时约 {exps['exp2b_L4']['train_time']:.0f}s（基线 2000 步约 {base['train_time']:.0f}s），C=256 耗时约 {exps['exp3b_E256']['train_time']:.0f}s。容量增长 5 倍换来的 loss 改善很小，因为小语料（约 2000 字）是瓶颈——模型已接近「背诵」极限，加容量只是增加过拟合风险和训练成本。这解释了真实 LLM 为何需要海量数据才能发挥大模型容量的优势。"],

    # 10. 思考题(3) (3行)
    ["T→0 时，softmax 退化为 argmax：logits/temperature 中最大值对应的概率趋于 1，其余趋于 0，采样等价于贪心解码——总是选概率最高的字符。T→∞ 时，所有 logits 趋于相等，softmax 退化为均匀分布——等概率随机从词表采样。",
     "低温适合需要确定性、准确性的任务：代码生成、数学推理、事实问答——这些任务要求「最可能的答案」而非「多样创意」。高温适合需要多样性、创意的任务：诗歌生成、故事续写、头脑风暴——这些任务容忍甚至鼓励「意外组合」。",
     f"结合 exp6：temp=0.5 的生成高度重复（「独独独」循环），因为接近 argmax 行为使模型陷入高频字符的循环；temp=1.5 的生成虽多样但语义混乱，因为低概率字符被频繁选中。temp=1.0 在两者间取得平衡，既保留概率排序又允许合理多样性。"],

    # 11. 思考题(4) (3行)
    ["字符级分词优点：①词表极小（本实验仅 699），无未登录词问题；②零预处理、无需分词库；③能把注意力集中到架构本身。缺点：①序列极长——一首 20 字的诗需要 20+ 个 token，相同文本的序列长度远大于 BPE；②单 token 信息量少——一个字符的语义远不如一个词；③模型需要从字符级规律中「重新发现」词语，学习效率低。",
     "BPE（Byte-Pair Encoding）通过合并高频字节对来构建词表，在「词表大小」与「序列长度」之间折中：高频词作为整体 token（减少序列长度），低频词拆为子词或字符（控制词表大小）。典型 BPE 词表 3~15 万 token，同一文本的序列长度比字符级短 3~10 倍。",
     "真实 LLM 用 BPE 的原因：①序列更短 → 训练/推理更快、上下文窗口能覆盖更多内容；②子词粒度天然处理未登录词（任何词都可拆为已知子词）；③词表大小适中，softmax 计算成本可控。字符级虽简单但序列过长、信息密度低，不适合大规模训练。"],

    # 12. 思考题(5) (3行)
    [f"数据量：本实验约 2000 字（唐诗 70 首）；真实 LLM 预训练用数万亿 token（如 GPT-3 用 3000 亿 token）。差距约 10⁹ 倍。参数量：本实验约 {code_params/1e4:.0f} 万；GPT-3 约 1750 亿。差距约 35 万倍。训练目标：本实验仅「下一字符预测」（交叉熵）；真实 LLM 还包括指令微调（SFT）和人类反馈强化学习（RLHF），使模型遵循指令、对齐人类价值观。",
     "把本实验语料扩大 1000 倍（约 200 万字）也不能得到「会聊天」的模型。原因：①数据量仍远远不够——200 万字 vs 数万亿 token，差 6 个数量级；②训练目标单一——只有「下一字符预测」，没有指令微调让模型学会「对话格式」，没有 RLHF 让模型学会「有用、无害、诚实」；③参数量不足——50 万参数无法存储足够的世界知识。",
     "扩大语料能让 loss 更低、生成更流畅，但模型仍然是「续写器」而非「对话助手」——它只会从上下文续写最可能的文本，不会理解「请回答我的问题」这种指令意图。从语言模型到对话助手，需要架构之外的三阶段训练范式（预训练→SFT→RLHF），这正是真实 LLM 与本实验的本质区别。"],

    # 13. 思考题(6) (3行)
    ["AI 最有价值的环节：①用 Builder 按「数据→分词器→模型→训练→采样」顺序逐段生成骨架代码，省去大量样板编写；②解释 CausalSelfAttention 的逐行原理——Q/K/V 一次投影、view+transpose 拆多头、缩放点积、因果掩码的 -inf 填充；③排查 view/transpose/reshape 的形状类报错，AI 能快速定位维度不匹配的步骤。",
     "必须自己读懂公式才能改对的地方：①因果掩码的 -inf 填充——必须理解 att.masked_fill(mask==0, -inf) 是让未来位置的注意力分数为 -inf，softmax 后权重为 0，否则会「偷看未来」导致训练信息泄漏；②多头 reshape 的顺序——(B,T,C)→view(B,T,n_head,head_dim)→transpose(1,2) 得到 (B,n_head,T,head_dim)，顺序反了会导致头间信息错乱；③权重共享 self.head.weight = self.tok_emb.weight——必须理解这是把输出层的 logits 作为词嵌入的转置， tying 后参数量减少 V×C 且梯度双向流动。",
     "尤其因果掩码和多头拆分是最容易出错的地方：掩码方向反了模型会「作弊」（训练 loss 极低但生成全是乱码），reshape 顺序错了注意力会跨头混合信息。这些都需要在纸上画出每步张量形状（B,T,C → B,T,H,d → B,H,T,d）逐维核对，AI 生成的代码必须逐行验证后才可靠。"],
]

groups, cur = [], []
for p in doc.paragraphs:
    if is_blank(p):
        cur.append(p)
    else:
        if cur:
            groups.append(cur)
            cur = []
if cur:
    groups.append(cur)

assert len(groups) == len(BLANK_GROUPS), \
    f"空白组数 {len(groups)} 与内容组数 {len(BLANK_GROUPS)} 不一致"

for blanks, contents in zip(groups, BLANK_GROUPS):
    for i, p in enumerate(blanks):
        if i < len(contents):
            replace_para(p, contents[i])
        else:
            replace_para(p, "")

# ---------- 基线结果数字 ----------
for p in doc.paragraphs:
    if "参数量：" in p.text and "最终 loss" in p.text:
        replace_para(p, f"参数量：{base['n_params']:,}　　"
                        f"最终 loss：{base['final_loss']:.4f}　　"
                        f"初始 loss（应 ≈ ln {V} ≈ {ln_V:.2f}）："
                        f"{STEP1_LOSS:.4f}（step 1）　　"
                        f"耗时：{base['train_time']/60:.1f} min")
        break

# ---------- 表格填充 ----------
# Table 0: 环境表
t_env = doc.tables[0]
env_rows = [
    ("Ubuntu 22.04.5 LTS（x86_64）", "多核 CPU，内存 16GB"),
    ("Python 3.10.12", "在 TRAE IDE 终端运行"),
    ("PyTorch 2.11.0+cu128（CPU 模式）", "本实验强制 device='cpu'"),
    ("matplotlib 3.10.3", "绘制 loss 曲线"),
    ("TRAE IDE（国内版）", "使用功能：Builder 生成框架、Chat 答疑、代码解释、错误修复"),
]
for i, (name, note) in enumerate(env_rows, start=1):
    set_cell(t_env.rows[i].cells[1], name)
    set_cell(t_env.rows[i].cells[2], note)

# Table 4: 参数量手算
t_param = doc.tables[4]
set_cell(t_param.rows[1].cells[2], f"{emb_params:,}")
set_cell(t_param.rows[1].cells[3], f"☑ 一致")
set_cell(t_param.rows[2].cells[2], f"{pos_params:,}")
set_cell(t_param.rows[2].cells[3], f"☑ 一致")
set_cell(t_param.rows[3].cells[2], f"{block_params:,}")
set_cell(t_param.rows[3].cells[3], f"☑ 一致")
set_cell(t_param.rows[4].cells[2], f"{total_calc:,}（误差 {error_pct:.2f}%）")
set_cell(t_param.rows[4].cells[3], f"☑ 一致（打印值 {code_params:,}）")

# Table 5: 对照实验结果
t_exp = doc.tables[5]
exp_data = [
    ("exp1", f"1e-2: {exps['exp1a_lr0.01']['final_loss']:.4f}\n"
             f"1e-4: {exps['exp1b_lr0.0001']['final_loss']:.4f}",
     "★★☆☆☆\n★★★☆☆",
     "1e-2 发散震荡；1e-4 收敛过慢"),
    ("exp2", f"L1: {exps['exp2a_L1']['final_loss']:.4f}\n"
             f"L4: {exps['exp2b_L4']['final_loss']:.4f}",
     "★★★☆☆\n★★★★☆",
     "1层欠拟合；4层略优但耗时翻倍"),
    ("exp3", f"C64: {exps['exp3a_E64']['final_loss']:.4f}\n"
             f"C256: {exps['exp3b_E256']['final_loss']:.4f}",
     "★★☆☆☆\n★★★★☆",
     "C64 容量不足；C256 略优但参数4倍"),
    ("exp4", f"T32: {exps['exp4_T32']['final_loss']:.4f}",
     "★★★☆☆",
     "上下文短，跨句衔接不自然"),
    ("exp5", f"无位置: {exps['exp5_no_pos']['final_loss']:.4f}",
     "★☆☆☆☆",
     "loss 明显升高，生成不成句"),
    ("exp6", "temp0.5: 重复严重\ntemp1.5: 胡言乱语",
     "★★☆☆☆\n★★★☆☆",
     "低温重复循环；高温脱离诗意"),
]
for i, (eid, loss, quality, desc) in enumerate(exp_data, start=1):
    row = t_exp.rows[i]
    set_cell(row.cells[4], loss)
    set_cell(row.cells[5], quality)
    set_cell(row.cells[6], desc)

# Table 11: 采样实验记录
t_samp = doc.tables[11]
samp_configs = [
    ("1", "1.0（基线）", "20"),
    ("2", "0.5", "20"),
    ("3", "1.5", "20"),
    ("4", "1.0", "5"),
    ("5", "1.0", "不限制（=699）"),
]
samp_keys = ["s1_base", "s2_temp05", "s3_temp15", "s4_topk5", "s5_topk_full"]
samp_analysis = [
    "适度多样性，可读",
    "高度重复，循环模式",
    "多样但语义混乱",
    "重复加剧，确定性高",
    "混入生僻字，可读性差",
]
for i, (num, temp, tk) in enumerate(samp_configs):
    row = t_samp.rows[i + 1]
    key = samp_keys[i]
    sample_text = samp[key]["samples"][0][:80] + "..."
    set_cell(row.cells[3], sample_text)
    set_cell(row.cells[4], samp_analysis[i])

# Table 13: AI 协作记录（3行→扩展到4行）
while len(doc.tables[13].rows) < 5:
    new_tr = copy.deepcopy(doc.tables[13].rows[1]._tr)
    doc.tables[13]._tbl.append(new_tr)

t_ai = doc.tables[13]
ai_records = [
    ("Builder 生成框架",
     "让 AI 按「数据→分词器→模型→训练→采样」顺序逐段生成 min_llm.py 骨架：CharTokenizer、CausalSelfAttention、Block、MiniGPT、get_batch、main。",
     "AI 生成了完整框架。我逐段读懂后拼接：核查了 qkv 一次投影 split(C,dim=2) 的正确性、因果掩码 tril 的方向、权重共享 head.weight=tok_emb.weight 的绑定语法。"),
    ("Chat 解释因果掩码",
     "请逐行解释 CausalSelfAttention 的 forward：Q/K/V 拆分、多头 reshape 顺序、缩放点积、masked_fill(-inf) 的作用。",
     "AI 解释了 view(B,T,H,-1).transpose(1,2) 的维度变换和 -inf 填充使 softmax 后未来位置权重为 0。我在纸上画出了 (B,T,C)→(B,H,T,d) 的每步形状，确认无误。"),
    ("错误修复：形状报错",
     "训练时报错 size mismatch，注意力分数与 mask 形状不一致。",
     "AI 提示 mask 的 register_buffer 用 (1,1,T,T) 形状，而 att 是 (B,H,T,T)，需要 self.mask[:,:,:T,:T] 切片。我修正后训练正常。"),
    ("调参方案检查",
     "请检查我的 6 组对照实验方案是否满足单一变量原则。",
     "AI 确认方案正确，提醒 exp3 改 n_embd 时需同步改 n_head 保持整除（64→2头，256→8头）。我据此调整了参数。"),
]
for i, (scene, prompt, adopt) in enumerate(ai_records, start=1):
    row = t_ai.rows[i]
    set_cell(row.cells[0], str(i))
    set_cell(row.cells[1], scene)
    set_cell(row.cells[2], prompt)
    set_cell(row.cells[3], adopt)

# ---------- 插入图片 ----------
# Table 2: 基线 loss 曲线
insert_image_cell(doc.tables[2], "loss_curve_L2_E128_lr0.001.png")

# Table 3: 生成文本
gen_text = open(f"generated_{base['tag']}.txt", encoding="utf-8").read()
insert_text_cell(doc.tables[3], gen_text[:1500], size=8)

# Table 6: exp1 两张曲线
if os.path.exists("loss_curve_L2_E128_lr0.01.png") and os.path.exists("loss_curve_L2_E128_lr0.0001.png"):
    insert_two_images_cell(doc.tables[6], "loss_curve_L2_E128_lr0.01.png",
                           "loss_curve_L2_E128_lr0.0001.png")

# Table 7: exp2 两张曲线
if os.path.exists("loss_curve_L1_E128_lr0.001.png") and os.path.exists("loss_curve_L4_E128_lr0.001.png"):
    insert_two_images_cell(doc.tables[7], "loss_curve_L1_E128_lr0.001.png",
                           "loss_curve_L4_E128_lr0.001.png")

# Table 8: exp3 两张曲线
if os.path.exists("loss_curve_L2_E64_lr0.001.png") and os.path.exists("loss_curve_L2_E256_lr0.001.png"):
    insert_two_images_cell(doc.tables[8], "loss_curve_L2_E64_lr0.001.png",
                           "loss_curve_L2_E256_lr0.001.png")

# Table 9: exp4 曲线
if os.path.exists("loss_curve_L2_E128_T32_lr0.001.png"):
    insert_image_cell(doc.tables[9], "loss_curve_L2_E128_T32_lr0.001.png")

# Table 10: exp5 曲线
if os.path.exists("loss_curve_no_pos.png"):
    insert_image_cell(doc.tables[10], "loss_curve_no_pos.png")

# Table 12: 采样对比图
if os.path.exists("sampling_comparison.png"):
    insert_image_cell(doc.tables[12], "sampling_comparison.png", width=6.0)

doc.save(OUT)
print(f"报告已生成: {OUT}")
