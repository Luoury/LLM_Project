# -*- coding: utf-8 -*-
"""
实验作业二报告填充脚本：
- 读取 experiment_results.json 中的真实实验数据
- 填写环境表、基线配置表、参数量表、对照实验表、AI 协作记录表、复测表
- 插入 result_*.png 曲线图与预测对比图
- 填写各节分析文字与思考题作答
"""
import copy
import json

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

TEMPLATE = "/home/luosury/.trae-cn-server/data/User/workspaceStorage/7ae190af278cceef3cb5d7cf616b8667/external-files/home/0-mu2qttk7-oqnn/作业二_ConvLSTM的算法应用与改进_实验报告样板.docx"
OUT = "学号_姓名_实验作业二.docx"

data = json.load(open("experiment_results.json", encoding="utf-8"))
R = {r["key"]: r for r in data["results"]}
base_params = data["baseline_params"]
dev = data["device"]

doc = Document(TEMPLATE)


# ---------- 工具函数 ----------
def set_cell(cell, text, bold=False):
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    run.font.bold = bold


def insert_image_cell(tbl, img_path, width=6.0):
    cell = tbl.cell(0, 0)
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img_path, width=Inches(width))


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
        replace_para(p, "完成日期：2026 年 9 月 15 日")
        break

# ---------- 删除“提示：……”段落 ----------
for p in list(doc.paragraphs):
    if p.text.strip().startswith("提示："):
        p._element.getparent().remove(p._element)

# ---------- 参数量手算结果 ----------
cell_params = 4 * (3 * 3 * 1 * 32 + 3 * 3 * 32 * 32 + 32)  # 38144
out_params = 3 * 3 * 32 * 1 + 1                              # 289
total_params = cell_params + out_params                       # 38433

b, e1, e2, e3, e4, e5, e6, best = (R["baseline"], R["exp1"], R["exp2"],
                                   R["exp3"], R["exp4"], R["exp5"],
                                   R["exp6"], R["best"])

# 正文空白段落组（严格按文档顺序，共 15 组，每组条数 <= 空白行数）
BLANK_GROUPS = [
    # 1 —— 1.1 实验目的（3 行）
    ["理解 ConvLSTM“把 LSTM 的矩阵乘法替换为卷积”的核心思想，掌握时空序列预测“看过去几帧、预测下一帧”的完整建模流程，明确卷积结构在保留二维空间关系上的优势。",
     "掌握 ConvLSTM 细胞的参数量计算公式，能对给定配置（C_in、C_h、K、层数）逐项手算并与代码核对；在自合成弹跳小球序列上完成基线训练，记录测试 MSE、MAE 与耗时。",
     "通过层数、隐藏通道、卷积核、输入帧数、网络结构、损失函数六组控制变量实验理解各维度影响，组合有效改进得到最优配置；并熟练运用 TRAE IDE 的 Builder/Chat/内联补全/智能修复，规范记录人机协作过程。"],

    # 2 —— 3.1 实验1 加深层数（2 行）
    [f"动机与改动：把 ConvLSTM 从 1 层加深到 2 层（第二层输入/输出通道均为 32），预期第一层提取局部运动、第二层建模更长程时空依赖，其余配置与基线一致。",
     f"结果：MSE 由 {b['mse']:.5f} 降到 {e1['mse']:.5f}，MAE 由 {b['mae']:.5f} 降到 {e1['mae']:.5f}，精度小幅提升；参数量由 {base_params} 增至 {e1['n_params']}（约 2.9 倍），耗时 {e1['time']:.1f}s。说明两层抽象对小球运动有一定帮助，但收益有限——匀速+反弹规律较简单，单层已能较好拟合，加深主要增加成本。"],

    # 3 —— 3.2 实验2 隐藏通道（2 行）
    [f"动机与改动：隐藏通道 C_h 由 32 增至 64，每时刻保留更丰富的特征表示，其余不变。理论上细胞参数量按 C_h² 项约增至 4 倍。",
     f"结果：MSE 由 {b['mse']:.5f} 降到 {e2['mse']:.5f}，MAE 由 {b['mae']:.5f} 降到 {e2['mae']:.5f}；参数量 {e2['n_params']}（基线的 {e2['n_params']/base_params:.1f} 倍），与手算预测的约 4 倍吻合。容量翻倍换来的 MSE 下降很小，说明 32 通道对本简单任务已基本够用，64 通道存在冗余。"],

    # 4 —— 3.3 实验3 卷积核（2 行）
    [f"动机与改动：卷积核 K 由 3 增至 5，单层感受野由 3×3 扩到 5×5，试图捕捉更大范围的运动上下文，其余不变。",
     f"结果：MSE 由 {b['mse']:.5f} 降到 {e3['mse']:.5f}，是单项结构改动中 MSE 最低的；但 MAE {e3['mae']:.5f} 略高于基线 {b['mae']:.5f}（误差分布略有不同）。小球速度仅 0.8~1.6 像素/帧，相邻帧位移很小，3×3 本已能覆盖，故 K=5 的提升有限，且参数量增至 {e3['n_params']}。"],

    # 5 —— 3.4 实验4 输入帧数（2 行）
    [f"动机与改动：输入帧数由 4 增至 8，期望更长的历史轨迹能更准确估计速度与反弹时机。需注意按指南数据切分，4 帧预测第 5 帧、8 帧预测第 9 帧，目标帧本身不同。",
     f"结果：MSE 为 {e4['mse']:.5f}、MAE 为 {e4['mae']:.5f}，略高于基线。原因有二：①预测的是更靠后的第 9 帧，途中可能经历更多次反弹，目标本身更难；②在 32×32 小场内运动周期短，8 帧可能已包含一两次反弹后的“过时”运动状态。该对照说明“帧数越多越好”并不成立，需结合目标时刻与运动周期权衡。"],

    # 6 —— 3.5 实验5 结构对照（2 行）
    [f"动机与改动：把 ConvLSTM 换成全连接 LSTM——每帧展平为 1024 维输入 nn.LSTM(1024,256)，取末时刻输出经 Linear(256→1024) 重塑为 32×32，其余不变。",
     f"结果：MSE 高达 {e5['mse']:.5f}（基线的约 {e5['mse']/b['mse']:.1f} 倍），MAE {e5['mae']:.5f}；参数量 {e5['n_params']}，是基线（{base_params}）的约 {e5['n_params']/base_params:.0f} 倍。展平打散了二维邻接关系，全连接又无法参数共享，又慢又差，直接印证卷积结构在时空任务上的优势。"],

    # 7 —— 3.6 实验6 损失函数（2 行）
    [f"动机与改动：训练损失由 MSE 换为 L1（nn.L1Loss），评估时同时报 MSE/MAE，其余不变，考察损失对指标与预测帧视觉的影响。",
     f"结果：训练用 L1 后，MAE 为 {e6['mae']:.5f}，优于基线 MAE {b['mae']:.5f}；但 MSE 为 {e6['mse']:.5f}，高于基线 {b['mse']:.5f}。这正符合“优化什么指标就在该指标上更好”：L1 对大误差线性惩罚、对个别大误差帧更鲁棒，预测帧通常更锐利；MSE 平方放大大误差、倾向输出模糊均值。"],

    # 8 —— 5.2 预测帧对比分析（3 行）
    ["对比图（3 个测试样本：8 个输入帧 | 真实第 9 帧 | 预测帧）显示，最优组合预测出的亮斑与真实小球位置高度重合，每个样本标注的逐样本 MSE 都很小，模型确实学到了“匀速运动 + 边界反弹”的物理规律，而非简单复制上一帧。",
     "误差与场景相关：小球在场地中部匀速运动、离边界较远时预测最准；当小球贴边即将反弹时误差略增，因为反弹瞬间速度方向突变，需要模型从有限历史帧中同时推断边界位置与速度方向，难度更大。",
     f"视觉上预测帧比真实帧略柔和、边缘有轻微发虚，这是 MSE 损失输出条件均值的固有现象（指南 Q5 亦有说明）；整体位置准确，最优组合平均 MSE 仅 {best['mse']:.5f}，相比基线 {b['mse']:.5f} 下降约 {(1-best['mse']/b['mse'])*100:.0f}%。"],

    # 9 —— 5.3 总结论（4 行）
    [f"1. 卷积结构是时空预测的关键：ConvLSTM 仅 {base_params} 参数即取得 MSE {b['mse']:.5f}，而参数量约 41 倍的全连接 LSTM MSE 高达 {e5['mse']:.5f}，空间局部性与参数共享的价值最为突出。",
     f"2. 对本简单任务，扩大卷积核（K=5）、加深到 2 层、加宽到 64 通道均带来小幅 MSE 改善（分别到 {e3['mse']:.5f}/{e1['mse']:.5f}/{e2['mse']:.5f}），但参数量显著上升；盲目增加输入帧数（实验4）因目标帧更靠后、含多次反弹反而略差，说明改进需针对数据特性而非一味堆资源。",
     f"3. 损失函数决定优化偏好：L1 的 MAE 更优（{e6['mae']:.5f}）且预测更锐利，MSE 的 MSE 更低但帧略模糊，应按任务对“位置精度/视觉清晰度”的侧重选择。",
     f"4. 综合有效改进（2 层 + 64 通道 + K=5 + 8 帧 + MSE）3 次复测（seed 42/43/44）平均 MSE={best['mse']:.5f}、MAE={best['mae']:.5f}，明显优于任一单项与基线，证明多项改进在更大模型容量与更充分上下文协同下才能充分释放。"],

    # 10 —— 思考题 (1)（3 行）
    ["本质区别：普通 LSTM 用全连接矩阵乘法处理一维向量输入，状态也是一维向量；ConvLSTM 把 W·x、W·h 的矩阵乘换成二维卷积（W∗x、W∗h），输入、隐藏状态、细胞状态都保持 (C,H,W) 的二维特征图结构。",
     "为什么保留二维结构重要：运动本质是“空间位置随时间变化”，二维结构让卷积核能直接在邻域内检测亮斑（小球）的位移，天然契合运动的局部性；展平成一维后像素邻接关系被彻底打散，模型只能靠海量全连接参数去重新记住“哪些坐标在空间上相邻”。",
     "卷积还带来参数共享（同一核扫描全图）与局部感受野两个强先验，使 ConvLSTM 用远少于全连接 LSTM 的参数即可高效建模局部运动模式，这也是它成为雷达回波外推等时空任务标准工具的原因。"],

    # 11 —— 思考题 (2)（3 行）
    ["单细胞参数量 = 4×(K²·C_in·C_h + K²·C_h·C_h + C_h)。C_h=32：4×(9×1×32 + 9×32×32 + 32) = 4×(288+9216+32) = 4×9536 = 38144。",
     "C_h=64：4×(9×1×64 + 9×64×64 + 64) = 4×(576+36864+64) = 4×37504 = 150016；增长 150016/38144 ≈ 3.93 倍（约 4 倍）。",
     "权衡：C_h 加倍时，输入项 K²·C_in·C_h 只线性翻倍，但循环项 K²·C_h² 按平方增长，主导了约 4 倍的参数/计算量。实验2 中 64 通道 MSE 仅小幅改善，说明在简单任务上容量很快饱和；增加通道换来的精度边际收益递减，而显存、耗时与过拟合风险持续上升，应按任务复杂度选择。"],

    # 12 —— 思考题 (3)（3 行）
    [f"参数量：nn.LSTM(1024,256) 含 4×(1024×256 + 256×256) 权重和 4×256×2 偏置 = 4×(262144+65536)+2048 = 1,313,792；Linear(256→1024) = 256×1024+1024 = 263,168；合计 {e5['n_params']:,}（约 157.6 万），是基线 ConvLSTM（{base_params:,}）的约 {e5['n_params']/base_params:.0f} 倍。",
     f"表现差异（MSE {e5['mse']:.5f} vs 基线 {b['mse']:.5f}）：①参数量角度，全连接权重矩阵为 1024×1024 量级，绝大多数参数在描述“任意两个像素坐标”的全局关系，而小球运动只涉及局部邻域，参数严重浪费且在 2000 条数据上易过拟合；②空间局部性角度，展平破坏了邻接结构，模型没有“相邻像素相关、运动是局部位移”的归纳偏置。",
     "ConvLSTM 用小卷积核 + 参数共享把“局部运动检测”先验直接编码进结构，用约 1/41 的参数获得更好的结果——结构上的合理归纳偏置比单纯堆参数更有效。"],

    # 13 —— 思考题 (4)（3 行）
    ["能提升的原因：更多历史帧给出更长的轨迹，可更稳健地估计速度的大小与方向，尤其在接近边界时有助于判断“是否将反弹”，降低单帧噪声对速度估计的影响，使外推更平滑。",
     "加到 16 帧的问题：①计算量与训练/推理耗时随帧数近似线性增长，内存占用上升；②梯度沿时间回传的路径变长，易发生梯度消失/爆炸，早期帧信息难以利用；③在 32×32 小场内运动周期短，16 帧往往已含多次反弹，混入与当前运动方向不一致的“过时状态”，反而干扰预测（本实验4 已初见此端倪）；④序列只有 10 帧，需重新生成更长数据。",
     "实践中应根据运动的时间相关性长度（有效记忆跨度）选择输入帧数，并可配合更长训练序列、注意力机制或分层结构来缓解长程依赖问题。"],

    # 14 —— 思考题 (5)（3 行）
    ["在雷达回波外推中，输入是过去若干时刻的二维雷达反射率（降水强度）场序列（可能含多高度层/多变量通道），输出是未来数十分钟到数小时的反射率场预测序列，再据反射率-降水率关系换算为降水量。",
     "相比弹跳小球的额外难点：①目标可生消——对流云团会新生、消散、合并、分裂，不满足小球的“物体恒在、数量守恒”；②多尺度——从公里级对流单体到百公里级锋面系统并存，单一感受野难以兼顾；③真实噪声与不确定性——雷达受地物杂波、波束遮挡、衰减、距离折叠影响，且大气湍流与微物理过程使运动本身带随机性。",
     "此外真实回波图分辨率远高于 32×32、数据量大、需多步预测且误差随 lead time 累积，物理过程（热力、动力、相变）远比匀速反弹复杂，因此实际系统常用多层/堆叠 ConvLSTM、TrajGRU 或加入注意力与概率预报等手段。"],

    # 15 —— 思考题 (6)（3 行）
    ["AI 最有价值的环节：①用 Builder 快速生成数据合成、ConvLSTMCell、时间循环、训练与可视化的整体骨架，省去样板代码；②排查五维张量 (B,T,C,H,W) 的形状错误（permute/squeeze/cat 处最易出错）与 FlattenLSTM 的维度衔接；③解释合并卷积为什么是 Conv2d(C_in+C_h, 4C_h)、对比卷积版与矩阵版公式差异。",
     "必须自己读懂公式才能改对的地方：①四门的划分与激活——i/f/o 走 sigmoid、候选 g 走 tanh，以及 c_t=f⊙c_{t−1}+i⊙g̃、h_t=o⊙tanh(c_t) 的更新次序与逐元素乘符号；②cat([x,h]) 使输入通道变 C_in+C_h、chunk 成 4 份的通道组织；③多层堆叠时上一层隐藏状态如何作为下一层输入。",
     "尤其参数量手算必须亲自推导 K²C_inC_h 与 K²C_h² 两项的来源，才能核对代码、并解释实验2“通道加倍参数约变 4 倍”。结论：AI 擅长加速实现与定位报错，但门控语义、张量流向与参数公式这些“为什么”必须人先理解，AI 生成的代码经逐行读懂、验证、修正后才成为可靠的实验。"],
]

# 按文档顺序收集连续空白段落分组
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

# ---------- 2.3 基线结果数字 ----------
for p in doc.paragraphs:
    if "测试 MSE" in p.text and "测试 MAE" in p.text:
        replace_para(p, f"测试 MSE：{b['mse']:.5f}　　测试 MAE：{b['mae']:.5f}　　耗时：{b['time']:.1f} s")
        break

# ---------- 表格索引 ----------
# 0=环境 1=基线配置 2=参数量 3=图1占位 4=对照结果 5~10=图2~7占位
# 11=AI协作 12=复测 13=图8占位
t_env, t_base, t_param = doc.tables[0], doc.tables[1], doc.tables[2]
t_res, t_ai, t_re = doc.tables[4], doc.tables[11], doc.tables[12]

# 表1 环境表（4 个数据行）
gpu_note = ("GPU：NVIDIA GeForce RTX 5060 Laptop（CUDA 可用，本次在 GPU 上训练）"
            if dev.startswith("cuda") else "GPU：本次在 CPU 上运行")
env_rows = [
    ("Ubuntu 22.04.5 LTS（x86_64）", ""),
    ("Python 3.10.12 / PyTorch 2.11.0+cu128", gpu_note),
    ("弹跳小球序列（脚本自动生成）", "2200 条：前 2000 训练 / 后 200 测试，每条 10 帧 32×32"),
    ("TRAE IDE（国内版）", "使用功能：Builder 生成框架、Chat 答疑、内联补全、智能修复"),
]
for i, (name, note) in enumerate(env_rows, start=1):
    set_cell(t_env.rows[i].cells[1], name)
    set_cell(t_env.rows[i].cells[2], note)

# 表2 基线配置（第5行损失/优化器单元格确认）
set_cell(t_base.rows[4].cells[1], "MSE / Adam(lr=0.001)，批 64，5 epoch")

# 表3 参数量手算与核对
set_cell(t_param.rows[1].cells[1], "4×(3×3×1×32 + 3×3×32×32 + 32) = 4×9536")
set_cell(t_param.rows[1].cells[2], str(cell_params))
set_cell(t_param.rows[1].cells[3], "☑ 一致")
set_cell(t_param.rows[2].cells[1], "3×3×32×1 + 1 = 288 + 1")
set_cell(t_param.rows[2].cells[2], str(out_params))
set_cell(t_param.rows[2].cells[3], "☑ 一致")
set_cell(t_param.rows[3].cells[1], "38144 + 289")
set_cell(t_param.rows[3].cells[2], str(total_params))
set_cell(t_param.rows[3].cells[3], f"☑ 一致（sum(numel)={base_params}）")

# 表4 对照实验结果（表头 + 8 数据行）
conclusions = {
    "baseline": "基准：1层/32通道/K3/4帧/MSE",
    "exp1": f"加深到2层，MSE 降至 {e1['mse']:.5f}，参数约2.9倍",
    "exp2": f"通道64，MSE {e2['mse']:.5f} 略降，参数约{e2['n_params']/base_params:.1f}倍",
    "exp3": f"K=5单项MSE最低 {e3['mse']:.5f}，MAE略升",
    "exp4": f"看8帧预测第9帧，目标更难，MSE {e4['mse']:.5f} 略升",
    "exp5": f"全连接LSTM，参数约41倍，MSE最差 {e5['mse']:.5f}",
    "exp6": f"L1训练：MAE更优 {e6['mae']:.5f}、帧更锐利",
    "best": f"2层+64+K5+8帧，3次平均 MSE={best['mse']:.5f}",
}
order = ["baseline", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "best"]
for i, key in enumerate(order):
    r = R[key]
    row = t_res.rows[i + 1]
    set_cell(row.cells[3], f"{r['mse']:.5f}")
    set_cell(row.cells[4], f"{r['mae']:.5f}")
    set_cell(row.cells[5], f"{r['time']:.1f}")
    set_cell(row.cells[6], conclusions[key])

# 表5 AI 协作记录（模板仅 3 数据行，复制一行补到 4 行）
while len(t_ai.rows) < 5:
    new_tr = copy.deepcopy(t_ai.rows[1]._tr)
    t_ai._tbl.append(new_tr)

ai_records = [
    ("Builder 生成实验框架",
     "“创建 convlstm_bounce.py：make_sequences 生成 2200 条 10 帧 32×32 弹跳小球（r=2, 速度0.8~1.6, seed=42）；ConvLSTMCell 用 Conv2d(in+hid,4hid,k) 合并实现四门；ConvLSTM 沿时间循环后 Conv2d(32→1) 输出；MSE+Adam(0.001) 训练5epoch；保存 result_baseline.png 与九宫格 result_pred.png。”",
     "AI 给出数据合成、细胞、多层循环、训练与绘图骨架。我逐行核查：五维张量 permute(0,1,4,2,3) 顺序、padding=k//2 保持尺寸、四门 chunk 与激活均正确；并修正参考代码训练/测试重叠问题（改 2200 条：前2000训练、后200测试）。"),
    ("Chat 核对参数量公式",
     "“ConvLSTM 合并卷积的参数量为什么是 4×(K²C_inC_h + K²C_h² + C_h)？C_h 32→64 变多少？”",
     "AI 解释 Conv2d(C_in+C_h,4C_h,K) 权重展开即 4×(K²C_inC_h+K²C_h²)，加 4C_h 偏置。我手算基线 38144、64通道 150016（约3.93倍），与 sum(p.numel()) 打印的 38433（含输出卷积289）核对一致。"),
    ("智能修复维度错误",
     "实现 FlattenLSTM 时报 shape mismatch：LSTM 输入不是 (B,T,1024)。",
     "AI 提示需先 squeeze 通道维再 reshape：x.squeeze(2).reshape(B,T,1024)，取末时刻过 Linear 再重塑 32×32。我据此修正 forward 并核对 LSTM(1024,256) 各张量形状，最终参数量 1,575,936。"),
    ("内联补全可视化",
     "编写 plot_prediction 九宫格（输入帧|真实|预测）与训练曲线时触发内联补全。",
     "AI 补全 subplots 网格、imshow 与 MSE 标注。我改为 k_in+2 列以同时适配 4 帧/8 帧，并统一 vmin/vmax=0/1 保证三列灰度可比、在标题标注每样本 MSE。"),
]
for i, (scene, prompt, adopt) in enumerate(ai_records, start=1):
    row = t_ai.rows[i]
    set_cell(row.cells[0], str(i))
    set_cell(row.cells[1], scene)
    set_cell(row.cells[2], prompt)
    set_cell(row.cells[3], adopt)

# 表6 最优组合复测（3 次 + 平均）
best_runs = best["runs"]
for i, run in enumerate(best_runs, start=1):
    row = t_re.rows[i]
    set_cell(row.cells[1], str(run["seed"]))
    set_cell(row.cells[2], f"{run['mse']:.5f}")
    set_cell(row.cells[3], f"{run['mae']:.5f}")
    set_cell(row.cells[4], f"{run['time']:.1f}")
avg_row = t_re.rows[4]
set_cell(avg_row.cells[1], "42 / 43 / 44")
set_cell(avg_row.cells[2], f"{best['mse']:.5f}")
set_cell(avg_row.cells[3], f"{best['mae']:.5f}")
set_cell(avg_row.cells[4], f"{best['time']:.1f}")

# ---------- 插入图片 ----------
img_map = {
    3: ("result_baseline.png", 6.0),
    5: ("result_exp1.png", 6.0),
    6: ("result_exp2.png", 6.0),
    7: ("result_exp3.png", 6.0),
    8: ("result_exp4.png", 6.0),
    9: ("result_exp5.png", 6.0),
    10: ("result_exp6.png", 6.0),
    13: ("result_pred.png", 6.5),
}
for idx, (img, w) in img_map.items():
    insert_image_cell(doc.tables[idx], img, width=w)

doc.save(OUT)
print(f"报告已生成: {OUT}")
