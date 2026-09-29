# -*- coding: utf-8 -*-
"""
Vector RAG / GraphRAG / WikiRAG 对比实验 —— 实验作业四
对应教材第 8 章《检索增强生成》
单文件、开箱即跑：内置「云山大学」语料 15 篇、标注问题 20 道（15 内置 + 5 自拟）、
预抽取三元组 40 条与条目库 9 条。不配 LLM API 也能完成全部检索评测。

常用命令：
  python hw4_rag.py --mode eval              # 三范式 × 20 题检索评测
  python hw4_rag.py --mode vector --q Q01    # 单题：检索
  python hw4_rag.py --mode graph --q Q01 --scope local
  python hw4_rag.py --mode wiki --q Q11
  python hw4_rag.py --mode k_scan            # K=1/2/3/5 敏感性扫描
  python hw4_rag.py --mode full              # 一键跑全部并保存 results.json
  python hw4_rag.py --mrr_check 3,1,2        # MRR 手算复核
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

# ============================================================
# 1. 语料库「云山大学」：15 篇虚构文档
# ============================================================
CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019 年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年 10 人；云山一等奖学金 3000 元/年，比例约 5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月 15 日放寒假。",
    "doc13": "图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3 层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。",
    "doc15": "国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

# ============================================================
# 2. 问题集：3 类 × 5 题内置 + 5 题自拟 = 20 题
# ============================================================
QUESTIONS = [
    # ---- 内置 15 题 ----
    {"id": "Q01", "type": "multi", "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？", "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi", "q": "AI 协会的指导教师主讲课程的课程代码是什么？", "evidence": ["doc11", "doc04", "doc06"]},
    {"id": "Q03", "type": "multi", "q": "认知计算全国重点实验室依托的学院成立于哪一年？", "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi", "q": "《知识图谱导论》授课教师的研究方向是什么？", "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi", "q": "机器人战队的指导教师的研究方向是什么？", "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact", "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact", "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact", "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact", "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact", "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？", "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？", "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？", "evidence": ["doc06", "doc11"]},
    # ---- 自拟 5 题 ----
    {"id": "Q16", "type": "fact", "q": "计算机学院现任院长是谁？", "evidence": ["doc03"]},
    {"id": "Q17", "type": "multi", "q": "机器人战队指导教师所在学院的院长是谁？", "evidence": ["doc11", "doc04", "doc02"]},
    {"id": "Q18", "type": "global", "q": "学校有哪些学生社团？分别由谁指导？", "evidence": ["doc11"]},
    {"id": "Q19", "type": "multi", "q": "岭南超算中心参与共建的学院院长是谁？", "evidence": ["doc10", "doc03"]},
    {"id": "Q20", "type": "global", "q": "学校有哪些国际交流项目？", "evidence": ["doc15"]},
]

# ============================================================
# 3. 分块
# ============================================================
def chunk_docs(size=120, overlap=20):
    chunks, step = [], size - overlap
    for doc_id, text in CORPUS.items():
        i = 0
        while True:
            chunks.append((doc_id, text[i:i + size]))
            if i + size >= len(text):
                break
            i += step
    return chunks


# ============================================================
# 4. 嵌入后端
# ============================================================
class Embedder:
    def __init__(self, model_name):
        self.ok = False
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.ok = True
            print(f"[embed] 已加载 {model_name}")
        except Exception as e:
            print(f"[embed] 加载失败（{type(e).__name__}），降级为字符 3-gram 相似度")

    def encode(self, texts):
        import numpy as np
        if self.ok:
            return self.model.encode(list(texts), normalize_embeddings=True)
        M = np.zeros((len(texts), 512))
        for r, t in enumerate(texts):
            t = re.sub(r"\s", "", t)
            for i in range(max(1, len(t) - 2)):
                g = t[i:i + 3] if len(t) >= 3 else t
                M[r, zlib.crc32(g.encode("utf-8")) % 512] += 1
            n = np.linalg.norm(M[r]) or 1.0
            M[r] /= n
        return M


def _dedup(scored, k):
    docs, seen = [], set()
    for doc, s in scored:
        if doc not in seen:
            docs.append(doc)
            seen.add(doc)
            if len(docs) >= k:
                break
    return docs


# ============================================================
# 5. Vector RAG
# ============================================================
class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb):
        self.emb = emb
        self.chunks = chunk_docs()
        self.M = emb.encode([c[1] for c in self.chunks])

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1]
        hits = [(self.chunks[i][0], float(scores[i])) for i in order]
        return _dedup(hits, k)


# ============================================================
# 6. GraphRAG
# ============================================================
PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019 年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI 协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018 年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3 学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI 协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000 元/年", "doc09"), ("校长奖学金", "金额", "20000 元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000 元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958 年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380 万册", "doc13"), ("春季学期", "开学日期", "3 月 2 日", "doc12"),
]

COMMUNITY_SUMMARIES = [
    ("社区 C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。",
     ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区 C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；学院与新加坡南洋理工大学有联合培养项目。",
     ["doc02", "doc03", "doc10", "doc15"]),
    ("社区 C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。",
     ["doc09", "doc11", "doc13"]),
]


class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb):
        import networkx as nx
        self.emb = emb
        self.G = nx.DiGraph()
        for h, r, t, src in PRE_TRIPLES:
            self.G.add_edge(h, t, rel=r, src=src)
        try:
            comms = nx.community.greedy_modularity_communities(self.G.to_undirected())
        except Exception:
            comms = [set(self.G.nodes)]
        print(f"[graph] 实体 {self.G.number_of_nodes()} 个 / 三元组 {len(PRE_TRIPLES)} 条 / 社区 {len(comms)} 个")
        self.n_nodes = self.G.number_of_nodes()
        self.n_edges = len(PRE_TRIPLES)
        self.n_communities = len(comms)
        self.summ_M = emb.encode([s for s, _ in COMMUNITY_SUMMARIES])

    def _entities(self, query, topn=2):
        ents = sorted([n for n in self.G.nodes if n in query], key=len, reverse=True)
        return ents[:topn]

    def retrieve(self, query, k=5, scope="local"):
        return self._local(query, k) if scope == "local" else self._global(query, k)

    def _local(self, query, k):
        import numpy as np
        ents = self._entities(query)
        seen, scored = set(), []
        frontier = ents[:]
        for _ in range(3):
            nxt = []
            for e in frontier:
                edges = list(self.G.out_edges(e, data=True)) + list(self.G.in_edges(e, data=True))
                for h, t, d in edges:
                    key = (h, d["rel"], t)
                    if key not in seen:
                        seen.add(key)
                        scored.append((d["src"], f"{h} —{d['rel']}→ {t}"))
                    other = t if h == e else h
                    if other not in frontier:
                        nxt.append(other)
            nxt.sort(key=lambda x: 0 if self.G.out_degree(x) > 0 else 1)
            frontier = nxt[:6]
        if not scored:
            return self._global(query, k)
        qv = self.emb.encode([query])[0]
        sims = self.emb.encode([t for _, t in scored]) @ qv
        best = {}
        for (doc, txt), s in zip(scored, sims):
            if doc not in best or s > best[doc]:
                best[doc] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return _dedup(ranked, k)

    def _global(self, query, k):
        qv = self.emb.encode([query])[0]
        scores = self.summ_M @ qv
        top = scores.argsort()[::-1][:2]
        docs = []
        for ci in top:
            for d in COMMUNITY_SUMMARIES[ci][1]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]


# ============================================================
# 7. WikiRAG
# ============================================================
PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于 1958 年，省属重点大学，本科生约 2.1 万人，校训「格物致知」。地铁 3 号线「云大站」距东门 200 米，校巴 5 条线路。校历：春季学期 3 月 2 日开学。",
     ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于 2018 年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目（每年 15 人）。",
     ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是 1985 年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。参与岭南超算中心运行管理。",
     ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019 年入职，研究检索增强生成，主讲《大模型通识课》，指导 AI 协会。赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。",
     ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。",
     ["doc05"]),
    ("大模型通识课", "《大模型通识课》（AI2101）：3 学分 32 学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求，教材《人工智能——深度学习大模型智能体》。",
     ["doc06"]),
    ("知识图谱导论", "《知识图谱导论》（AI3305）：2 学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，需先修《数据结构》。",
     ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金 8000 元/年（约 2%）；校长奖学金 20000 元/年（全校 10 人）；云山一等奖学金 3000 元/年（约 5%）。",
     ["doc09"]),
    ("学生社团与校园生活", "AI 协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00-22:00 开放，人工智能分馆在理科楼 B 座。",
     ["doc11", "doc13"]),
]


class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb):
        self.emb = emb
        self.M = emb.encode([b for _, b, _ in PRE_ENTRIES])

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1][:3]
        docs = []
        for i in order:
            for d in PRE_ENTRIES[i][2]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]


# ============================================================
# 8. 评测
# ============================================================
def evaluate(systems, questions, k=5, verbose=True):
    if verbose:
        print(f"\n===== 检索评测（文档级，Top-{k}，{len(questions)} 题）=====")
        print(f"{'范式':<12}{'问题类型':<10}{'Recall@'+str(k):>9}{'MRR':>8} 完整命中")
    results = {}
    for name, s in systems.items():
        results[name] = {}
        for tp in ("fact", "multi", "global"):
            qs = [q for q in questions if q["type"] == tp]
            if not qs:
                continue
            recs, rrs, full = [], [], 0
            per_q = []
            for q in qs:
                docs = s.retrieve(q["q"], k)
                ev = set(q["evidence"])
                got = ev & set(docs[:k])
                recs.append(len(got) / len(ev))
                rank = next((i + 1 for i, d in enumerate(docs) if d in ev), 0)
                rrs.append(1.0 / rank if rank else 0.0)
                full += (got == ev)
                per_q.append(dict(qid=q["id"], retrieved=docs[:k],
                                  evidence=list(q["evidence"]), hit=list(got),
                                  rank=rank, recall=len(got) / len(ev)))
            results[name][tp] = dict(
                recall=sum(recs) / len(recs),
                mrr=sum(rrs) / len(rrs),
                full=full,
                total=len(qs),
                per_q=per_q,
            )
            if verbose:
                print(f"{name:<14}{tp:<12}{sum(recs)/len(recs):>7.3f}"
                      f"{sum(rrs)/len(rrs):>8.3f}  {full}/{len(qs)}")
    return results


# ============================================================
# 9. LLM 生成环节（可选）
# ============================================================
ANSWER_PROMPT = """你是问答助手。只依据下面资料回答问题；资料不足以回答时明确说「资料中未提及」，禁止编造。
问题：{q}
资料：
{ctx}
用 1~3 句话回答，句末用括号注明用到的资料编号。"""


def llm_chat(prompt, temperature=0.0):
    from openai import OpenAI
    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        return None
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(model=model, temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


# ============================================================
# 10. 生成知识图谱可视化图片
# ============================================================
def draw_graph(path="graph.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import networkx as nx

    _font_path = os.path.expanduser("~/.local/share/fonts/noto-sc/NotoSansSC-Regular.otf")
    if os.path.exists(_font_path):
        fm.fontManager.addfont(_font_path)
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
    plt.rcParams["axes.unicode_minus"] = False

    G = nx.DiGraph()
    for h, r, t, src in PRE_TRIPLES:
        G.add_edge(h, t, rel=r)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="Noto Sans SC")
    edge_labels = {(h, t): d["rel"] for h, t, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=6, font_family="Noto Sans SC")
    plt.title("云山大学知识图谱（GraphRAG）", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存 {path}")


# ============================================================
# 11. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业四：三种 RAG 对比实验")
    ap.add_argument("--mode", choices=["eval", "vector", "graph", "wiki",
                                        "k_scan", "full"], default="eval")
    ap.add_argument("--q", default="Q01")
    ap.add_argument("--scope", choices=["local", "global"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emb", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--mrr_check", default=None, help="如 3,1,2")
    args = ap.parse_args()

    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        m = sum(1.0 / r for r in ranks) / len(ranks)
        print("MRR = (" + " + ".join(f"1/{r}" for r in ranks) +
              f") / {len(ranks)} = {m:.3f}")
        return

    emb = Embedder(args.emb)
    systems = {
        "Vector RAG": VectorRAG(emb),
        "GraphRAG": GraphRAG(emb),
        "WikiRAG": WikiRAG(emb),
    }

    if args.mode == "eval":
        evaluate(systems, QUESTIONS, args.k)
        return

    if args.mode in ("vector", "graph", "wiki"):
        name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG"}[args.mode]
        q = next(q for q in QUESTIONS if q["id"].upper() == args.q.upper())
        docs = systems[name].retrieve(q["q"], args.k, args.scope)
        print(f"\n[{q['id']}·{q['type']}] {q['q']}")
        print("检索到的文档：", docs)
        ctx = "\n".join(f"[{d}] {CORPUS[d]}" for d in docs)
        ans = llm_chat(ANSWER_PROMPT.replace("{q}", q["q"]).replace("{ctx}", ctx))
        if ans:
            print("\n  回答：", ans)
        else:
            print("\n（未配置 LLM API，仅输出检索结果与上下文）\n" + ctx)
        return

    if args.mode == "k_scan":
        print("\n===== K 敏感性扫描（内置 15 题）=====")
        builtin = QUESTIONS[:15]
        scan = {}
        for k in [1, 2, 3, 5]:
            print(f"\n--- K={k} ---")
            scan[k] = evaluate(systems, builtin, k, verbose=True)
        with open("k_scan.json", "w", encoding="utf-8") as f:
            json.dump(scan, f, ensure_ascii=False, indent=2)
        print(f"\nK 扫描结果已保存 k_scan.json")
        return

    if args.mode == "full":
        print("\n========== 一键运行全部实验 ==========")
        # 1. 三范式 × 20 题评测
        eval_res = evaluate(systems, QUESTIONS, 5)

        # 2. K 敏感性扫描（内置 15 题）
        print("\n===== K 敏感性扫描 =====")
        builtin = QUESTIONS[:15]
        k_scan = {}
        for k in [1, 2, 3, 5]:
            print(f"\n--- K={k} ---")
            k_scan[k] = evaluate(systems, builtin, k, verbose=True)

        # 3. 失败案例收集
        print("\n===== 失败案例收集 =====")
        failures = []
        for name, s in systems.items():
            for q in QUESTIONS[:15]:
                docs = s.retrieve(q["q"], 5)
                ev = set(q["evidence"])
                got = ev & set(docs[:5])
                if len(got) < len(ev):
                    failures.append(dict(
                        system=name, qid=q["id"], qtype=q["type"],
                        question=q["q"], retrieved=docs[:5],
                        evidence=list(q["evidence"]), missed=list(ev - got),
                    ))
        print(f"共收集到 {len(failures)} 个失败案例")

        # 4. 知识图谱可视化
        draw_graph("graph.png")

        # 5. 保存全部结果
        out = dict(
            embed_model=args.emb,
            embed_ok=emb.ok,
            n_questions=len(QUESTIONS),
            n_builtin=15,
            n_custom=5,
            eval_k5=eval_res,
            k_scan=k_scan,
            failures=failures,
            graph=dict(n_nodes=systems["GraphRAG"].n_nodes,
                       n_edges=systems["GraphRAG"].n_edges,
                       n_communities=systems["GraphRAG"].n_communities),
        )
        with open("experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n全部实验完成，结果已保存 experiment_results.json")


if __name__ == "__main__":
    main()
