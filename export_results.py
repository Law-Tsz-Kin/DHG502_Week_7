# -*- coding: utf-8 -*-
"""Run the full pipeline and export results.csv + results.html."""
import csv
import html
import re
import jieba
from qhchina import load_stopwords
from qhchina.analytics.topicmodels import LDAGibbsSampler

FILE = "毛泽东选集.txt"
N_TOPICS, ITERATIONS, SEED, MIN_WORDS = 10, 200, 42, 5

# ---------- 1. Split into documents ----------
with open(FILE, encoding="utf-8") as f:
    lines = f.read().split("\n")

for i, ln in enumerate(lines):
    if ln.strip() == "版本信息":
        lines = lines[:i]
        break

n = len(lines)
DATE_RE = re.compile(r"^（一九[〇○O0零一二三四五六七八九十]+年[^）]*）$")

cands = [i for i in range(n) if DATE_RE.match(lines[i].rstrip()) and len(lines[i]) <= 30]

doc_starts = []
for ci in cands:
    j = ci - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    title_block = []
    while j >= 0 and lines[j].strip() != "":
        title_block.insert(0, lines[j])
        j -= 1
    blanks = 0
    while j >= 0 and lines[j].strip() == "":
        blanks += 1
        j -= 1
    if blanks < 3 or not title_block:
        continue
    title = re.sub(r"[\*＊]\s*$", "", "".join(title_block).strip()).strip()
    m = re.search(r"一九([〇○O0零一二三四五六七八九]{2})年", lines[ci])
    if not m:
        continue
    zmap = {"〇": "0", "○": "0", "O": "0", "零": "0", "一": "1", "二": "2", "三": "3",
            "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
    year = 1900 + int("".join(zmap.get(c, c) for c in m.group(1)))
    doc_starts.append((title, year, ci))

documents = []
for idx, (title, year, start) in enumerate(doc_starts):
    end = doc_starts[idx + 1][2] if idx + 1 < len(doc_starts) else len(lines)
    body = []
    for ln in lines[start + 1:end]:
        t = ln.strip().replace("\u3000", "").replace(" ", "")
        if t == "注释":
            break
        if ln.strip().startswith("〔"):
            continue
        body.append(ln)
    text = re.sub(r"〔[0-9〇○一二三四五六七八九十]+〕", "", "\n".join(body))
    documents.append((title, year, text))

# ---------- 2. Tokenize ----------
stop = load_stopwords("zh_sim")
token_docs, kept = [], []
for title, year, text in documents:
    toks = [w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop and not w.isspace()]
    if len(toks) >= 100:
        token_docs.append(toks)
        kept.append((title, year, len(toks)))

# ---------- 3. LDA ----------
lda = LDAGibbsSampler(n_topics=N_TOPICS, iterations=ITERATIONS,
                      random_state=SEED, min_word_count=MIN_WORDS)
lda.fit(token_docs)
topics = lda.get_topics(n_words=10)
top_docs = [lda.get_top_documents(t, n_docs=2) for t in range(N_TOPICS)]

# ---------- 4. results.csv ----------
with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["topic_id", "rank", "word", "weight",
                "typical_doc_1", "typical_doc_1_year", "typical_doc_1_prob",
                "typical_doc_2", "typical_doc_2_year", "typical_doc_2_prob"])
    for t_id, words in enumerate(topics):
        d1, d2 = top_docs[t_id]
        t1, y1, _ = kept[d1[0]]
        t2, y2, _ = kept[d2[0]]
        for rank, (word, weight) in enumerate(words, 1):
            w.writerow([t_id, rank, word, f"{weight:.5f}",
                        t1, y1, f"{d1[1]:.4f}", t2, y2, f"{d2[1]:.4f}"])

# ---------- 5. results.html ----------
COLORS = ["#e05c5c", "#e08b3a", "#d4a017", "#7a9e3b", "#3ba272",
          "#2fa3a8", "#4a7fd4", "#7a5fd4", "#c25fd4", "#d45f8a"]

decades = {}
for _, year, _ in kept:
    decades[f"{year // 10 * 10}s"] = decades.get(f"{year // 10 * 10}s", 0) + 1
dec_max = max(decades.values())

cards = []
for t_id, words in enumerate(topics):
    wmax = max(p for _, p in words)
    chips = "".join(
        f'<span class="chip"><span class="chip-word">{html.escape(w)}</span>'
        f'<span class="chip-bar" style="width:{p / wmax * 100:.0f}%;background:{COLORS[t_id]}"></span>'
        f'<span class="chip-w">{p * 100:.1f}%</span></span>'
        for w, p in words)
    docs_html = "".join(
        f'<li><span class="doc-title">{html.escape(kept[d[0]][0])}</span>'
        f'<span class="doc-year">（{kept[d[0]][1]}）</span>'
        f'<span class="doc-prob">p={d[1]:.3f}</span></li>'
        for d in top_docs[t_id])
    cards.append(f'<article class="card" style="--c:{COLORS[t_id]}">'
                 f'<h2><span class="t-num">主題 {t_id}</span></h2>{chips}'
                 f'<h3>最典型文獻</h3><ul class="docs">{docs_html}</ul></article>')

dec_bars = "".join(
    f'<div class="dec"><span class="dec-label">{d}</span>'
    f'<div class="dec-bar" style="width:{v / dec_max * 100:.0f}%">{v}</div></div>'
    for d, v in sorted(decades.items(), key=lambda x: x[0]))

all_docs = "".join(
    f"<tr><td>{html.escape(t)}</td><td>{y}</td><td>{nt}</td></tr>"
    for t, y, nt in kept)

page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>《毛澤東選集》LDA 主題模型分析</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1c2330; --muted:#6b7280; --line:#e5e7eb; }}
* {{ box-sizing:border-box; margin:0; }}
body {{ font-family:"PingFang TC","Microsoft JhengHei",system-ui,sans-serif;
       background:var(--bg); color:var(--ink); line-height:1.6; padding:32px 16px; }}
.wrap {{ max-width:1100px; margin:0 auto; }}
header h1 {{ font-size:1.7rem; margin-bottom:.3rem; }}
header p {{ color:var(--muted); }}
.badges {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 28px; }}
.badge {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:8px 14px; font-size:.85rem; }}
.badge b {{ font-size:1.05rem; display:block; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr)); gap:18px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-left:5px solid var(--c);
        border-radius:12px; padding:18px 20px; }}
.card h2 {{ font-size:1.05rem; color:var(--c); margin-bottom:12px; }}
.chips {{ display:flex; flex-direction:column; gap:6px; }}
.chip {{ display:grid; grid-template-columns:110px 1fr 52px; align-items:center; gap:10px;
        font-size:.9rem; }}
.chip-word {{ font-weight:600; }}
.chip-bar {{ height:10px; border-radius:5px; min-width:2px; }}
.chip-w {{ color:var(--muted); font-size:.78rem; text-align:right; }}
.card h3 {{ font-size:.85rem; color:var(--muted); margin:14px 0 6px; font-weight:600; }}
.docs {{ list-style:none; padding:0; }}
.docs li {{ display:flex; gap:8px; align-items:baseline; padding:3px 0; font-size:.9rem; }}
.doc-title {{ font-weight:600; }}
.doc-year {{ color:var(--c); }}
.doc-prob {{ margin-left:auto; color:var(--muted); font-size:.78rem; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:18px 20px; margin-top:26px; }}
.panel h2 {{ font-size:1.05rem; margin-bottom:12px; }}
.dec {{ display:grid; grid-template-columns:64px 1fr; gap:10px; align-items:center;
       margin:4px 0; font-size:.85rem; }}
.dec-bar {{ background:#4a7fd4; color:#fff; border-radius:5px; padding:2px 8px;
           font-size:.75rem; min-width:2em; }}
details summary {{ cursor:pointer; font-weight:600; }}
table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:.88rem; }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }}
footer {{ margin-top:26px; color:var(--muted); font-size:.8rem; }}
</style>
</head>
<body>
<div class="wrap">
<header>
<h1>《毛澤東選集》LDA 主題模型分析</h1>
<p>Latent Dirichlet Allocation（Gibbs Sampling）· qhchina + jieba</p>
<div class="badges">
<div class="badge"><b>252</b>切分文獻</div>
<div class="badge"><b>{len(kept)}</b>進入模型（≥100 詞）</div>
<div class="badge"><b>{N_TOPICS}</b>主題</div>
<div class="badge"><b>{ITERATIONS}</b>迭代</div>
<div class="badge"><b>seed={SEED}</b>min_word_count={MIN_WORDS}</div>
</div>
</header>
<div class="grid">{''.join(cards)}</div>
<div class="panel">
<h2>文獻年代分佈</h2>{dec_bars}
</div>
<div class="panel">
<details>
<summary>全部 {len(kept)} 篇進入模型的文獻（標題 · 年份 · 詞數）</summary>
<table><tr><th>標題</th><th>年份</th><th>詞數</th></tr>{all_docs}</table>
</details>
</div>
<footer>資料來源：毛澤東選集.txt · 分詞：jieba（≥2 字）· 停用詞：qhchina load_stopwords("zh_sim") · 結構化結果請見 results.csv</footer>
</div>
</body>
</html>"""

with open("results.html", "w", encoding="utf-8") as f:
    f.write(page)

print(f"Documents found: {len(documents)} | kept: {len(kept)}")
print("Wrote results.csv and results.html")
