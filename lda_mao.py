# -*- coding: utf-8 -*-
"""Split 毛泽东选集 into documents, tokenize with jieba, and run LDA topic modeling."""
import re
import jieba
from qhchina import load_stopwords
from qhchina.analytics.topicmodels import LDAGibbsSampler

FILE = "毛泽东选集.txt"

with open(FILE, encoding="utf-8") as f:
    lines = f.read().split("\n")

# Cut trailing version-info block
for i, ln in enumerate(lines):
    if ln.strip() == "版本信息":
        lines = lines[:i]
        break

n = len(lines)

# A document date line: a short standalone line like （一九二五年十二月一日）
# Zero may be written 〇, ○, O or 0
DATE_RE = re.compile(r"^（一九[〇○O0零一二三四五六七八九十]+年[^）]*）$")

cands = [i for i in range(n) if DATE_RE.match(lines[i].rstrip()) and len(lines[i]) <= 30]


def is_notes_marker(ln):
    t = ln.strip().replace("\u3000", "").replace(" ", "")
    return t == "注释"


# Keep a candidate as a document start only if the block of lines right before it
# (its title) is itself preceded by a large blank run (>= 3 blank lines).
# Sub-date lines inside multi-part documents are preceded by only 1 blank line.
doc_starts = []  # (title, year, line_index)
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
        continue  # sub-date inside a document, not a new document
    title = "".join(title_block)
    title = re.sub(r"[\*＊]\s*$", "", title.strip()).strip()
    m = re.search(r"一九([〇○O0零一二三四五六七八九]{2})年", lines[ci])
    if not m:
        continue
    zmap = {"〇": "0", "○": "0", "O": "0", "零": "0", "一": "1", "二": "2", "三": "3",
            "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
    year = int("".join(zmap.get(c, c) for c in m.group(1)))
    doc_starts.append((title, year, ci))

documents = []
for idx, (title, year, start) in enumerate(doc_starts):
    end = doc_starts[idx + 1][2] if idx + 1 < len(doc_starts) else len(lines)
    body_lines = []
    for ln in lines[start + 1:end]:
        if is_notes_marker(ln):      # footnotes section begins -> stop
            break
        if ln.strip().startswith("〔"):  # footnote lines
            continue
        body_lines.append(ln)
    text = "\n".join(body_lines)
    text = re.sub(r"〔[0-9〇○一二三四五六七八九十]+〕", "", text)  # inline footnote markers
    documents.append((title, year, text))

print(f"Documents found: {len(documents)}")

stop = load_stopwords("zh_sim")
token_docs = []
kept = []
for title, year, text in documents:
    toks = [w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop and not w.isspace()]
    if len(toks) >= 100:
        token_docs.append(toks)
        kept.append((title, year, len(toks)))

print(f"Documents kept after tokenizing (>=100 tokens): {len(kept)}")
print(f"Total tokens: {sum(len(t) for t in token_docs)}")

lda = LDAGibbsSampler(n_topics=10, iterations=200, random_state=42, min_word_count=5)
lda.fit(token_docs)

topics = lda.get_topics(n_words=10)
for t_id, words in enumerate(topics):
    print(f"\n=== Topic {t_id} ===")
    print("Top words: " + ", ".join(w for w, p in words))
    for doc_id, prob in lda.get_top_documents(t_id, n_docs=2):
        title, year, ntok = kept[doc_id]
        print(f"  - {title} （{year}）  [p={prob:.3f}, tokens={ntok}]")
