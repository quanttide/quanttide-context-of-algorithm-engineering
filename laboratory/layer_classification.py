#!/usr/bin/env python3
"""内容归属分类实验。

给一篇文档的正文，判断它属于哪一层（日志、洞察、意图、路线图……）。
标签取自文档实际所在的目录，无需人工标注。

用法：
    uv run --with scikit-learn python3 layer_classification.py
    uv run --with scikit-learn python3 layer_classification.py --step 1
    uv run --with scikit-learn python3 layer_classification.py --root ~/repos/quanttide
"""

import argparse
import os
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

LAYERS = ["journal", "insight", "intention", "roadmap",
          "context", "profile", "brochure", "report"]

SKIP_DIRS = {".git", "node_modules", "_build", "build", "dist", "target",
             ".venv", ".gocache", "coverage"}

STEPS = {
    1: ("日志与洞察", {"journal", "insight"}),
    2: ("内容流向四层", {"journal", "insight", "intention", "roadmap"}),
    3: ("八层全量", set(LAYERS)),
}

MIN_CHARS = 150
MAX_CHARS = 1500


def collect(roots):
    """遍历各仓库的 data/<层>/ 目录，取正文与标签（标签 = 层目录名）。"""
    texts, labels = [], []
    for base in roots:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            parts = root.split(os.sep)
            if "data" not in parts:
                continue
            i = parts.index("data")
            if i + 1 >= len(parts) or parts[i + 1] not in LAYERS:
                continue
            for name in files:
                if not name.endswith(".md"):
                    continue
                with open(os.path.join(root, name), errors="ignore") as fh:
                    raw = fh.read()
                # 去掉 Markdown 标记，只留文字
                text = re.sub(r"[#*`>\-\|\[\]()]", " ", raw)[:MAX_CHARS]
                if len(text.strip()) < MIN_CHARS:
                    continue
                texts.append(text)
                labels.append(parts[i + 1])
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=3, choices=sorted(STEPS),
                        help="1 两分类；2 四层；3 八层（默认）")
    parser.add_argument("--root", action="append", default=None,
                        help="仓库根目录，可重复；默认 ~/repos/quanttide 与 ~/repos/quanttide-founder")
    args = parser.parse_args()

    roots = args.root or [os.path.expanduser("~/repos/quanttide"),
                          os.path.expanduser("~/repos/quanttide-founder")]
    name, keep = STEPS[args.step]
    texts, labels = collect(roots)
    pairs = [(t, l) for t, l in zip(texts, labels) if l in keep]
    texts = [t for t, _ in pairs]
    labels = [l for _, l in pairs]

    counts = Counter(labels)
    baseline = max(counts.values()) / len(labels)

    model = make_pipeline(
        TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=3000, C=5),
    )
    scores = cross_val_score(model, texts, labels,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))

    print(f"第 {args.step} 步 {name}")
    print(f"  样本 {len(texts)}  类别 {len(counts)}")
    print(f"  分布 {dict(counts.most_common())}")
    print(f"  基线（全猜最多的一类） {baseline:.3f}")
    print(f"  模型 5 折准确率 {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()
