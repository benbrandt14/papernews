#!/usr/bin/env python3
"""Train the AI-likeness classifier and export it for pure-Python inference.

This is the lyc8503/AITextDetector training recipe —
TfidfVectorizer(analyzer="char", ngram_range=(2, 4)) + LinearSVC — fitted
on labeled *English* human/LLM text instead of the upstream Chinese
creative-writing corpus, plus Platt probability calibration on a held-out
split. The fitted model is exported to a small gzipped-JSON artifact that
`papernews.ai_detect.LinearTextClassifier` evaluates in pure Python at
pipeline runtime (no scikit-learn in production).

Corpora (combine freely; held-out metrics are printed and stored in the
artifact so the numbers are reproducible):

  default          HC3-English (Hello-SimpleAI): ~24k questions with paired
                   human and ChatGPT answers. The standard labeled corpus.
  --ghostbuster D  A checkout of github.com/vivek3141/ghostbuster-data:
                   in-repo human vs gpt/claude text across essay, Reuters
                   news, and creative-writing domains. The `reuter` domain
                   is the closest match to papernews traffic.
  --jsonl F ...    Any local corpus in HC3 schema ("human_answers" /
                   "chatgpt_answers" string-list fields per record).

Usage (network needed only for the HC3 download; run locally or in CI —
the Claude sandbox blocks huggingface.co):

    uv run --with scikit-learn python scripts/train_ai_classifier.py
    uv run --with scikit-learn python scripts/train_ai_classifier.py \
        --ghostbuster ../ghostbuster-data --max-features 20000

Then commit the refreshed papernews/ai_model.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papernews.ai_detect import DEFAULT_MODEL_PATH, LinearTextClassifier  # noqa: E402

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
CACHE = Path(__file__).resolve().parent / ".cache" / "hc3_all.jsonl"

MIN_WORDS = 60  # skip stubs that carry no stylistic signal

# (label, text) — label 1 = AI-generated.
Sample = tuple[int, str]


def download_hc3() -> Path:
    if CACHE.exists():
        print(f"using cached corpus: {CACHE}")
        return CACHE
    import requests

    print(f"downloading HC3-English: {HC3_URL}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(HC3_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(CACHE, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"saved to {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    return CACHE


def _long_enough(text: str) -> bool:
    return len(text.split()) >= MIN_WORDS


def load_hc3_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for text in rec.get(key) or []:
                    if text and _long_enough(text):
                        samples.append((label, text))
    return samples


def load_ghostbuster_dir(root: Path) -> list[Sample]:
    """A vivek3141/ghostbuster-data checkout: domain/{human,gpt,claude,...}/*.txt.

    Anything under a `human` directory is label 0; other generator
    directories are label 1. Log-probability and prompt sidecars are
    skipped — only the actual text files train the model.
    """
    samples: list[Sample] = []
    for txt in sorted(root.rglob("*.txt")):
        parts = [p.lower() for p in txt.relative_to(root).parts]
        if any("logprob" in p or "prompt" in p or p == "perturb" for p in parts):
            continue
        text = txt.read_text(encoding="utf-8", errors="replace")
        if _long_enough(text):
            samples.append((0 if "human" in parts else 1, text))
    return samples


def train(
    samples: list[Sample],
    ngram_range: tuple[int, int],
    max_features: int,
    min_df: int,
    seed: int,
    corpus_desc: str,
) -> dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.svm import LinearSVC
    except ImportError:
        raise SystemExit(
            "training needs scikit-learn: rerun via "
            "`uv run --with scikit-learn python scripts/train_ai_classifier.py`"
        ) from None

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    # 80/10/10: fit / calibration / held-out test.
    fit_set = shuffled[: int(n * 0.8)]
    calib_set = shuffled[int(n * 0.8) : int(n * 0.9)]
    test_set = shuffled[int(n * 0.9) :]
    print(
        f"samples: {n} (fit {len(fit_set)} / calib {len(calib_set)} / test {len(test_set)})"
    )
    print(f"class balance: {sum(label for label, _ in shuffled)} AI / {n} total")

    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    x_fit = tfidf.fit_transform([t for _, t in fit_set])
    svc = LinearSVC(random_state=seed)
    svc.fit(x_fit, [label for label, _ in fit_set])

    # Platt calibration on its own split so the sigmoid isn't fit to
    # decision values the SVM already memorized.
    d_calib = svc.decision_function(tfidf.transform([t for _, t in calib_set]))
    platt = LogisticRegression()
    platt.fit(d_calib.reshape(-1, 1), [label for label, _ in calib_set])

    d_test = svc.decision_function(tfidf.transform([t for _, t in test_set]))
    y_test = [label for label, _ in test_set]
    p_test = platt.predict_proba(d_test.reshape(-1, 1))[:, 1]
    metrics = {
        "held_out_auc": round(float(roc_auc_score(y_test, d_test)), 4),
        "held_out_accuracy": round(
            float(accuracy_score(y_test, [int(p >= 0.5) for p in p_test])), 4
        ),
        "n_train": len(fit_set),
        "n_test": len(test_set),
        "corpus": corpus_desc,
        "seed": seed,
    }
    print(f"held-out ROC-AUC:  {metrics['held_out_auc']}")
    print(f"held-out accuracy: {metrics['held_out_accuracy']}")

    vocab = {g: int(i) for g, i in tfidf.vocabulary_.items()}
    return {
        "model_id": f"svc-char{ngram_range[0]}{ngram_range[1]}-en-{len(vocab) // 1000}k",
        "ngram_range": list(ngram_range),
        "vocabulary": vocab,
        "idf": [round(float(v), 6) for v in tfidf.idf_],
        "coef": [round(float(v), 6) for v in svc.coef_[0]],
        "intercept": round(float(svc.intercept_[0]), 6),
        "calibration": {
            "a": round(float(platt.coef_[0][0]), 6),
            "b": round(float(platt.intercept_[0]), 6),
        },
        "metadata": metrics,
    }


def export(artifact: dict, out: Path, verify_samples: list[Sample]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(artifact, f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB gzipped)")

    # Round-trip sanity: the pure-Python runtime must load and score.
    clf = LinearTextClassifier.load(out)
    probs = [clf.predict_proba(t) for _, t in verify_samples[:5]]
    print(
        f"runtime round-trip OK: model_id={clf.model_id}, sample probs {[f'{p:.3f}' for p in probs]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--jsonl", type=Path, nargs="*", default=[], help="HC3-schema corpora"
    )
    ap.add_argument("--ghostbuster", type=Path, help="ghostbuster-data checkout dir")
    ap.add_argument("--no-hc3", action="store_true", help="skip the HC3 download")
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=4)
    ap.add_argument("--limit", type=int, help="subsample size (after shuffle)")
    ap.add_argument("--seed", type=int, default=1789)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()

    samples: list[Sample] = []
    corpus_bits: list[str] = []
    if not args.no_hc3:
        samples += load_hc3_jsonl(download_hc3())
        corpus_bits.append("hc3-english")
    for path in args.jsonl:
        samples += load_hc3_jsonl(path)
        corpus_bits.append(path.name)
    if args.ghostbuster:
        gb = load_ghostbuster_dir(args.ghostbuster)
        print(f"ghostbuster-data: {len(gb)} samples")
        samples += gb
        corpus_bits.append("ghostbuster-data")
    if not samples:
        raise SystemExit("no training samples — provide a corpus")
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.limit]

    artifact = train(
        samples,
        (args.ngram_min, args.ngram_max),
        args.max_features,
        args.min_df,
        args.seed,
        "+".join(corpus_bits),
    )
    export(artifact, args.out, samples)


if __name__ == "__main__":
    main()
