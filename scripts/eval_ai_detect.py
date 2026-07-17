#!/usr/bin/env python3
"""Empirical evaluation of papernews.ai_detect against a labeled corpus.

The AI-likeness screen ships with hand-set weights (see ai_detect.py for
why the upstream AITextDetector model couldn't be reused: it is trained
on Chinese creative writing). This harness measures how those weights
actually discriminate on labeled English human/LLM text, so the
threshold knobs in [preferences] can be set on evidence instead of vibes.

Default corpus: HC3-English (Hello-SimpleAI, ~24k QA pairs with both
human and ChatGPT answers; the standard labeled human/LLM corpus).
Caveats to keep in mind when reading results: HC3 is Q&A prose, not
news, and its generations are ChatGPT-3.5-era — treat the numbers as a
sanity floor, not a certificate.

Usage (needs network for the one-time download; run locally or in CI —
the Claude sandbox blocks huggingface.co):

    uv run python scripts/eval_ai_detect.py                # download + evaluate
    uv run python scripts/eval_ai_detect.py --limit 4000   # faster subsample
    uv run python scripts/eval_ai_detect.py --threshold 0.5
    uv run python scripts/eval_ai_detect.py --jsonl my.jsonl  # any local corpus
    uv run python scripts/eval_ai_detect.py --train-baseline  # needs scikit-learn
    uv run python scripts/eval_ai_detect.py --smoke        # offline plumbing check

A local --jsonl file must contain records with "human_answers" and/or
"chatgpt_answers" string-list fields (HC3's schema).

--train-baseline additionally trains the exact AITextDetector recipe
(TfidfVectorizer(analyzer="char", ngram_range=(2, 4)) + LinearSVC) on
an 80/20 split of the same corpus, showing what a trained linear model
buys over the hand-set weights. Requires `uv pip install scikit-learn`.

Reports ROC-AUC (threshold-free), per-class score distributions, and
the derank/false-positive rates at the configured threshold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papernews.ai_detect import score_text  # noqa: E402

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
CACHE = Path(__file__).resolve().parent / ".cache" / "hc3_all.jsonl"

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


def load_samples(path: Path, limit: int | None, seed: int) -> list[Sample]:
    """One sample per answer, keeping only reliably-sized texts.

    The screen only ever acts on `reliable` scores, so the evaluation
    mirrors that: too-short answers are excluded up front.
    """
    samples: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for text in rec.get(key) or []:
                    if text and score_text(text).reliable:
                        samples.append((label, text))
    random.Random(seed).shuffle(samples)
    if limit:
        samples = samples[:limit]
    return samples


def roc_auc(pairs: list[tuple[int, float]]) -> float:
    """Threshold-free ROC-AUC via the Mann-Whitney U rank statistic."""
    ranked = sorted(pairs, key=lambda p: p[1])
    n_pos = sum(1 for label, _ in ranked if label == 1)
    n_neg = len(ranked) - n_pos
    if not n_pos or not n_neg:
        raise SystemExit("need both classes present to compute AUC")
    # Average ranks over ties so identical scores don't bias either class.
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # ranks are 1-based
        rank_sum += avg_rank * sum(1 for k in range(i, j) if ranked[k][0] == 1)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


def histogram(values: list[float], width: int = 40) -> str:
    bins = [0] * 10
    for v in values:
        bins[min(9, int(v * 10))] += 1
    peak = max(bins) or 1
    return "\n".join(
        f"  {i / 10:.1f}–{(i + 1) / 10:.1f} {'#' * round(b / peak * width):<{width}} {b}"
        for i, b in enumerate(bins)
    )


def summarize(name: str, scores: list[float]) -> None:
    print(f"\n{name} (n={len(scores)})")
    print(
        f"  mean {sum(scores) / len(scores):.3f}"
        f" · p10 {percentile(scores, 10):.3f}"
        f" · median {percentile(scores, 50):.3f}"
        f" · p90 {percentile(scores, 90):.3f}"
    )
    print(histogram(scores))


def evaluate(samples: list[Sample], threshold: float) -> None:
    pairs = [(label, score_text(text).ai_likelihood) for label, text in samples]
    human = [s for label, s in pairs if label == 0]
    ai = [s for label, s in pairs if label == 1]

    print(f"\n=== papernews.ai_detect on {len(pairs)} labeled samples ===")
    print(f"ROC-AUC: {roc_auc(pairs):.4f}  (0.5 = coin flip, 1.0 = perfect)")
    summarize("human-written scores", human)
    summarize("AI-generated scores", ai)

    caught = sum(1 for s in ai if s >= threshold) / len(ai)
    false_pos = sum(1 for s in human if s >= threshold) / len(human)
    print(f"\nat ai_derank_threshold={threshold}:")
    print(f"  AI deranked (recall):        {caught:.1%}")
    print(f"  humans deranked (false pos): {false_pos:.1%}")
    print(
        "\nremember the design goal: configurable noise reduction, not perfect"
        "\ndetection — a mediocre recall with a low false-positive rate is fine."
    )


def train_baseline(samples: list[Sample], seed: int) -> None:
    """The upstream AITextDetector recipe, retrained on this English corpus."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.svm import LinearSVC
    except ImportError:
        raise SystemExit(
            "--train-baseline needs scikit-learn: uv pip install scikit-learn"
        ) from None

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    train, test = shuffled[:cut], shuffled[cut:]

    print("\n=== trained baseline (TF-IDF char 2-4 + LinearSVC) ===")
    print(f"train {len(train)} / test {len(test)}")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
    x_train = tfidf.fit_transform([t for _, t in train])
    svc = LinearSVC()
    svc.fit(x_train, [label for label, _ in train])

    x_test = tfidf.transform([t for _, t in test])
    y_test = [label for label, _ in test]
    print(f"accuracy: {accuracy_score(y_test, svc.predict(x_test)):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, svc.decision_function(x_test)):.4f}")
    print(f"features: {len(tfidf.vocabulary_)} char n-grams")


SMOKE_HUMAN = (
    "The launch slipped again. Nobody at the pad seemed surprised. Engineers "
    "flagged a helium leak on Tuesday, and by Thursday the fix was still being "
    "argued over in a windowless conference room. The customer, a small "
    "startup out of Helsinki, took the delay in stride; their last ride waited "
    "five months. What worries the range office is the weather, since a front "
    "is stalling over the Gulf and the recovery ship cannot hold station in "
    "nine-foot swells. If Saturday scrubs, the next window opens on the 14th. "
    "Meanwhile the booster sat in the hangar, grid fins folded. Turnaround "
    "used to take months. Now the bottleneck is paperwork, one official "
    "joked, not hardware. Nobody laughed harder than the schedulers."
)
SMOKE_AI = (
    "In today's fast-paced world, this launch is a testament to human "
    "ingenuity. The mission plays a crucial role in the ever-evolving "
    "landscape of spaceflight. Moreover, the rocket seamlessly integrates "
    "cutting-edge technology with proven practices. It's important to note "
    "that the provider has embarked on a journey to revolutionize access to "
    "orbit. Furthermore, the booster underscores the importance of "
    "sustainable solutions. Additionally, the mission provides valuable "
    "insights into the future of the industry. In conclusion, this launch is "
    "a game-changer that will elevate your understanding of rocketry. "
    "Whether you're a casual observer or a seasoned expert, the ever-changing "
    "landscape offers a rich tapestry of innovation delivered seamlessly."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", type=Path, help="local corpus instead of HC3")
    ap.add_argument("--limit", type=int, help="subsample size (after shuffle)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1789)
    ap.add_argument("--train-baseline", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="offline plumbing check")
    args = ap.parse_args()

    if args.smoke:
        samples: list[Sample] = [(0, SMOKE_HUMAN), (1, SMOKE_AI)] * 20
        evaluate(samples, args.threshold)
        return

    path = args.jsonl or download_hc3()
    samples = load_samples(path, args.limit, args.seed)
    evaluate(samples, args.threshold)
    if args.train_baseline:
        train_baseline(samples, args.seed)


if __name__ == "__main__":
    main()
