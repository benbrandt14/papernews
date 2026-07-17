#!/usr/bin/env python3
"""Empirical evaluation of the AI-likeness classifier on a labeled corpus.

Scores every sample with the *installed* classifier artifact (the same
pure-Python inference path the pipeline runs — papernews.ai_detect) and
reports ROC-AUC, per-class score distributions, and the derank /
false-positive rates at the configured threshold. Train an artifact
first with scripts/train_ai_classifier.py; evaluating on a corpus the
model wasn't trained on (e.g. train on HC3, evaluate on a
ghostbuster-data slice) is the honest way to read these numbers.

Usage (needs network for the one-time HC3 download; run locally or in
CI — the Claude sandbox blocks huggingface.co):

    uv run python scripts/eval_ai_detect.py                # HC3 → evaluate
    uv run python scripts/eval_ai_detect.py --limit 4000   # faster subsample
    uv run python scripts/eval_ai_detect.py --threshold 0.5
    uv run python scripts/eval_ai_detect.py --jsonl my.jsonl   # any HC3-schema corpus
    uv run python scripts/eval_ai_detect.py --ghostbuster DIR  # ghostbuster-data checkout
    uv run --with scikit-learn python scripts/eval_ai_detect.py --smoke  # offline check
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ai_classifier import (  # noqa: E402
    Sample,
    download_hc3,
    load_ghostbuster_dir,
    load_hc3_jsonl,
)

from papernews.ai_detect import get_classifier, score_text  # noqa: E402


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
    clf = get_classifier()
    if clf is None:
        raise SystemExit(
            "no classifier artifact installed — train one first:\n"
            "  uv run --with scikit-learn python scripts/train_ai_classifier.py\n"
            "or point PAPERNEWS_AI_MODEL at an artifact."
        )
    print(f"classifier: {clf.model_id} (trained on {clf.metadata.get('corpus', '?')})")

    pairs: list[tuple[int, float]] = []
    for label, text in samples:
        m = score_text(text)
        if m.reliable and m.ai_likelihood is not None:
            pairs.append((label, m.ai_likelihood))
    human = [s for label, s in pairs if label == 0]
    ai = [s for label, s in pairs if label == 1]

    print(f"\n=== {len(pairs)} reliable labeled samples ===")
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


def smoke() -> None:
    """Offline plumbing check: train a throwaway model on two synthetic
    registers, install it via PAPERNEWS_AI_MODEL, evaluate on held-out
    samples from the same registers."""
    from train_ai_classifier import export, train

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from conftest import _AI_POOL, _HUMAN_POOL  # type: ignore[import-not-found]

    rng = random.Random(11)
    # 12 sentences ≈ 140 words: comfortably above the reliability floor.
    make = lambda pool: " ".join(rng.choices(pool, k=12))  # noqa: E731
    samples = [(0, make(_HUMAN_POOL)) for _ in range(80)] + [
        (1, make(_AI_POOL)) for _ in range(80)
    ]
    held_out = [(0, make(_HUMAN_POOL)) for _ in range(20)] + [
        (1, make(_AI_POOL)) for _ in range(20)
    ]
    artifact = train(samples, (2, 4), 4000, 2, 11, "smoke")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "smoke_model.json.gz"
        export(artifact, out, samples)
        os.environ["PAPERNEWS_AI_MODEL"] = str(out)
        evaluate(held_out, threshold=0.6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", type=Path, help="local HC3-schema corpus instead of HC3")
    ap.add_argument("--ghostbuster", type=Path, help="ghostbuster-data checkout dir")
    ap.add_argument("--limit", type=int, help="subsample size (after shuffle)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1789)
    ap.add_argument("--smoke", action="store_true", help="offline plumbing check")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    samples: list[Sample] = []
    if args.ghostbuster:
        samples += load_ghostbuster_dir(args.ghostbuster)
    if args.jsonl:
        samples += load_hc3_jsonl(args.jsonl)
    if not samples:
        samples = load_hc3_jsonl(download_hc3())
    random.Random(args.seed).shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]
    evaluate(samples, args.threshold)


if __name__ == "__main__":
    main()
