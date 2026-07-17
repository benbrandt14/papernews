import os
import random
import sys
from pathlib import Path

import pytest
from hypothesis import settings

# Register Hypothesis profiles
settings.register_profile("pr_check", max_examples=20)
settings.register_profile("deep_audit", max_examples=250)

# Conditionally load the profile based on the GitHub branch
github_ref = os.environ.get("GITHUB_REF", "")
if github_ref == "refs/heads/main":
    settings.load_profile("deep_audit")
else:
    settings.load_profile("pr_check")


# --- Tiny AI-likeness classifier fixture -------------------------------------
# Tests of the triage screen and the render footer need a real trained
# artifact (production ships none until one is trained on a real corpus).
# We train a miniature model with the actual trainer on two synthetic
# stylistic registers — enough to pin the plumbing, not a claim about
# real-world discrimination.

_HUMAN_POOL = [
    "The launch slipped again and nobody at the pad seemed surprised.",
    "Engineers flagged a helium leak on Tuesday morning during checkout.",
    "The customer, a small startup out of Helsinki, took the delay in stride.",
    "A front is stalling over the Gulf and the swells are building fast.",
    "The booster sat in the hangar with its grid fins folded like a bird.",
    "Turnaround used to take months; now the bottleneck is paperwork.",
    "Range safety wants another look at the destruct package first.",
    "Their last ride waited five months for a slot on a shared manifest.",
    "One official joked that schedules are written in pencil out here.",
    "The recovery ship cannot hold station in nine-foot seas, period.",
]
_AI_POOL = [
    "In today's fast-paced world, this launch is a testament to human ingenuity.",
    "The mission plays a crucial role in the ever-evolving landscape of spaceflight.",
    "Moreover, the rocket seamlessly integrates cutting-edge technology with proven practices.",
    "It's important to note that the provider has embarked on a journey to revolutionize access to orbit.",
    "Furthermore, the booster underscores the importance of sustainable solutions.",
    "Additionally, the mission provides valuable insights into the future of the industry.",
    "In conclusion, this launch is a game-changer that will elevate your understanding.",
    "Whether you're a casual observer or a seasoned expert, the landscape offers a rich tapestry of innovation.",
    "The team leverages the power of iterative design to navigate the complexities involved.",
    "This initiative unlocks the potential of rapid reusability across the sector.",
]


@pytest.fixture(scope="session")
def tiny_ai_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a miniature trained classifier artifact (session-cached)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train_ai_classifier import export, train

    rng = random.Random(7)
    samples = []
    for pool, label in ((_HUMAN_POOL, 0), (_AI_POOL, 1)):
        for _ in range(80):
            samples.append((label, " ".join(rng.sample(pool, 6))))

    artifact = train(
        samples,
        ngram_range=(2, 4),
        max_features=4000,
        min_df=2,
        seed=7,
        corpus_desc="tiny-test-fixture",
    )
    out = tmp_path_factory.mktemp("ai_model") / "tiny_ai_model.json.gz"
    export(artifact, out, samples)
    return out


@pytest.fixture
def ai_classifier_env(tiny_ai_model: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the pipeline at the tiny classifier for this test."""
    monkeypatch.setenv("PAPERNEWS_AI_MODEL", str(tiny_ai_model))
    return tiny_ai_model
