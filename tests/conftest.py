import os

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
