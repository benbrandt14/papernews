"""URL and title normalization for the article registry.

The registry keys articles by URL, but feeds serve the same story under
cosmetically different URLs (tracking params, http vs https, trailing
slashes, www) and aggregators resurface a story the original feed already
carried. Canonicalizing the key — and keeping a normalized title as a
second key — is what makes "never typeset twice" hold in practice.

Canonical forms are registry keys only; display always uses the original.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

# Query parameters that identify the click, not the page.
_TRACKING_PARAM_RE = re.compile(
    r"^(utm_\w+|fbclid|gclid|msclkid|mc_[ce]id|ref|ref_src|cmpid|source|s|si)$",
    re.IGNORECASE,
)

_TITLE_JUNK_RE = re.compile(r"[^a-z0-9]+")


def canonical_url(url: str) -> str:
    """Normalize a URL into a stable registry key.

    Folds scheme to https, lowercases and strips www. from the host,
    drops tracking query params, the fragment, and any trailing slash.
    Non-URL strings pass through stripped, so odd source_ids still key.
    """
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return url

    netloc = parts.netloc.lower()
    netloc = netloc.removeprefix("www.")
    query = "&".join(
        pair
        for pair in parts.query.split("&")
        if pair and not _TRACKING_PARAM_RE.match(pair.split("=", 1)[0])
    )
    path = parts.path.rstrip("/")
    return urlunsplit(("https", netloc, path, query, ""))


def title_key(title: str) -> str:
    """Normalize a title into a fuzzy-exact dedupe key.

    Lowercase, alphanumerics only, single-space separated. Deliberately
    conservative: only *identical* normalized titles collide, so two
    different stories about the same event never mask each other.
    Returns "" for titles too short to be distinctive — callers must
    treat "" as "no title key" and never match on it.
    """
    key = _TITLE_JUNK_RE.sub(" ", title.lower()).strip()
    if len(key) < 12:
        return ""
    return key
