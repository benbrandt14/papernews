from __future__ import annotations

from dataclasses import dataclass

import trafilatura
from trafilatura.metadata import extract_metadata


@dataclass
class Article:
    source: str
    url: str
    title: str
    text: str
    published: str | None = None  # ISO date from page metadata, may be None
    format: str = "standard"
    tldr: str | None = None


def extract(url: str, title: str, source: str) -> Article | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_images=True,
        favor_precision=True, # Prevent navigation cruft & short entries from being included
    )
    if not text or len(text) < 200:
        return None
    published: str | None = None
    try:
        md = extract_metadata(downloaded)
        if md and md.date:
            published = md.date  # trafilatura returns "YYYY-MM-DD"
    except Exception:
        pass
    return Article(source=source, url=url, title=title, text=text, published=published)

def extract_arxiv_html(url: str, title: str, source: str, triage_data: dict) -> Article:
    """Extracts the full text from ArXiv HTML endpoint or returns the abstract if digest."""
    if triage_data.get("action") == "digest":
        return Article(
            source=source,
            url=url,
            title=title,
            text=triage_data.get("abstract", ""),
            format="digest",
            tldr=triage_data.get("tldr")
        )

    if triage_data.get("action") == "deep_dive":
        import requests
        from bs4 import BeautifulSoup
        import re

        # Convert /abs/ to /html/
        html_url = url.replace("/abs/", "/html/")
        # Ensure we hit a versioned url by default or we might get redirect. requests handles redirects.

        try:
            r = requests.get(html_url, timeout=15)
            r.raise_for_status()

            # If the URL wasn't versioned and arxiv gives us a versioned page, r.url will have it.
            base_url = r.url.rstrip('/')

            soup = BeautifulSoup(r.text, "html.parser")

            # Extract Math
            for math_tag in soup.find_all("math"):
                tex = math_tag.get("alttext", "")
                if tex:
                    # replace the tag with the raw text wrapped in $$
                    math_tag.replace_with(f"$${tex}$$")

            # Extract Images
            for img in soup.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "image")
                if not src.startswith("http"):
                    # construct absolute URL
                    # Arxiv src is usually relative like 'extracted/1234/fig.png'
                    src = f"{base_url}/{src.lstrip('/')}"
                md_img = f"![{alt}]({src})"
                img.replace_with(md_img)

            # Remove fluff
            for tag in soup.find_all(["nav", "footer"]):
                tag.decompose()

            # Remove references section. Usually in a div or section with class 'ltx_bibliography' or id 'bib'
            for tag in soup.find_all(class_="ltx_bibliography"):
                tag.decompose()

            # Extract textual paragraphs
            # We want to maintain some structure, but trafilatura is what we used before.
            # ArXiv HTML uses 'ltx_p' for paragraphs, 'ltx_title' for headers.
            # We can extract text simply by grabbing body text or using select.
            body = soup.find("body") or soup
            extracted_text = body.get_text(separator="\n\n", strip=True)

            # Basic cleanup
            extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)

            return Article(
                source=source,
                url=url,
                title=title,
                text=extracted_text,
                format="deep_dive",
                tldr=None
            )
        except Exception as e:
            # Fallback to standard
            import sys
            sys.stderr.write(f"  [warn] extract_arxiv_html failed: {e}\n")
            return Article(
                source=source,
                url=url,
                title=title,
                text=triage_data.get("abstract", ""),
                format="standard",
                tldr=None
            )

    # Default fallback
    return Article(source=source, url=url, title=title, text=triage_data.get("abstract", ""))
