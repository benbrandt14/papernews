from __future__ import annotations

import logging
from pathlib import Path
from pypdf import PdfReader
import chromadb
from google import genai
import os

log = logging.getLogger(__name__)

def _log(msg: str) -> None:
    import sys
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()

def ingest_pdfs(pdf_dir: Path, collection_name: str = "archives"):
    """Reads PDFs from a directory, chunks them, and stores them in ChromaDB."""
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        _log(f"[ingest-archives] Invalid directory: {pdf_dir}")
        return

    _log(f"[ingest-archives] Initializing ChromaDB client for collection '{collection_name}'...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    genai_client = genai.Client()

    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        _log(f"[ingest-archives] No PDFs found in {pdf_dir}")
        return

    _log(f"[ingest-archives] Found {len(pdfs)} PDFs.")

    for pdf_path in pdfs:
        _log(f"  -> Reading {pdf_path.name}")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            _log(f"  [error] Failed to read {pdf_path.name}: {e}")
            continue

        if not text.strip():
            continue

        chunks = _chunk_text(text, max_chars=2000)
        _log(f"     Generated {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.name}_chunk_{i}"

            existing = collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                continue

            try:
                response = genai_client.models.embed_content(
                    model='text-embedding-004',
                    contents=chunk,
                )
                embedding = response.embeddings[0].values

                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": pdf_path.name}]
                )
            except Exception as e:
                _log(f"  [error] Failed to embed chunk {i}: {e}")

def _chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = p
        else:
            current_chunk += "\n\n" + p
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_archives_article(articles: list[dict], collection_name: str = "archives") -> dict | None:
    if not articles:
        return None

    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        _log(f"[archives] Could not load ChromaDB (have you ingested yet?): {e}")
        return None

    if collection.count() == 0:
        _log("[archives] Collection is empty. Skipping.")
        return None

    _log("[archives] Determining today's theme...")
    genai_client = genai.Client()

    context_items = "\n".join([f"- {a['title']}: {a.get('summary', '')}" for a in articles[:5]])

    prompt = (
        "You are an editor for a daily news digest. Look at the following top stories "
        "and identify a single overarching specific theme, technology, or cultural trend "
        "that ties 1 or 2 of them together. Respond with ONLY a short search query "
        "(max 5 words) representing this theme (e.g., 'artificial intelligence breakthroughs' or 'space exploration').\n\n"
        f"Top stories:\n{context_items}"
    )

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        theme_query = response.text.strip()
        _log(f"[archives] Selected theme: '{theme_query}'")
    except Exception as e:
        _log(f"[archives] Failed to generate theme: {e}")
        return None

    _log(f"[archives] Searching archives for '{theme_query}'...")
    try:
        embed_res = genai_client.models.embed_content(
            model='text-embedding-004',
            contents=theme_query,
        )
        query_embedding = embed_res.embeddings[0].values

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        if not results['documents'] or not results['documents'][0]:
            _log("[archives] No relevant historical articles found.")
            return None

        best_doc = results['documents'][0][0]
        source_meta = results['metadatas'][0][0]['source']
        _log(f"[archives] Found relevant chunk from {source_meta}")
    except Exception as e:
        _log(f"[archives] Failed to query archives: {e}")
        return None

    _log("[archives] Extracting direct historical text...")

    try:
        from datetime import date

        # Clean up the filename for the title (e.g., "Popular_Mechanics_1985.pdf" -> "Popular Mechanics 1985")
        clean_title = source_meta.replace('_', ' ').replace('.pdf', '')

        return {
            "source": "From the Archives",
            "url": f"local://archives/{source_meta}",
            "title": clean_title,
            "text": best_doc.strip(),
            "summary": f"A relevant excerpt from {clean_title} matching the theme: {theme_query}",
            "date": date.today().strftime("%b %d, %Y"),
        }
    except Exception as e:
        _log(f"[archives] Failed to extract archive article: {e}")
        return None
