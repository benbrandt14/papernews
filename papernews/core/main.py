import tomllib
import pluggy
from prefect import flow, task, get_run_logger
from typing import List
from papernews.models import RawDocument, ArticleChunk
from papernews.core.router import ROUTER
from papernews.plugins import hookspecs, rss_plugin, wiki_plugin
from papernews.render import build_pdf

# ==========================================
# STAGE 1: ZERO-COST TRIAGE
# ==========================================
@task(name="1. Fast Keyword Triage")
def zero_cost_triage(docs: List[RawDocument]) -> List[RawDocument]:
    """Drops obvious noise using standard string matching before any LLM costs are incurred."""
    logger = get_run_logger()
    survivors = []
    
    exclusion_terms = ["mice", "rat model", "in vitro", "patch notes", "minor update"]
    
    for doc in docs:
        # Pass through wiki events/quotes directly without length checks
        if doc.content_type in ["wiki_event", "wiki_quote"]:
            survivors.append(doc)
            continue
            
        text_lower = doc.raw_text.lower()
        if any(term in text_lower for term in exclusion_terms):
            logger.info(f"Dropped (Keyword Triage): {doc.metadata.get('title')}")
            continue
        if len(text_lower) < 300:
            logger.info(f"Dropped (Too Short): {doc.metadata.get('title')}")
            continue
            
        survivors.append(doc)
        
    logger.info(f"Triage complete. {len(docs)} raw -> {len(survivors)} survived.")
    return survivors

# ==========================================
# STAGE 2: RELEVANCE SCORING
# ==========================================
@task(name="2. Rank Relevance")
def rank_documents(docs: List[RawDocument]) -> List[RawDocument]:
    """Scores articles based on priority."""
    logger = get_run_logger()
    
    for doc in docs:
        # Pass through wiki content with top priority so it never gets cut by the budget
        if doc.content_type in ["wiki_event", "wiki_quote"]:
            doc.metadata["rank_priority"] = 1
            continue

        text_lower = doc.raw_text.lower()
        score = 3  # Default standard score
        
        if any(term in text_lower for term in ["breakthrough", "guideline", "fda approved", "unveil", "major"]):
            score = 1
        elif doc.content_type == "academic_pdf":
            score = 2 
            
        doc.metadata["rank_priority"] = score
        
    docs.sort(key=lambda x: x.metadata.get("rank_priority", 3))
    logger.info("Ranking complete.")
    return docs

# ==========================================
# STAGE 3: BUDGET ENFORCEMENT
# ==========================================
@task(name="3. Enforce LLM Budget")
def enforce_quota(docs: List[RawDocument], max_articles: int = 12) -> List[RawDocument]:
    """Strictly limits the number of articles passed to the expensive rewriting LLM."""
    logger = get_run_logger()
    
    if len(docs) > max_articles:
        logger.warning(f"Budget Cap: Dropping {len(docs) - max_articles} low-ranked articles to save LLM tokens.")
        docs = docs[:max_articles]
        
    logger.info(f"Final approved batch: {len(docs)} articles cleared for processing.")
    return docs

# ==========================================
# STAGE 4: EXPENSIVE REWRITE
# ==========================================
# ---> UPDATED: Return ArticleChunk
@task(name="4. Generate Article Chunk", retries=3, retry_delay_seconds=5)
def process_single_document(doc: RawDocument) -> ArticleChunk:
    """The expensive stage: Calls Gemini-2.5-Flash to rewrite and categorize the article."""
    strategy = ROUTER.get(doc.content_type)
    if not strategy:
        raise ValueError(f"No processing strategy found for content_type: {doc.content_type}")
    return strategy(doc)

# ==========================================
# MAIN ORCHESTRATION FLOW
# ==========================================
@flow(name="Build Newspaper Pipeline")
def build_newspaper():
    
    try:
        with open("sources.toml", "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        config = {}

    pm = pluggy.PluginManager("papernews")
    pm.add_hookspecs(hookspecs)
    pm.register(rss_plugin)
    pm.register(wiki_plugin)

    raw_docs = []
    plugin_results = pm.hook.fetch_documents(config=config)
    for result_list in plugin_results:
        raw_docs.extend(result_list)

    print(f"Total Raw Ingestion: {len(raw_docs)} documents.")

    survivors = zero_cost_triage(raw_docs)
    ranked_docs = rank_documents(survivors)
    final_docs = enforce_quota(ranked_docs, max_articles=10) 

    chunk_futures = process_single_document.map(final_docs)

    valid_chunks = []
    for future in chunk_futures:
        try:
            chunk = future.result() if hasattr(future, 'result') else future
            valid_chunks.append(chunk)
        except Exception as e:
            print(f"Failed processing chunk: {e}")

    if valid_chunks:
        build_pdf(valid_chunks)
    else:
        print("Pipeline yielded no renderable content chunks.")

if __name__ == "__main__":
    build_newspaper()