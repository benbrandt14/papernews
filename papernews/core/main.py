import tomllib
import pluggy
from prefect import flow, task
from papernews.models import RawDocument, LayoutChunk
from papernews.core.router import ROUTER
from papernews.plugins import hookspecs, rss_plugin, academic_plugin
from papernews.render import build_pdf

@task(retries=3, retry_delay_seconds=5)
def process_single_document(doc: RawDocument) -> LayoutChunk:
    """Processes a single document using the appropriate strategy from ROUTER."""
    strategy = ROUTER.get(doc.content_type)
    if not strategy:
        raise ValueError(f"No processing strategy found for content_type: {doc.content_type}")
    return strategy(doc)

@flow(name="Build Newspaper")
def build_newspaper():
    """Main execution flow to ingest, process, and render the newspaper."""

    # 1. Load the actual configuration
    try:
        with open("sources.toml", "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        print("Warning: sources.toml not found. Using an empty configuration.")
        config = {}

    # 2. Setup pluggy
    pm = pluggy.PluginManager("papernews")
    pm.add_hookspecs(hookspecs)
    pm.register(rss_plugin)
    pm.register(academic_plugin)

    # 3. Ingest documents using the active plugins
    docs = []
    results = pm.hook.fetch_documents(config=config)
    for result_list in results:
        docs.extend(result_list)

    print(f"Fetched {len(docs)} documents.")

    # 4. Process documents concurrently (Prefect maps the task automatically)
    chunks = process_single_document.map(docs)

# 5. Resolve the generated LayoutChunks
    print(f"Generated {len(chunks)} layout chunks.")
    valid_chunks = []
    for chunk_future in chunks:
        try:
            # Resolve the Prefect future object
            chunk = chunk_future.result() if hasattr(chunk_future, 'result') else chunk_future
            valid_chunks.append(chunk)
        except Exception as e:
            print(f" - Failed to process a document: {e}")

    # 6. Render the Newspaper!
    if valid_chunks:
        build_pdf(valid_chunks)
    else:
        print("No valid chunks generated. Skipping PDF build.")

if __name__ == "__main__":
    build_newspaper()