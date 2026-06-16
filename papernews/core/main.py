import pluggy
from prefect import flow, task
from papernews.models import RawDocument, LayoutChunk
from papernews.core.router import ROUTER
from papernews.plugins import hookspecs, rss_plugin

@task(retries=3)
def process_single_document(doc: RawDocument) -> LayoutChunk:
    """Processes a single document using the appropriate strategy from ROUTER."""
    strategy = ROUTER.get(doc.content_type)
    if not strategy:
        raise ValueError(f"No processing strategy found for content_type: {doc.content_type}")
    return strategy(doc)

@flow(name="Build Newspaper")
def build_newspaper():
    """Main execution flow to ingest, process, and render the newspaper."""

    # 1. Setup pluggy
    pm = pluggy.PluginManager("papernews")
    pm.add_hookspecs(hookspecs)
    pm.register(rss_plugin)

    # 2. Ingest
    # Use a dummy config since this is a mock setup
    dummy_config = {}
    docs = []
    # call hooks, pluggy returns a list of results per plugin, so we flatten it
    results = pm.hook.fetch_documents(config=dummy_config)
    for result_list in results:
        docs.extend(result_list)

    print(f"Fetched {len(docs)} documents.")

    # 3. Process documents concurrently (Prefect maps the task)
    # Mapping in prefect 2/3 is done with task.map()
    chunks = process_single_document.map(docs)

    # Optional wait for completion if using deferral, but generally map resolves
    # depending on engine. To simply print we can iterate over the futures/results.
    print(f"Generated {len(chunks)} layout chunks:")
    for chunk in chunks:
        # In prefect 2/3, chunks from map are often futures. Resolving them:
        try:
            resolved_chunk = chunk.result() if hasattr(chunk, 'result') else chunk
            print(f" - {resolved_chunk.headline} ({resolved_chunk.template_type})")
        except Exception as e:
            print(f" - Failed to process a document: {e}")

if __name__ == "__main__":
    build_newspaper()
