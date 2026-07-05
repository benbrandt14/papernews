"""The explicit pluggy contract for papernews plugins.

Three hook families:

  * fetch_sources      — Stage 1 ingestion: produce RawDocuments.
  * enrich_articles    — Stage 3.5: whole-day, cross-article pass that
                         attaches sidecar data (salience, entities,
                         comments, open questions) to ArticleChunks.
  * fetch_decorations  — Stage 4: front-page decorations.

Registering the spec means pluggy validates hook signatures at
registration time instead of failing silently at call time.
"""

from __future__ import annotations

import pluggy

from papernews.config import AppConfig
from papernews.models import ArticleChunk, FrontpageDecorations, RawDocument
from papernews.store import SimpleStore

hookspec = pluggy.HookspecMarker("papernews")


class PapernewsSpec:
    @hookspec
    def fetch_sources(self, source_config: AppConfig) -> list[RawDocument]:
        """Fetch raw documents for the sources this plugin handles.

        Called once per run. Implementations must only act on the
        sources whose `kind` they own and return an empty list
        otherwise.
        """
        raise NotImplementedError

    @hookspec
    def enrich_articles(
        self,
        articles: list[ArticleChunk],
        source_config: AppConfig,
        store: SimpleStore,
    ) -> None:
        """Attach sidecar data to the day's articles, in place.

        This is the whole-day, cross-article pass: implementations see
        every surviving article at once, so they can interlink, score,
        or annotate across the edition. Mutating the passed ArticleChunk
        models is the intended mechanism; the return value is ignored.
        """
        raise NotImplementedError

    @hookspec
    def fetch_decorations(self, source_config: AppConfig) -> FrontpageDecorations:
        """Fetch front-page decorations (world news, quote, DYK facts).

        Results from multiple plugins are merged field-wise, later
        registrations winning. Only set the fields you actually fetched
        (leave the rest unset) so the merge stays meaningful.
        """
        raise NotImplementedError
