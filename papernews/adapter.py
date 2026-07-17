"""Stage 4 bridge: flattens pipeline models into Jinja template variables.

This is the single serialization point between the typed backend and the
bespoke Typst template. The template never reads Pydantic models directly;
everything it consumes is produced here.
"""

from papernews.models import ArticleChunk, RenderContext


def article_to_dict(chunk: ArticleChunk) -> dict:
    """
    Converts an ArticleChunk Pydantic model into a dictionary suitable for Jinja templating.
    Explicitly includes @property attributes from nested models like Telemetry which are
    otherwise ignored by model_dump().
    """
    # Start with standard Pydantic serialization
    data = chunk.model_dump()

    # Pydantic's model_dump() ignores @property methods by default.
    # The Jinja template strictly expects these fields to exist in the dictionary,
    # so we must inject them manually.
    if hasattr(chunk, "telemetry") and chunk.telemetry is not None:
        data["telemetry"]["formatted_tokens"] = chunk.telemetry.formatted_tokens
        data["telemetry"]["formatted_cost"] = chunk.telemetry.formatted_cost

    # Same story for the AI-likeness stylometrics footer.
    if chunk.ai_metrics is not None:
        m = chunk.ai_metrics
        data["ai_metrics"]["formatted_likelihood"] = m.formatted_likelihood
        data["ai_metrics"]["formatted_burstiness"] = m.formatted_burstiness
        data["ai_metrics"]["formatted_diversity"] = m.formatted_diversity
        data["ai_metrics"]["formatted_phrase_rate"] = m.formatted_phrase_rate

    # Filled in by build_pdf — emission needs the workdir for image fetching.
    data["body_typst"] = ""

    return data


def render_context_to_template_vars(ctx: RenderContext) -> dict:
    """Flatten a RenderContext into the variables template.typ.j2 expects.

    The template's `decorations` block mixes front-page decorations with
    run metadata (generation time, token/cost totals); this is where that
    template-facing shape is assembled.
    """
    decorations = ctx.decorations.model_dump()
    decorations.update(
        generation_time=ctx.generation_time,
        total_tokens=ctx.total_tokens,
        total_cost=ctx.total_cost,
    )

    return {
        "date": ctx.date,
        "articles": [article_to_dict(a) for a in ctx.articles],
        "decorations": decorations,
        "stats": ctx.stats.model_dump(),
        "lead_article_index": ctx.lead_article_index,
    }
