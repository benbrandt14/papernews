from papernews.models import ArticleChunk

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

    return data
