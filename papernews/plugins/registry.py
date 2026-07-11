"""Single place where the plugin manager is constructed.

Every pipeline stage that fires hooks goes through get_plugin_manager()
so hookspecs are always loaded and built-in plugins are always
registered — no more ad-hoc PluginManager construction per stage.
"""

from __future__ import annotations

import os

import pluggy

from papernews.plugins import hookspecs


def _disabled_plugins() -> set[str]:
    """Plugin module names to skip, from PAPERNEWS_DISABLE_PLUGINS.

    Comma-separated short module names, e.g. "wiki_plugin,curiosity_plugin".
    Lets a deployment turn features off without code changes.
    """
    raw = os.environ.get("PAPERNEWS_DISABLE_PLUGINS", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def get_plugin_manager() -> pluggy.PluginManager:
    """Build a plugin manager with hookspecs and the enabled built-in plugins."""
    pm = pluggy.PluginManager("papernews")
    pm.add_hookspecs(hookspecs.PapernewsSpec)

    from papernews.plugins import (
        curiosity_plugin,
        hn_plugin,
        rss_plugin,
        salience_plugin,
        wiki_plugin,
    )

    disabled = _disabled_plugins()
    for module in (
        rss_plugin,
        hn_plugin,
        wiki_plugin,
        curiosity_plugin,
        salience_plugin,
    ):
        if module.__name__.rsplit(".", 1)[-1] not in disabled:
            pm.register(module)

    pm.check_pending()
    return pm
