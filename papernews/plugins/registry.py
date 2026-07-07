"""Single place where the plugin manager is constructed.

Every pipeline stage that fires hooks goes through get_plugin_manager()
so hookspecs are always loaded and built-in plugins are always
registered — no more ad-hoc PluginManager construction per stage.
"""

from __future__ import annotations

import pluggy

from papernews.plugins import hookspecs


def get_plugin_manager() -> pluggy.PluginManager:
    """Build a plugin manager with hookspecs and all built-in plugins."""
    pm = pluggy.PluginManager("papernews")
    pm.add_hookspecs(hookspecs.PapernewsSpec)

    from papernews.plugins import (
        curiosity_plugin,
        hn_plugin,
        rss_plugin,
        wiki_plugin,
    )

    pm.register(rss_plugin)
    pm.register(hn_plugin)
    pm.register(wiki_plugin)
    pm.register(curiosity_plugin)

    pm.check_pending()
    return pm
