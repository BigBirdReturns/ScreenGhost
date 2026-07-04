"""Surface Capability Matrix — the decided capture path per platform.

Single source of truth for docs/SURFACE_CAPABILITY_MATRIX.md and the tests. Each
row is a *decision*: which of the user's inherited access paths ScreenGhost uses
on that surface, in preference order.

`proof` states how far the row is carried in-repo:
  * "event-schema" — api routing exercised against representative event payloads
  * "fixture"      — view-tree path proven by an adapter conformance fixture
  * "gap"          — no path; unsupported_surface, named

LIVE platform API integration (real LINE/Meta/marketplace connections) is NOT
built and is a frozen [2b]-class item. `api` rows are decided strategy + event
schema, not a live connection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.capture import STRATEGIES


@dataclass(frozen=True)
class Surface:
    key: str
    label: str
    strategies: Tuple[str, ...]   # ordered preference
    proof: str                    # event-schema | fixture | gap
    note: str


SURFACES: List[Surface] = [
    Surface("line_oa", "LINE Official Account", ("api", "view_tree"),
            "event-schema",
            "Messaging API webhook (the seller's own OA); view-tree only for "
            "in-app gaps. Thai arrives exact via the API."),
    Surface("fb_page_messenger", "Facebook Page / Messenger", ("api", "view_tree"),
            "event-schema",
            "Messenger Platform webhook via the seller's Page — the app's UI "
            "obfuscation and stripped accessibility are BYPASSED, not fought."),
    Surface("instagram_dm", "Instagram Direct", ("api",),
            "event-schema",
            "Instagram Messaging API via the linked Facebook Page."),
    Surface("marketplace", "Shopee / Lazada / TikTok Shop", ("api", "view_tree"),
            "event-schema",
            "Open-platform order/chat APIs where the seller is enrolled; "
            "view-tree fallback for chat the API omits."),
    Surface("web_storefront", "Web storefront / WebView", ("view_tree",),
            "fixture",
            "Exact text from the DOM/view tree; no API needed."),
    Surface("messenger_app_obfuscated", "Messenger app, accessibility stripped",
            ("none",), "gap",
            "No usable text on the scrape path (Meta strips accessibility). "
            "Route to the Page API instead; if the seller has no Page, this is "
            "an honest unsupported_surface, not a refutation of the design."),
]


def validate() -> None:
    keys = [s.key for s in SURFACES]
    assert len(keys) == len(set(keys)), "duplicate surface keys"
    for s in SURFACES:
        assert s.strategies, f"{s.key} has no strategies"
        assert all(x in STRATEGIES for x in s.strategies), f"{s.key} bad strategy"
        assert s.proof in ("event-schema", "fixture", "gap")


validate()
