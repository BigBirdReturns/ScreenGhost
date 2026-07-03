"""Load a deterministic demo fixture into SellerWorld objects.

For operator-demo repeatability only. The adversarial proof demo still generates
fresh worlds from the operator seed; this fixture is a fixed, human-readable set
of scenarios (clean / messy / adversarial / stock-out / repeat-CF / variant
ambiguity / payment screenshot / cancellation+modification).
"""
from __future__ import annotations

import json
from typing import List

from core.orders import ChatMessage, reduce_ledger
from core.population import (
    Buyer, LabeledMessage, SellerProfile, SellerWorld, Sku,
)


def load_seller_worlds(path: str) -> List[SellerWorld]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    worlds: List[SellerWorld] = []
    for s in data["sellers"]:
        catalog = [Sku(c["sku"], c.get("name", ""), list(c.get("aliases", [])),
                       list(c.get("variants", [])), int(c.get("price", 0)),
                       bool(c.get("stock", 1))) for c in s["catalog"]]
        buyers = [Buyer(b["buyer_id"], b["display_name"]) for b in s["buyers"]]
        profile = SellerProfile(
            seller_id=s["seller_id"], cohort=s.get("cohort", ""),
            catalog=catalog, buyers=buyers, channel=s.get("channel", "line"),
            traffic=s.get("traffic", "normal"), adversarial=s.get("adversarial", []))
        messages: List[LabeledMessage] = []
        seq = []
        for m in s["messages"]:
            cm = ChatMessage(m["buyer_display"], m["ts"], m["text"])
            lm = LabeledMessage(
                msg=cm, is_order_bearing=m.get("order", False),
                event_type=m.get("event", "chatter"), buyer_id=m["buyer_id"],
                sku=m.get("sku"), qty=m.get("qty"), variant=m.get("variant"),
                should_emit=m.get("should_emit", False),
                ambiguous=m.get("ambiguous", False),
                adversarial=m.get("adversarial"))
            messages.append(lm)
            if lm.should_emit:
                seq.append((lm.event_type, lm.buyer_id, lm.sku, lm.qty))
        worlds.append(SellerWorld(profile=profile, messages=messages,
                                  ledger=reduce_ledger(seq)))
    return worlds
