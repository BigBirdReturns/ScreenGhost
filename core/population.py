"""A synthetic seller population with labeled ground truth.

The rate bench proved the engine survives arrival *volume*. This proves it
survives merchant *reality*: ambiguity, repetition, modification, cancellation,
Thai spelling variation, and code-switching, across many shop types — each with
a ground-truth order ledger the pipeline must reproduce or visibly fail to.

Honesty boundary, stated up front because it's load-bearing:
  * This generates messages AND labels them. So the metrics measure whether the
    same capture->event pipeline **preserves exact text, dedupes repeated
    observations, and reproduces the labeled ledger** across a broad population.
  * It does **not** prove the parser generalizes to real merchant Thai — the
    generator and parser share a grammar, so clean-cohort recall is expected to
    be high by construction. The *interesting* numbers are the adversarial
    suites, where the pipeline meets cases the view tree genuinely cannot
    resolve (same display name, coarse timestamps, payment screenshots with no
    text) and is required to fail in the open.
  * Synthetic sellers are not live sellers. No business lift is claimed here.

Deterministic: seeded ``random.Random`` so every receipt is reproducible.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.orders import ChatMessage, EventType


# --------------------------------------------------------------------------- #
# Catalog / seller model
# --------------------------------------------------------------------------- #
@dataclass
class Sku:
    code: str
    name: str            # Thai display name
    aliases: List[str]   # what buyers actually type
    variants: List[str]  # size/color; empty if none
    price: int
    in_stock: bool = True


@dataclass
class Buyer:
    buyer_id: str
    display_name: str    # what the view tree sees — may collide across buyers


@dataclass
class SellerProfile:
    seller_id: str
    cohort: str
    catalog: List[Sku]
    buyers: List[Buyer]
    channel: str         # line | fb_comment | inbox | live
    traffic: str         # normal | busy | hot | burst | quiet
    adversarial: List[str] = field(default_factory=list)


@dataclass
class LabeledMessage:
    msg: ChatMessage
    # ground truth
    is_order_bearing: bool
    event_type: str
    buyer_id: str
    sku: Optional[str]
    qty: Optional[int]
    variant: Optional[str]
    should_emit: bool
    ambiguous: bool = False   # unresolvable from the view tree alone
    adversarial: Optional[str] = None


@dataclass
class SellerWorld:
    profile: SellerProfile
    messages: List[LabeledMessage]
    ledger: Dict[Tuple[str, str], int]   # ground-truth final order ledger


# --------------------------------------------------------------------------- #
# Cohort definitions — small hand pools, expanded procedurally
# --------------------------------------------------------------------------- #
_COHORTS: Dict[str, Dict] = {
    "cosmetics":   {"skus": [("A01", "เซรั่มหน้าใส", ["serum", "เซรั่ม"]),
                              ("A02", "ครีมกันแดด", ["กันแดด", "sunscreen"])],
                    "variants": [], "adv": ["alias", "spelling", "codeswitch"]},
    "clothing":    {"skus": [("B01", "เสื้อยืด", ["เสื้อ", "tee"]),
                              ("B02", "กางเกงยีนส์", ["ยีนส์", "jeans"])],
                    "variants": ["S", "M", "L", "แดง", "ดำ"],
                    "adv": ["variant_ambig", "spelling"]},
    "bakery":      {"skus": [("C01", "เค้กวันเกิด", ["เค้ก", "cake"]),
                              ("C02", "คุกกี้", ["cookie"])],
                    "variants": [], "adv": ["preorder_addr", "chatter"]},
    "livemixed":   {"skus": [("D01", "กระเป๋า", ["bag"]),
                              ("D02", "รองเท้า", ["shoes"])],
                    "variants": ["37", "38", "39"],
                    "adv": ["repeat_cf", "multi_product", "codeswitch"]},
    "secondhand":  {"skus": [("E01", "มือสองเสื้อโค้ท", ["โค้ท", "coat"])],
                    "variants": [], "adv": ["dup_rows", "spelling"]},
    "phoneacc":    {"skus": [("F01", "เคสมือถือ", ["เคส", "case"]),
                             ("F02", "สายชาร์จ", ["สายชาร์จ", "cable"])],
                    "variants": ["iPhone", "Samsung"],
                    "adv": ["alias", "stockout_sub"]},
    "kids":        {"skus": [("G01", "ชุดเด็ก", ["ชุด"]),
                             ("G02", "ของเล่น", ["toy"])],
                    "variants": ["1ปี", "2ปี"], "adv": ["chatter", "modify"]},
    "collectibles":{"skus": [("H01", "โมเดล", ["model", "figure"])],
                    "variants": [], "adv": ["samedisplayname", "cancel"]},
    "homegoods":   {"skus": [("I01", "หมอน", ["pillow"]),
                             ("I02", "ผ้าห่ม", ["blanket"])],
                    "variants": [], "adv": ["modify", "cancel"]},
    "adversarial": {"skus": [("Z01", "สินค้า", ["item"]),
                             ("Z02", "สินค้า", ["item"])],  # deliberate alias clash
                    "variants": ["A", "B"],
                    "adv": ["samedisplayname", "repeat_cf", "payment_ref",
                            "variant_ambig", "dup_rows", "stockout_sub"]},
}

_CHATTER = ["สวยจังค่ะ", "ราคาเท่าไหร่คะ", "มีสีอื่นไหม", "ส่งวันไหนคะ",
            "❤️❤️", "สนใจค่ะ", "ทักแชทไปแล้วนะคะ"]
_POLITE = ["ค่ะ", "คับ", "ครับ", "จ้า", ""]
_NAMES = ["Nok", "Ploy", "Bee", "Mai", "Aum", "Fon", "Nan", "Beam", "Ann"]
_COHORT_KEYS = list(_COHORTS)


def _qty_phrase(rng: random.Random, qty: int) -> str:
    return rng.choice([f"x{qty}", f"{qty} ตัว", f"{qty} ชิ้น", f"{qty}pcs"])


def _sku_token(rng: random.Random, sku: Sku, adv: List[str]) -> str:
    """How the buyer refers to the item — code, name, or alias (ambiguity)."""
    choices = [sku.code, sku.name] + (sku.aliases if "alias" in adv else [])
    return rng.choice(choices)


def generate_population(n: int = 1000, seed: int = 1337) -> List[SellerProfile]:
    root = random.Random(seed)
    sellers: List[SellerProfile] = []
    for i in range(n):
        rng = random.Random(root.random())
        cohort = _COHORT_KEYS[i % len(_COHORT_KEYS)]
        spec = _COHORTS[cohort]
        catalog = [
            Sku(code, name, aliases,
                list(spec["variants"]),
                price=rng.randint(50, 1500),
                in_stock=not ("stockout_sub" in spec["adv"] and j == 0))
            for j, (code, name, aliases) in enumerate(spec["skus"])
        ]
        # Buyer pool; adversarial cohorts force display-name collisions.
        collide = "samedisplayname" in spec["adv"]
        buyers = []
        for b in range(rng.randint(6, 20)):
            # Default display names are unique (name+index) so identity is clean;
            # collision cohorts deliberately share one name across buyers, which
            # is the real "view tree can't tell two buyers apart" failure.
            name = _NAMES[0] if collide and b % 2 == 0 else f"{rng.choice(_NAMES)}{b}"
            buyers.append(Buyer(buyer_id=f"{cohort[:3]}{i}_b{b}", display_name=name))
        sellers.append(SellerProfile(
            seller_id=f"{cohort}-{i:04d}",
            cohort=cohort,
            catalog=catalog,
            buyers=buyers,
            channel=rng.choice(["line", "fb_comment", "inbox", "live"]),
            traffic=rng.choice(["normal", "busy", "hot", "burst", "quiet"]),
            adversarial=list(spec["adv"]),
        ))
    return sellers


def _emit(labels: List[LabeledMessage], seq: List[Tuple[str, str, Optional[str], Optional[int]]],
          m: ChatMessage, *, order_bearing: bool, etype: str, buyer: Buyer,
          sku: Optional[str], qty: Optional[int], variant: Optional[str],
          should_emit: bool, ambiguous: bool = False, adv: Optional[str] = None) -> None:
    labels.append(LabeledMessage(
        msg=m, is_order_bearing=order_bearing, event_type=etype,
        buyer_id=buyer.buyer_id, sku=sku, qty=qty, variant=variant,
        should_emit=should_emit, ambiguous=ambiguous, adversarial=adv))
    if should_emit:
        seq.append((etype, buyer.buyer_id, sku, qty))


def generate_stream(profile: SellerProfile, seed: int = 0
                    ) -> SellerWorld:
    """Produce a labeled comment stream + ground-truth ledger for one seller."""
    # Stable, process-independent seed (builtin hash() is salted per process,
    # which silently broke cross-run receipt reproducibility).
    _digest = hashlib.sha256(f"{profile.seller_id}|{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(_digest[:8], "big"))
    adv = profile.adversarial
    labels: List[LabeledMessage] = []
    ledger_seq: List[Tuple[str, str, Optional[str], Optional[int]]] = []
    n_msgs = {"quiet": 30, "normal": 80, "busy": 160,
              "hot": 260, "burst": 200}[profile.traffic]

    tick = 0

    def ts() -> str:
        nonlocal tick
        tick += 1
        return f"t{tick:05d}"

    for _ in range(n_msgs):
        roll = rng.random()
        buyer = rng.choice(profile.buyers)
        sku = rng.choice(profile.catalog)

        # ~25% chatter
        if roll < 0.25:
            m = ChatMessage(buyer.display_name, ts(), rng.choice(_CHATTER))
            _emit(labels, ledger_seq, m, order_bearing=False,
                  etype=EventType.CHATTER, buyer=buyer, sku=None, qty=None,
                  variant=None, should_emit=False)
            continue

        # payment-screenshot reference: order-bearing intent, no structured text
        if "payment_ref" in adv and roll < 0.30:
            m = ChatMessage(buyer.display_name, ts(), "โอนแล้วนะคะ [รูปสลิป]")
            _emit(labels, ledger_seq, m, order_bearing=True,
                  etype=EventType.CHATTER, buyer=buyer, sku=None, qty=None,
                  variant=None, should_emit=False, ambiguous=True, adv="payment_ref")
            continue

        qty = rng.randint(1, 3)
        variant = rng.choice(sku.variants) if sku.variants else None

        # spelling variation: a misspelled alias the catalog resolver can't
        # match -> a genuine order that fails to resolve. Honest recall miss.
        if "spelling" in adv and rng.random() < 0.12:
            garbled = sku.name[:-2] + sku.name[-1:] if len(sku.name) > 3 else sku.name + "ฟ"
            m = ChatMessage(buyer.display_name, ts(),
                            f"CF {garbled} {_qty_phrase(rng, qty)} ค่ะ")
            _emit(labels, ledger_seq, m, order_bearing=True, etype=EventType.ORDER,
                  buyer=buyer, sku=sku.code, qty=qty, variant=None,
                  should_emit=True, adv="spelling")
            continue

        token = _sku_token(rng, sku, adv)
        vtext = f" {variant}" if (variant and "variant_ambig" not in adv) else ""
        # variant_ambig: buyer omits the required variant -> unresolvable
        ambiguous_variant = bool(sku.variants) and "variant_ambig" in adv
        base = f"CF {token} {_qty_phrase(rng, qty)}{vtext} {rng.choice(_POLITE)}".strip()

        m = ChatMessage(buyer.display_name, ts(), base)
        _emit(labels, ledger_seq, m, order_bearing=True, etype=EventType.ORDER,
              buyer=buyer, sku=sku.code, qty=qty,
              variant=None if ambiguous_variant else variant,
              should_emit=True, ambiguous=ambiguous_variant,
              adv="variant_ambig" if ambiguous_variant else None)

        # duplicated visible row (scroll/replay re-shows the SAME comment)
        if "dup_rows" in adv and rng.random() < 0.3:
            labels.append(LabeledMessage(
                msg=m, is_order_bearing=True, event_type=EventType.ORDER,
                buyer_id=buyer.buyer_id, sku=sku.code, qty=qty, variant=variant,
                should_emit=False, adversarial="dup_rows"))  # must dedupe, no new event

        # repeated "CF" from same buyer (double-tap enthusiasm), distinct ts ->
        # is a genuine second order in our model unless identical within ts.
        if "repeat_cf" in adv and rng.random() < 0.2:
            m2 = ChatMessage(buyer.display_name, ts(), "CF CF ค่ะ")
            _emit(labels, ledger_seq, m2, order_bearing=True,
                  etype=EventType.ORDER, buyer=buyer, sku=None, qty=None,
                  variant=None, should_emit=False, ambiguous=True, adv="repeat_cf")

        # modify quantity later
        if "modify" in adv and rng.random() < 0.2:
            new_q = rng.randint(1, 5)
            m3 = ChatMessage(buyer.display_name, ts(),
                             f"เปลี่ยนเป็น {new_q} ชิ้น {sku.code} ค่ะ")
            _emit(labels, ledger_seq, m3, order_bearing=True,
                  etype=EventType.MODIFY, buyer=buyer, sku=sku.code, qty=new_q,
                  variant=None, should_emit=True, adv="modify")

        # cancel later
        if "cancel" in adv and rng.random() < 0.15:
            m4 = ChatMessage(buyer.display_name, ts(),
                             f"ยกเลิก {sku.code} ค่ะ")
            _emit(labels, ledger_seq, m4, order_bearing=True,
                  etype=EventType.CANCEL, buyer=buyer, sku=sku.code, qty=None,
                  variant=None, should_emit=True, adv="cancel")

        # stock-out substitution: item is out; ground truth is no clean line
        if "stockout_sub" in adv and not sku.in_stock and rng.random() < 0.5:
            m5 = ChatMessage(buyer.display_name, ts(),
                             f"CF {token} {_qty_phrase(rng, qty)} ค่ะ")
            _emit(labels, ledger_seq, m5, order_bearing=True,
                  etype=EventType.ORDER, buyer=buyer, sku=sku.code, qty=qty,
                  variant=None, should_emit=False, ambiguous=True, adv="stockout_sub")

    from core.orders import reduce_ledger
    ground_truth = reduce_ledger(ledger_seq)
    return SellerWorld(profile=profile, messages=labels, ledger=ground_truth)


def build_population(n: int = 1000, seed: int = 1337) -> List[SellerWorld]:
    return [generate_stream(p, seed=seed) for p in generate_population(n, seed)]


def resolve_sku(text: str, catalog: List[Sku]) -> Optional[str]:
    """Map a comment to a SKU code via the seller's catalog — the component a
    real agent needs because buyers type names/aliases, not codes.

    Literal code wins; otherwise a unique name/alias hit resolves; zero hits or
    an alias shared by two SKUs returns None (the genuine ambiguity failure).
    """
    import re

    low = text.lower()
    for s in catalog:
        if re.search(rf"\b{re.escape(s.code.lower())}\b", low):
            return s.code
    hits = set()
    for s in catalog:
        for token in [s.name] + s.aliases:
            if token and token.lower() in low:
                hits.add(s.code)
    return next(iter(hits)) if len(hits) == 1 else None
