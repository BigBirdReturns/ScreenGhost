"""Product-side catalog resolver — normalization + bounded fuzzy, honest misses.

Separate from `core.population.resolve_sku` (the frozen proof-stack baseline) so
improving the product does NOT move the proof numbers. This resolver adds:
  * Unicode NFC normalization + case/space + Thai politeness-particle stripping,
  * alias/name normalization,
  * a bounded (edit-distance <= 1) fuzzy alias match that resolves ONLY when
    exactly one catalog item matches — genuine ambiguity still returns None.

It never invents a SKU: zero matches or two matches → None, reported honestly.
And `variant_missing` flags an order whose SKU needs a variant the buyer didn't
give, so the pipeline marks it incomplete instead of hallucinating a done order.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

_PARTICLES = ("ค่ะ", "คะ", "ครับ", "คับ", "จ้า", "จ้ะ", "ค่า", "นะคะ", "นะครับ")
_MIN_FUZZY_LEN = 4  # don't fuzzy-match short tokens like "bag"/"tee"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").strip().lower()
    for p in _PARTICLES:
        s = s.replace(p, " ")
    return re.sub(r"\s+", " ", s).strip()


def _within_one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la < lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            i += 1  # skip one char in the longer string
    return True


def _fuzzy_contains(ntext: str, token: str) -> bool:
    token = normalize(token)
    if len(token) < _MIN_FUZZY_LEN:
        return False
    for L in (len(token) - 1, len(token), len(token) + 1):
        if L <= 0:
            continue
        for i in range(0, len(ntext) - L + 1):
            if _within_one_edit(ntext[i:i + L], token):
                return True
    return False


def resolve(text: str, skus: List) -> Optional[str]:
    """Map a comment to a SKU code, or None (zero or ambiguous match)."""
    low = text.lower()
    for s in skus:                                   # 1. literal code
        if re.search(rf"\b{re.escape(s.code.lower())}\b", low):
            return s.code
    ntext = normalize(text)
    hits = set()                                     # 2. normalized name/alias
    for s in skus:
        for tok in [s.name] + list(s.aliases):
            if tok and normalize(tok) and normalize(tok) in ntext:
                hits.add(s.code)
    if len(hits) == 1:
        return next(iter(hits))
    if hits:
        return None                                  # ambiguous -> honest miss
    fuzzy = set()                                    # 3. bounded fuzzy, unambiguous only
    for s in skus:
        for tok in [s.name] + list(s.aliases):
            if tok and _fuzzy_contains(ntext, tok):
                fuzzy.add(s.code)
    return next(iter(fuzzy)) if len(fuzzy) == 1 else None


def variant_missing(text: str, sku) -> bool:
    """True if the SKU requires a variant and the buyer named none."""
    if not getattr(sku, "variants", None):
        return False
    nt = normalize(text)
    return not any(normalize(v) and normalize(v) in nt for v in sku.variants)
