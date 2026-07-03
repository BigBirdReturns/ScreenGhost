"""Catalog CSV import/export — so the resolver uses a real imported catalog.

CSV columns: sku,name,aliases,variants,price,stock
`aliases` and `variants` are ';'-separated inside their cell, e.g.

    sku,name,aliases,variants,price,stock
    A01,เซรั่มหน้าใส,serum;เซรั่ม,,390,1
    B02,กางเกงยีนส์,ยีนส์;jeans,S;M;L,590,1
"""
from __future__ import annotations

import csv
from typing import Dict, List

from core.population import Sku

_FIELDS = ["sku", "name", "aliases", "variants", "price", "stock"]


def _split(cell: str) -> List[str]:
    return [p.strip() for p in (cell or "").split(";") if p.strip()]


def read_catalog_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(_FIELDS[:2]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"catalog CSV missing required columns: {missing}")
        rows = []
        for r in reader:
            if not (r.get("sku") or "").strip():
                continue
            rows.append({
                "sku": r["sku"].strip(),
                "name": (r.get("name") or "").strip(),
                "aliases": _split(r.get("aliases", "")),
                "variants": _split(r.get("variants", "")),
                "price": int(r["price"]) if (r.get("price") or "").strip() else 0,
                "stock": int(r["stock"]) if (r.get("stock") or "").strip() else 1,
            })
    return rows


def write_catalog_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for r in rows:
            w.writerow([r["sku"], r.get("name", ""),
                        ";".join(r.get("aliases", [])),
                        ";".join(r.get("variants", [])),
                        r.get("price", 0), r.get("stock", 1)])


def rows_to_skus(rows: List[Dict]) -> List[Sku]:
    """Adapt imported catalog rows to Sku objects the resolver understands."""
    return [Sku(code=r["sku"], name=r.get("name", ""),
                aliases=list(r.get("aliases", [])),
                variants=list(r.get("variants", [])),
                price=int(r.get("price", 0)),
                in_stock=bool(int(r.get("stock", 1))))
            for r in rows]
