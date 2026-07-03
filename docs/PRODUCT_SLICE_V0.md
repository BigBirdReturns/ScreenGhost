# Product Slice v0 — Reviewable Order Ledger

**This is not live hardware proof. This is not business proof.** It is the first
usable seller workflow built on the proof spine: a local review loop that turns
ScreenGhost's captured messages into a corrected, exportable order ledger.

ScreenGhost's product object is a **reviewable order ledger**, not a dashboard
screenshot. Captured comments become *proposed* order events; a human reviews,
corrects, accepts or rejects; the final ledger exports with a receipt.

## Run it

```bash
python examples/review_ledger_demo.py --seed "op" --sellers 10          # populate, review, export
python examples/review_ledger_demo.py --seed "op" --sellers 10 --serve  # open the local review UI
```

`--serve` opens a localhost web UI: an inbox of proposed/needs_info events, the
raw capture + parsed event + catalog panels, a correction form, the seller
ledger, and an export button.

## What it is

- **Persistent local store** (`core/ledger_store.py`, SQLite): captures,
  order_events, order_event_sources, corrections, sellers, buyers,
  catalog_items, ledger_snapshots, exports. Raw captures are **write-once**;
  corrections are **append-only** with full lineage.
- **Review states**: `proposed → accepted / rejected / corrected / needs_info /
  cancelled / fulfilled`, with enforced transitions.
- **Corrections** record field, old→new, reason, source (`human/resolver/replay/
  test`) and are **replayable** — captures + the correction log reproduce the
  corrected values (`replay_event_values`).
- **Catalog CSV import** (`core/catalog_io.py`): `sku,name,aliases,variants,
  price,stock`. The resolver runs against the imported catalog.
- **Export** (`core/review.py`): `orders.csv`, `corrections.csv`,
  `captures.csv`, and `receipt.txt`.

## The receipt (denominators kept separate)

seed · sellers · **total messages** vs **order-bearing messages** · proposed ·
accepted · rejected · corrected · needs_info · unicode corruptions · duplicate
events · **exact ledger reproduction before correction** vs **after correction**.

The before/after reproduction is measured against the *synthetic* ground-truth
ledger — a metric on synthetic data, never a claim about a real seller. In a
typical run, human correction lifts exact ledger reproduction materially (e.g.
~0.4 → ~0.8 of sellers), which is the point: the review loop is where ambiguous
captures become a correct ledger.

## What it is not (no category borrowing)

- Not `[2b]` — no phone, no `adb`, no real app was touched.
- Not `[3]` — no seller adoption, revenue, or seller-hour claim.
- Synthetic sellers are not live sellers. See
  [`docs/OBJECTION_MATRIX.md`](OBJECTION_MATRIX.md) for the category discipline;
  this slice sits on `[1]` and does not launder it into a higher category.
