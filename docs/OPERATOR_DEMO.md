# Operator Demo — run the ledger without a developer

A someone-who-didn't-build-it walkthrough of the Reviewable Order Ledger. It
generates a small seller world, runs the pipeline into a local store, reviews
it, exports the ledger, and lets you replay it — all from one command.

**This is a product slice, not proof.** It is *not* hardware proof (`[2b]`) and
*not* business proof (`[3]`). No phone, no live app, no revenue claim. Synthetic
sellers are not live sellers.

## Run it

```bash
# fresh world from your seed
python examples/operator_demo.py --seed "op" --sellers 10

# the fixed demo fixture (repeatable scenarios)
python examples/operator_demo.py --fixture examples/fixtures/seller_world_demo_seed.json

# open the local review UI
python examples/operator_demo.py --seed "op" --sellers 10 --serve
```

## What it does

1. Builds (or loads) sellers, each with a catalog, buyers, and messages.
2. Runs the pipeline: each comment becomes a **proposed** order event (or
   **proposed_incomplete** if the SKU needs a variant the buyer didn't give).
3. Auto-reviews: confident events → **accepted**; the rest → **needs_info**;
   incomplete orders **never** auto-accept.
4. A simulated reviewer resolves needs_info items (accept / correct / reject).
5. Exports `orders.csv`, `corrections.csv`, `captures.csv`, `receipt.txt`.

## Where exports appear

Under `--out` (default `log/operator_demo/`): the CSVs, `receipt.txt`, and the
SQLite `ledger.db`. A copy of the receipt is also written to
`examples/receipts/operator_demo_seed_<slug>.txt`.

## The review UI (`--serve`)

A localhost page: an inbox of events, the raw capture + parsed event + catalog
panels, a correction form (buyer / SKU / quantity / variant), and per-event
controls: **accept, reject, correct, mark needs_info, mark cancelled, mark
fulfilled, export**. Every action writes an append-only record:

- a **correction** row (field, old→new, reason, source), or
- a **transition** row (from-status → to-status, source).

Raw captures are never mutated — corrections and transitions only add facts.

## Inspect corrections

```bash
sqlite3 log/operator_demo/ledger.db \
  "SELECT event_id, field, old_value, new_value, source FROM corrections"
sqlite3 log/operator_demo/ledger.db \
  "SELECT event_id, from_status, to_status, source FROM transitions"
```

Or read `corrections.csv` in the export folder.

## Replay the ledger

```bash
python examples/replay_ledger.py --store log/operator_demo/ledger.db
```

It re-derives each seller's ledger from the raw captures + the correction log
(not from the live event values) and reports `replay matched snapshot: True/
False` per seller. Exit 0 if all match. This is the durability guarantee: the
ledger reproduces from the append-only record alone.

## The receipt (denominators kept separate)

seed · sellers · **total messages** vs **order-bearing messages** · proposed ·
proposed_incomplete · accepted · corrected · rejected · needs_info · cancelled ·
fulfilled · unicode corruptions · duplicate events · export paths ·
**ledger reproduction before vs after correction** · **replay matched**.

The reproduction numbers are measured against the *synthetic* ground-truth
ledger — a metric on synthetic data, never a business claim.
