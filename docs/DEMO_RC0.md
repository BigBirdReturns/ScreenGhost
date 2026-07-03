# ScreenGhost Demo RC0

A local release candidate you can clone cold, install, run, inspect, export, and
replay — without anyone narrating it.

**What this proves:** a reviewable order-ledger product slice works end to end on
synthetic data — capture → propose → review/correct → export → replay, with an
append-only audit trail and a reproducible receipt.

**What this does NOT prove:** no hardware proof, no business proof, no
live-seller proof. Synthetic sellers are not live sellers.

## Install

```bash
git clone <repo> && cd ScreenGhost
python -m tools.setup_demo      # or: make setup
```

Setup creates runtime dirs, initializes a demo store, and runs a fast smoke
test. It prints the next command.

## Run the demo

```bash
python examples/operator_demo.py --seed "op" --sellers 10     # or: make demo
```

It prints every path: **store**, exports (`orders.csv`, `corrections.csv`,
`captures.csv`, `receipt.txt`), the full **receipt**, and the **replay command**.

## Open the review UI

```bash
python examples/operator_demo.py --seed "op" --sellers 10 --serve   # or: make serve
```

Localhost page: events grouped by status (proposed / proposed_incomplete /
needs_info / accepted / corrected / rejected / cancelled / fulfilled), the raw
capture beside the parsed fields (with confidence + parser path), correction and
transition history, per-event controls, and an export button.

## Correct an event

In the UI: pick an event → set field (buyer / SKU / quantity / variant) → value →
reason → **apply**. Every action writes an **append-only** record (a correction
or a transition). Raw captures are never mutated.

## Export & replay

Export writes the CSVs + receipt. Then:

```bash
python examples/replay_ledger.py --store log/operator_demo/ledger.db   # or: make replay
```

Replay re-derives each ledger from captures + corrections alone and reports
`replay matched: True/False` per seller (exit 0 if all match).

## Inspect the receipt / reproduce it

The receipt keeps denominators separate (total vs order-bearing messages) and
records `schema_version`, `replay_matched`, and `proof_claims_forbidden`.

```bash
python examples/verify_demo_receipt.py --receipt examples/receipts/operator_demo_seed_op.txt
```

This re-runs the same seed and confirms every denominator, export, and replay
reproduces (wall-clock time is not compared). The population is deterministic
across processes, so `MATCH` is the expected result. `make verify` runs it.

## Schema & durability

The store records `schema_version`. Opening a store whose schema is **newer**
than this build supports fails closed with a clear `SchemaError` rather than
guessing. Raw captures survive reopen unchanged.

## Artifact hygiene

Runtime output goes under `log/` and `artifacts/` (git-ignored). Only the
canonical fixture, canonical receipts, docs, and tests are committed.
