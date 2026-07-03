# Adapter Conformance v0

Every "but what if the real app layout is different?" becomes an executable
conformance ticket: a fixture, an expected candidate ledger, a named failure
cause, and a reproducible receipt.

**Core rule:** an adapter **never emits order events**. It emits **candidates**.
`OrderBook` emits events. The adapter's only job is to turn a surface snapshot
into candidates that satisfy the contract below.

Not hardware proof, not live-seller proof, not business proof. All hashing is
stable sha256 (never Python's salted builtin `hash()`), so fixtures, expected
ledgers, and receipts reproduce from a cold run.

## The candidate contract

Each candidate carries:

| field | meaning |
|---|---|
| `capture_id` | stable id of this observation |
| `source_app` / `source_surface` | which app / surface produced it |
| `thread_or_screen_id` | conversation/screen id if visible |
| `first_seen_at` / `last_seen_at` | first and most recent snapshot it appeared in |
| `raw_text` | exact text, byte-for-byte |
| `unicode_ok` | raw_text equals its NFC form (no corruption) |
| `sender` | commenter label if visible (may be null) |
| `node_bounds` / `row_bounds` | geometry of the node and its row |
| `candidate_key` / `dedupe_key` | identity for grouping and scroll-dedupe |
| `payload_type` | one of the payload classes below |
| `visibility` | visible / offscreen |
| `parser_eligible` | whether OrderBook should consider it (text/emoji_text) |
| `snapshot_hash` | sha of the source snapshot |

**Payload types:** `text`, `emoji_text`, `sticker`, `image`,
`payment_screenshot`, `location`, `attachment`, `unknown`.

## Conformance checks

For each fixture the verifier reports pass/fail on: **unicode** (no corruption),
**row grouping** (rows separate; overlapping rows fail), **dedupe** (scrolled/
re-shown rows collapse), **payload classification** (produced payload set matches
the golden expected candidates), and **visible-window loss**.

## Single-cause failure taxonomy

Every adapter failure is **exactly one** of:

`no_text_exposed` · `unicode_corruption` · `row_grouping_failure` ·
`dedupe_failure` · `payload_classification_failure` · `visible_window_loss` ·
`unsupported_surface` · `malformed_fixture`.

Multi-cause attribution is itself a failure (`MULTI_CAUSE`, nonzero exit).

## Fixture metadata

Each fixture is `<!--META {json}-->` followed by one or more `<hierarchy>`
snapshots. META declares: `fixture_id`, `surface_type`, `expected_verdict`
(`PASS`/`EXPECTED_FAIL`), `expected_failure_cause`,
`positive_conditions_exercised`, `negative_conditions_exercised`,
`expected_candidates` (the golden ledger), `expected_ledger_hash`, and
`fixture_hash`. A fixture cannot pass without exercising at least one positive
and one negative condition. An `EXPECTED_FAIL` fixture that silently passes is a
hard failure (`UNDECLARED_PASS`).

## Run it

```bash
python examples/adapter_conformance.py --fixture examples/adapter_fixtures/line_like_basic.xml
python examples/adapter_conformance.py --all        # scorecard; exit 1 on any undeclared failure
python examples/adapter_to_ledger_demo.py --fixture examples/adapter_fixtures/line_like_basic.xml
python -m tools.gen_adapter_fixtures                # regenerate the fixtures
```

The two `EXPECTED_FAIL` fixtures demonstrate the classifier:
`pathological_overlap.xml` → `row_grouping_failure`; `missing_text_nodes.xml` →
`no_text_exposed`.

## What this settles

> If an app surface exposes exact text and bounded rows, the adapter must prove
> it by conformance fixture. If it does not expose text, the failure is
> explicit: `no_text_exposed`. The architecture is not disproven by an adapter
> failure; the adapter has work to do — unless the surface withholds text
> entirely.
