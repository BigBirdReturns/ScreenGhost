# Changelog

All notable changes to ScreenGhost. This project keeps every claim executable;
each entry below is backed by tests and/or a reproducible receipt.

## [0.1.0] — Evidence Kernel

The release where the project crosses from argument into an operating kernel.
Every original objection is now a measured category, a receipt, or a frozen
claim. Not hardware proof, not live-seller proof, not business proof.

### Added

- **Exact-text extraction** — read the OS view tree as text, no OCR on the text
  path (`core/texttree.py`); Thai preserved byte-for-byte.
- **Rate/keep-up bench** — order-event recall + honest firehose boundary
  (`core/ingest.py`, `examples/keepup_bench.py`).
- **Synthetic seller population harness** — 1,000 labeled sellers, 10 cohorts,
  adversarial suites, ledger reproduction (`core/population.py`,
  `core/eval_population.py`, `examples/population_bench.py`).
- **Rendered view-tree parity `[2a]`** — same worlds rendered to UiAutomator XML
  and recovered through the real extraction code; drift + boundary
  (`core/android_fixture.py`, `examples/parity_bench.py`).
- **Objection matrix + receipt** — every objection with allowed/forbidden claim
  and live-refinable status (`core/objections.py`,
  `docs/OBJECTION_MATRIX.md`, `examples/objection_receipt.py`).
- **Pop-proof demo** — operator-seeded adversarial proof theater with
  single-cause failure attribution and self-aborting guards
  (`examples/pop_proof_demo.py`).
- **Reviewable Order Ledger** — SQLite store, review states, append-only
  corrections + transitions, catalog CSV import, export, replay
  (`core/ledger_store.py`, `core/review.py`, `core/review_server.py`).
- **Operator Demo Pack + product resolver** — one-command demo, incomplete-order
  handling, deterministic fixture (`examples/operator_demo.py`,
  `core/resolver.py`, `core/fixtures.py`, `docs/OPERATOR_DEMO.md`).
- **Demo RC0** — one-command setup, reproducibility verifier, schema versioning
  (fail-closed on newer schema), UI polish, artifact hygiene
  (`tools/setup_demo.py`, `examples/verify_demo_receipt.py`, `Makefile`,
  `docs/DEMO_RC0.md`).
- **Adapter Conformance Pack** — candidate contract, 13 fixtures, verifier +
  scorecard, single-cause failure taxonomy, candidate→ledger replay
  (`core/adapter.py`, `examples/adapter_conformance.py`,
  `examples/adapter_to_ledger_demo.py`, `docs/ADAPTER_CONFORMANCE.md`).
- **Claim boundaries** consolidated (`docs/CLAIM_BOUNDARIES.md`); release
  verifier (`examples/verify_release.py`).

### Fixed

- **Determinism**: `population.generate_stream` seeded its RNG from Python's
  salted builtin `hash()`, so `build_population` was not reproducible across
  processes and committed receipts never actually reproduced. Now stable
  sha256. All receipts regenerated; `verify_demo_receipt` returns MATCH.

### Frozen (not claimed)

- `[2b-1]` real-device hardware, `[2b-2]` real-app surfaces, `[3]` business /
  seller-hour outcome. See `docs/CLAIM_BOUNDARIES.md`.
