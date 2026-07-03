# ScreenGhost v0.1 — Evidence Kernel

The architecture is no longer waiting on belief. Pipeline behavior, synthetic
seller worlds, rendered view-tree parity, the reviewable ledger workflow,
reproducibility, and adapter conformance are all tested. Failures are named.
Frozen claims stay frozen. A remaining objection must now arrive as a failing
fixture, a resolver case, or a product issue.

## What runs

```bash
python -m tools.setup_demo                         # dirs, deps, init DB, smoke
python examples/operator_demo.py --seed op --sellers 10   # the product demo
python examples/operator_demo.py --seed op --serve        # local review UI
python examples/adapter_conformance.py --all              # adapter scorecard
python examples/verify_release.py                         # one release receipt
make test                                                 # full suite (132)
```

## What is proven

- Exact Unicode (Thai) preserved end-to-end — no OCR on the text path.
- Order-event recall/precision under a broad labeled synthetic population.
- Rendered view-tree recovery equals the in-process pipeline (`[2a]`).
- A reviewable order ledger: propose → correct (append-only) → export → replay.
- Receipts reproduce from a cold run (deterministic, stable-hashed).
- App-surface differences resolve to named adapter causes.

## What is NOT claimed

No hardware proof, no live-seller proof, no business-outcome proof, no universal
real-app compatibility, no claim the parser generalizes to all merchant Thai.
Full list: [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md).

## How to reproduce receipts

```bash
python examples/verify_demo_receipt.py --receipt examples/receipts/operator_demo_seed_op.txt
python examples/pop_proof_demo.py --seed "pop-picks-this-live" --sellers 1000
python examples/population_bench.py 300
python examples/parity_bench.py 100
```

Same seed → same numbers, in any process.

## How to inspect failure attribution

- Population/adversarial: `python examples/pop_proof_demo.py --seed <x> --sellers 100`
  (Act IV single-cause attribution).
- Adapter surfaces: `python examples/adapter_conformance.py --all` — every
  failure is exactly one named cause; `missing_text_nodes.xml → no_text_exposed`
  and `pathological_overlap.xml → row_grouping_failure` demonstrate the classifier.

## Next work (normal product triage, not the impossibility claim)

Adapter fixtures, resolver improvements, review-UI polish, export formats,
packaging, and eventually platform-specific policy. File issues with the
templates in `.github/ISSUE_TEMPLATE/`.
