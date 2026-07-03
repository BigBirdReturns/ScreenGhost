# Claim Boundaries

Every forbidden claim in one place. If a document, receipt, or demo appears to
assert one of these, it is a bug — file a *documentation claim-boundary issue*.

## What ScreenGhost v0.1 does NOT claim

1. **No hardware proof.** Nothing here has run on a physical device via
   `adb`/`uiautomator`. Category `[2b-1]` is frozen. The view-tree work is
   proven only against *rendered* UiAutomator XML (`[2a]`).
2. **No live-seller proof.** No real seller, account, or app session was
   touched. Synthetic sellers are not live sellers.
3. **No business-outcome proof.** No revenue, adoption, or seller-hour lift is
   claimed. Category `[3]` is frozen.
4. **No universal real-app compatibility claim.** ScreenGhost does not assert
   that any given app surface exposes exact text. It asserts a *contract*: if a
   surface exposes exact text and bounded rows, its adapter must pass
   conformance; if it does not, the failure is the explicit, named
   `no_text_exposed`.
5. **No claim that the parser generalizes to all merchant Thai.** The synthetic
   generator shares a grammar with the parser, so clean-cohort recall is high by
   construction. That proves *pipeline integrity*, not real-language coverage.
   Adversarial suites are where real failures are exposed.

## What IS proven (and where)

| claim | evidence | category |
|---|---|---|
| exact Unicode preserved end-to-end (Thai) | population + parity + adapter benches | [1]/[2a] |
| order-event recall/precision under labeled load | population harness | [1] |
| view-tree recovery == in-process (rendered) | parity bench | [2a] |
| reviewable ledger: propose→correct→export→replay | operator demo, replay | product |
| receipts reproduce from a cold run | `verify_demo_receipt`, determinism test | product |
| app-surface differences are named adapter causes | adapter conformance | [2a] |

See [`OBJECTION_MATRIX.md`](OBJECTION_MATRIX.md) for the full per-objection
allowed/forbidden breakdown, and [`ADAPTER_CONFORMANCE.md`](ADAPTER_CONFORMANCE.md)
for the surface contract.
