# ScreenGhost Interface Procedure v0

**Human interaction is the fallback protocol.** A vendor can revoke APIs, alter
OAuth scopes, cloud-gate access, or drop Matter support — but the product still
has to show a state to a human and let that human press a control. This module
operates that human-facing surface the way a careful human would, with **custody,
procedure, and proof**. Not GUI automation: **auditable human-equivalent
operation** — visible state, bounded action, verified result, sealed trace.

## Doctrine

```
APIs when they are local, open, and reliable.
UI procedures when APIs are captured, cloud-gated, or vendor-controlled.
Human approval for anything sensitive (the procedure IS the approved artifact).
Evidence sealed every time.
```

## Architecture

```
Intent      your rule engine decides what should happen        (not built here)
Operator    ProcedureRunner executes ONE approved procedure    core/interface_procedure.py
Surface     browser (PlaywrightSurfaceDriver) — later: Android emulator, vendor app
Evidence    BEFORE pixels · action trace · AFTER pixels · verification
Custody     genesis seals the whole trace as one axm-hybrid1 shard
Review      GhostBox observes downstream (drift / repeated failure)  (later layer)
```

## The procedure is the unit of approval

```
Procedure proc-lamp-on-v0:
  anchor    #dashboard-title == "Home Dashboard"     the surface must be the known one
  target    #tile-living-room-lamp                    must exist,
            label contains "Living Room Lamp"         must be what was approved,
            inside ApprovedBounds(20,100,300,160)     must not have moved
  action    ONE click, at the target's center, inside approved bounds only
  verify    #lamp-state == "On" within timeout        the visible state must confirm
```

An unbounded procedure (empty anchor, empty verification, …) is **refused before
the driver is touched**.

## Drift is a first-class outcome

If the tile moves, the app updates, the label changes, or the verification text
never appears, the runner **does not improvise**. It stops at the failed check,
captures the drift state, and the trace names the reason:

| Drift | When | Click performed? |
|---|---|---|
| `anchor_missing` | the known surface anchor isn't rendered | no — stops before anything |
| `target_missing` | the approved control doesn't exist | no |
| `target_label_mismatch` | the control isn't what was approved | no |
| `target_outside_bounds` | the control moved off-approval | **no — the click is never re-aimed** |
| `verification_failed` | clicked, visible state never confirmed | yes — and that state is the evidence |

The non-improvisation property is itself tested: after any drift, the driver call
log shows no further actions. **That is the difference between an operator and
malware.** A changed UI is a *visible* failure that asks for a new approved
procedure — a much better failure mode than a silent ecosystem dependency.

## Evidence tier — explicit

The sealed trace is `interface_procedure_trace`: **rendered pixels + one bounded
action + visible verification only** — not vendor backend state, not API truth,
not platform truth, not legal-grade provenance by itself. Each screenshot rides
with its unchanged **Pixel Evidence v0** manifest (`pixel_capture` tier, bytes
hashed and sealed verbatim, never rewritten). Verification honesty: the state
check reads rendered on-screen text at an approved locator (the view-tree rung).
**No OCR, no image model, no pixel interpretation.**

## Run it

```bash
python examples/interface_procedure_demo.py            # completed: Off -> On, sealed
python examples/interface_procedure_demo.py --drift    # dead tile: drift, sealed anyway
python -m pytest tests/test_interface_procedure.py -q
```

## Live receipts (this environment)

| Check | Result |
|---|---|
| Real Chromium renders the local fixture; anchor→before→bounds→click→after→verify | **COMPLETED** (Off → On) |
| Sealed trace | **PASS** (`sh1_a35b4cdc…`, out-of-band key) |
| Detached verify (no browser / ScreenGhost / GhostBox) | **PASS** |
| Dead tile | **drift `verification_failed`**, clicked once, after-state sealed (`sh1_bced4df3…`) |
| Moved bounds | **drift `target_outside_bounds`**, click never performed (state still "Off") |
| Anchor / target / label drift | **stops without clicking** |
| No improvisation after drift | **proven** (driver call log) |
| Unbounded procedure | **refused before the driver is touched** |
| Wrong key / missing key | **FAIL / NO_TRUSTED_KEY** |
| PNG bytes verbatim in the sealed shard | **yes** |
| No ghostbox / OCR / vision import; playwright lazy | **yes** (subprocess-isolated) |
| Test suite | **19/19** |

**Evidence tier of this slice:** bounded-operation-with-sealed-trace proven
against a **deterministic local fixture surface** (real Chromium, `file://`, zero
network, zero vendor) — labeled `local-fixture-surface`, NOT Google Home /
SmartThings / Roku. The same runner + procedure shape targets those surfaces
later; each needs its own approved procedure and its own live receipts. Crypto
backend remains the pure-Python `dilithium-py` fallback — functional, not
load-proven.

## Control question

Can ScreenGhost execute one approved, bounded interface procedure against a
rendered surface — observe the state, operate the allowed control, verify the
visible result — and seal the whole trace so it verifies after the browser,
ScreenGhost, and GhostBox are removed, while any deviation stops the run and is
itself sealed as drift evidence?

**v0 answer: yes** — against the local fixture surface, with the non-improvisation
property proven, both outcomes (completed and drift) sealed through genesis.
