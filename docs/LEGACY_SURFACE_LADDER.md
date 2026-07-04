# The Legacy Surface Ladder

"The ghost inherits the user's system" does not stop at a phone. Push it to the
floor and you reach the machines the modern stack forgot: a COBOL mainframe, an
ancient ERP, a control panel with no API and — at the very bottom — no digital
output at all. The capture ladder just grows two rungs downward:

```
api  →  view_tree  →  vision  →  physical(robot + webcam)  →  unsupported
built    built        declared   declared, FROZEN            named floor
```

Same thesis, same Candidate contract, same ledger. A surface proven on one rung
is reusable on the next, because the pipeline never learns which source a
candidate came from. `core/surfaces.py` is the source of truth for which rung
each surface routes to; `core/capture.py` lists the ladder in `STRATEGIES`.

## Rung 1 (new): the green screen is a *view_tree*, not a vision problem

The instinct is to point a camera at a green screen. Don't — you'd be throwing
away structure you already have. A 3270/5250 terminal exposes its screen as a
**field buffer**: text with exact grid positions and a protected/unprotected
attribute, carried over the TN3270 datastream **the user's own terminal session
already receives**. Reading it is screen scraping / HLLAPI — decades old, exact,
and lossless. So:

- A terminal field is a `TextNode` with real bounds. The grid *row* is literal,
  so **row grouping is exact** — there is no y-band heuristic and no
  `row_grouping_failure` mode on this surface. It's a stronger surface than a
  fuzzy chat UI, not a weaker one.
- The label→value pairing is deterministic: the protected field to the left of a
  data field is its label. `CUST: NOK RATANA` reads like a chat row —
  `sender=CUST`, `body=NOK RATANA` — with no guessing.
- EBCDIC→Unicode is the exact-bytes discipline, the mainframe analog of keeping
  Thai byte-for-byte on the phone path.

**What is built here:** `core/legacy.py` maps a decoded screen buffer to the
same `Candidate` contract, exercised against
`examples/legacy_fixtures/green_screen_order_panel.json` and demonstrated by
`examples/legacy_ladder.py`. A mainframe-sourced order reaches the identical
OrderBook a LINE-sourced order does.

**What is FROZEN ([2b], not claimed):** a live TN3270 network connection and the
EBCDIC decode against a real mainframe. This is the exact same boundary the `api`
rows hold — a decided strategy + representative payload, not a live session.

## Rung 2 (new): physical actuation — the honest floor

When a machine has no API and no readable buffer — a standalone controller, a
device whose only output is a physical display — the ghost still inherits the
user's *hands and eyes*: a robot presses the real buttons, a webcam reads the
screen. This is the literal end of "anything a keyboard, a mouse, and a fingertip
can do."

It is **declared and FROZEN, not built**, for one honest reason: **the webcam
brings OCR back.** This rung reintroduces exactly the error class the phone and
green-screen paths were built to avoid. So it cannot ship without a **read-back
verification loop** — after every actuation, re-read the screen and checksum it
against the expected state; refuse to proceed on mismatch. Claiming it before
that loop exists would be claiming OCR reliability, which the repo does not.

## Ingesting the original documentation — and the three landmines

The appeal of legacy is that the systems came with manuals. Ingest them — but
know what they are and what they are not.

1. **The docs are a hypothesis, not an oracle.** Thirty-year-old COBOL
   documentation has drifted from actual behavior. The docs are a *prior*; the
   running system is the oracle. Reconcile docs against observed behavior and
   treat each divergence as a **finding**, not a failure.

2. **"Brute-force all the inputs" — never against production.** A real legacy
   system has side effects; you cannot fuzz a live financial mainframe. The
   named techniques are:
   - **characterization testing** — pin current behavior *before* you touch it,
   - **differential testing** — old system vs. new, same inputs, diff the
     outputs,
   - **active automata learning** (Angluin's L\* and kin) — *learn* the state
     machine with far fewer queries than exhaustive enumeration.
   All of them require a **read-only probe or a replica**. The hard rule:
   ScreenGhost learns the black box; it does not hammer prod. Destructive input
   enumeration against a production system is out of scope, by policy.

3. **The webcam rung is OCR — see Rung 2.** Any claim that reaches for pixels
   must carry the read-back loop, or it does not get made.

## What is proven here vs frozen

| Rung | Status | Evidence |
|---|---|---|
| green screen → Candidate contract | **built, tested** | `core/legacy.py`, fixture, `tests/test_legacy.py` |
| green-screen order → same ledger | **built, tested** | `screen_to_order_line` → OrderBook parity test |
| live TN3270 / EBCDIC connection | **frozen [2b]** | not built; decided strategy only |
| physical(robot + webcam) actuation | **frozen [2b]** | declared floor; needs read-back loop |
| learning a black box from docs+probing | **frozen [3]** | method named; requires replica/read-only |

See [`SURFACE_CAPABILITY_MATRIX.md`](SURFACE_CAPABILITY_MATRIX.md) and
[`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md). The rule that made the rest of the
repo strong holds here: never claim a rung you have not at least exercised
against a fixture, and never hide the rung you have not built.
