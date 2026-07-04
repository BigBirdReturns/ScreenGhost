# ScreenGhost in Threat Geometry: the level-1 intake layer

Status: alignment doc. Places ScreenGhost in the AXM layering and pins the
boundary object it emits.

The field theory lives in the GhostBox repo:
[`GhostBox/docs/THREAT_GEOMETRY.md`](https://github.com/BigBirdReturns/GhostBox/blob/main/docs/THREAT_GEOMETRY.md).
This doc is the ScreenGhost end of that seam.

## One job: turn a surface into evidence

ScreenGhost was the intake fight. Its whole discipline — inherit the user's own
access, try `api` → `view_tree` → `vision` → `unsupported_surface`, never let a
category borrow trust, keep the denominator honest — exists to answer exactly one
question at the boundary: **what did the surface show, and how faithfully did we
capture it.**

In Threat Geometry terms, ScreenGhost is **layer 1, sensor abstraction, on the
informational substrate.** It converts messy UI, chat, and screen state into a
structured, timestamped, provenanced observation. It is intake, not truth. It
detects nothing about threats; it produces the faithful record that the level-2
attention layer reasons over.

```
  surface (UI / chat / screen)
        │   ScreenGhost: api > view_tree > vision > unsupported_surface
        ▼
  EvidenceEvent  ──►  GhostBox (level 2)  ──►  AttentionFinding  ──►  claim-test
   (this repo)          "where is the tension"       "surfaced claim"    "verdict"
```

## The boundary object: EvidenceEvent

The canonical shape is defined by the GhostBox interop contract
(`ghostbox/interop/contracts.py`). ScreenGhost emits it via
[`core/evidence_event.py`](../core/evidence_event.py):

| Field | Meaning |
|---|---|
| `source` | who observed (`screen_ghost`) |
| `surface` | the capture path used — `api` / `view_tree` / `screen` — carrying the path's trust to the boundary |
| `observation` | the structured state captured (app, screen, elements, topic, confidence, …) |
| `captured_at` | when |
| `raw_ref` | pointer to the underlying capture (screenshot / db row), not the bytes themselves |
| `provenance` | `proven` = **faithful capture**, not that the content is true |
| `event_id` | content-addressed over the identity fields; stable across processes and repos |

### Why a mirror, not an import

ScreenGhost carries no dependency on GhostBox — peers across a contract seam, not
a monolith. So `core/evidence_event.py` reimplements the ID derivation to be
**byte-for-byte identical** to the contract, and
[`tests/test_evidence_event.py`](../tests/test_evidence_event.py) pins that
equality against golden values produced by the GhostBox contract itself. The
same observation, constructed in either repo, yields the same `event_id`, with no
shared code. If the two drift, the test fails; the contract wins and the mirror
is the bug.

### The honesty line, kept

`provenance = proven` asserts the **capture** is faithful — the on-screen bytes
were recorded without corruption. It asserts nothing about whether the observed
content is accurate. A vision-path capture and a view-tree capture of the same
screen are *different surfaces with different trust*, and that difference is
preserved in the `surface` field rather than flattened. This is the same rule
the rest of ScreenGhost already enforces (`docs/OBJECTION_MATRIX.md`,
`docs/SURFACE_CAPABILITY_MATRIX.md`), carried onto the level-1 boundary.

## What is proven here, and what is not

- **Proven:** ScreenGhost emits a well-formed EvidenceEvent whose `event_id`
  agrees with the GhostBox contract byte-for-byte (`make test`, or
  `python -m pytest tests/test_evidence_event.py`).
- **Not claimed:** live, continuous wiring of ScreenGhost's device path into a
  running GhostBox instance. That integration is a frozen `[2b]`-class item in
  ScreenGhost's own ledger — the contract and the emitter are proven; the live
  connection is not. No category borrows trust.

## See also

- Field theory: `GhostBox/docs/THREAT_GEOMETRY.md`
- Layering and roles: `GhostBox/docs/AXM_LEVEL_2_ALIGNMENT.md`,
  `GhostBox/docs/REPO_ROLE_MAP.md`
- Boundary spec: `GhostBox/docs/INTEROP_CONTRACTS.md`
- Audit (incl. this emitter): `GhostBox/docs/AXM_LEVEL_2_AUDIT.md`
