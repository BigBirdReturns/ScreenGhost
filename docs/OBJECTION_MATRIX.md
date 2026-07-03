# Objection Matrix — closing the solution space

The original critique named Thai OCR, vision latency, stickers/locations/
attachments/reactions, live-commerce burst, 50k concurrency, uptime, IP
blocking, and missing user infrastructure — as if they were one impossibility.
They are not one thing. This document forces each objection to **stand alone**:
each gets a named test category, an executable bench, a receipt, and an explicit
statement of the claim it is *allowed* to make and the claim it is *forbidden*
from making.

The deeper distinction the critique missed is **extraction vs distribution**.
One design harvests centralized state (and inherits shared-origin blocking, PC
requirements, and window multiplexing). The other treats **each seller's own
phone** as the execution boundary — the phone already owns the account, IP, app
session, and screen state — and ScreenGhost's job is to turn that local screen
state into a replayable event ledger **without OCR on the text path**.

This is a systems argument, not a personal one. The critique failed because it
mistook one extraction architecture for the entire design space.

## Two hard rules

1. **Denominator discipline.** No bench reports a rate without saying which:
   total messages, order-bearing messages, UI nodes, parser candidates, emitted
   events, or seller-level ledgers.
2. **No category borrows trust from a higher one.** Synthetic population proof
   `[1]` does not become device proof `[2b-1]`; device proof does not become
   real-app proof `[2b-2]`; real-app proof does not become business proof `[3]`.

## Evidence categories

| id | what it proves | state |
|----|----------------|-------|
| `[1]` | pipeline integrity on a synthetic population, in-process | proven |
| `[2a]` | rendered view-tree parity (UiAutomator XML → group → ledger) | proven |
| `[2b-1]` | real-device fixture parity via `adb shell uiautomator dump` | **frozen** (no device) |
| `[2b-2]` | real-app surface compatibility (benign LINE/FB/Shopee threads) | **frozen** |
| `[3]` | seller outcome (before/after seller-hour) | **frozen**, not claimed |

## The matrix

`core/objections.py` is the machine-readable source of truth; this table mirrors
it. Run `python examples/objection_receipt.py --live` for the live status.

| id | objection | answer | cat | status | allowed claim | forbidden claim |
|----|-----------|--------|-----|--------|---------------|-----------------|
| THAI_TEXT_RELIABILITY | Thai unreliable via OCR | exact Unicode from view tree, no OCR | [2a] | PASS | exact Thai preserved, corruption 0 through tested path | that every real app exposes clean Thai nodes ([2b-2]) |
| VISION_LATENCY | vision slow/token-heavy | no model on text path | [1] | PASS | capture path is model-free, sub-second p95 | that no edge ever needs a model |
| NON_TEXT_PAYLOADS | stickers/locations/slips matter | typed non-text nodes, not order text | [1]/[2a] | PARTIAL | payloads typed from content-desc; slips surfaced as unstructured | that sticker/slip bytes are decoded |
| LIVE_COMMERCE_BURST | burst + 50–100 buyers | finite viewport, cadence, dedupe, keeps_up | [1]/[2a] | PASS | full recall + headroom at busy/hot; loss reported past window | that an arbitrary firehose never drops |
| CENTRAL_IP_BLOCKING | shared IP gets blocked | no central actuation; seller's own device/IP | [2b-1] | ARCH | no shared origin required by design | that every platform ToS permits it |
| USER_INFRASTRUCTURE | sellers have no PC | phone-first, local logs, replay | [2b-1] | ARCH | design needs only the seller's phone | that an on-device build runs today |
| WINDOW_MANAGEMENT | too many windows drown it | one device = one seller surface; finite viewport | [2a] | PASS | scroll replay deduped, window loss reported | that one device multiplexes many sellers |
| PARSER_GENERALIZATION | real language is messier | separate parser failure from capture seam | [1] | PARTIAL | seam delta ~0, failures attributed to parser and reported | that the parser generalizes to real Thai |
| APP_SURFACE_DRIFT | real trees differ | adapter risk unless text is unavailable | [2a]→[2b-2] | PARTIAL | survives font/size/bubble/dark; overlap fails openly | that real trees group like the fixture |
| BUSINESS_OUTCOME | capture ≠ seller value | true; freeze until seller-hour exists | [3] | FROZEN | nothing about revenue/adoption | any seller-hour or revenue claim |

## The two biggest open items (stated, not hidden)

- **PARSER_GENERALIZATION** — the generator shares the parser's grammar, so
  clean recall is high by construction. This proves *pipeline integrity*, not
  that the parser reads real merchant Thai. Highest remaining risk.
- **APP_SURFACE_DRIFT / [2b-1] / [2b-2]** — everything device-side is proven
  only against a *rendered* fixture. Whether a real LINE/FB/Shopee tree matches
  needs hardware this environment does not have.

Both are **adapter tickets or data problems, not architecture problems** —
unless a real surface hides exact text entirely, which is the one finding that
would reopen the architecture. That is precisely what `[2b-2]` must test.
