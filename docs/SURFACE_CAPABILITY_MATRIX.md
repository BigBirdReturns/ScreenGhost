# Surface Capability Matrix

Solve the goal — a seller's chat flow becomes a reviewable order ledger — not
one platform's solution to it. The goal is fixed; the architecture is free. So
each surface gets the capture path that **doesn't have the weakness**, in
preference order.

## The principle: the ghost inherits the user's system

ScreenGhost runs on the user's own device and account. Its capture options are
therefore exactly the user's own access, tried in order:

1. **api** — official platform events the user's authenticated session already
   receives (LINE OA / Facebook Page / Instagram). On this path the app's UI
   obfuscation is *irrelevant* — you never touch the app.
2. **view_tree** — exact on-screen text the user can read (UiAutomator), for
   surfaces with no API.
3. **vision** — last resort, genuine pixels only (never the text path).
4. **physical** — a robot presses real buttons and a webcam reads the screen,
   for a machine with no digital output. Declared and *frozen* (see the legacy
   ladder), because the webcam reintroduces OCR error.
5. **none** — no path at all → `unsupported_surface`, named.

The ladder reaches past phones all the way to the mainframe and the machine with
no digital output — see [`LEGACY_SURFACE_LADDER.md`](LEGACY_SURFACE_LADDER.md).

Every path emits the **same candidate contract**, so the ledger pipeline never
knows which source a candidate came from. That is why a strategy proven on one
surface is reusable on the next.

## The matrix

`core/surfaces.py` is the source of truth; this table mirrors it.

| surface | strategies (in order) | proof | note |
|---|---|---|---|
| LINE Official Account | api → view_tree | event-schema | Messaging API webhook; view-tree only for in-app gaps. Thai exact via API. |
| Facebook Page / Messenger | api → view_tree | event-schema | Page webhook — the app's obfuscation + stripped accessibility are **bypassed, not fought**. |
| Instagram Direct | api | event-schema | Messaging API via the linked Page. |
| Shopee / Lazada / TikTok Shop | api → view_tree | event-schema | Open-platform order/chat APIs where enrolled. |
| Web storefront / WebView | view_tree | fixture | Exact text from the DOM/view tree; no API needed. |
| Messenger app, accessibility stripped | none | gap | No API path AND no readable text → `unsupported_surface`. Route to the Page API; if the seller has no Page, this is an honest gap, not a refutation. |
| Mainframe green screen (3270/5250) | view_tree | fixture | The terminal buffer is a **structured field source** (TN3270/HLLAPI), not pixels — exact text with grid positions the user's session already receives. Grid rows make grouping exact. Live connection + EBCDIC decode is frozen. See [`LEGACY_SURFACE_LADDER.md`](LEGACY_SURFACE_LADDER.md). |
| No digital output (robot + webcam) | physical → none | gap | The ladder's floor: robot presses buttons, webcam reads the screen. **Frozen, unbuilt** — the webcam is OCR and needs a read-back verification loop before any action is trusted. |

## The Messenger objection, answered by a decision

"Try it on Messenger — they obfuscate every UI element and disable accessibility
specifically for you." True, on the *scrape* path — and the repo proves it:
`examples/adapter_conformance.py --fixture messenger_app_obfuscated.xml` →
`EXPECTED_FAIL (no_text_exposed)`. But the scrape path is the wrong decision. A
seller with a Facebook Page receives messages via the **Messenger Platform
webhook** — the ghost, running as the seller, inherits that access. The
obfuscation is bypassed because you never open the app. The objection routes to
`api`; the scrape failure is named, not hidden.

## What is proven here vs frozen

- **Proven (tested):** the routing (`api` → `view_tree` → `vision` → `none`),
  the single candidate contract across sources, and the named
  `unsupported_surface` verdict.
- **Frozen ([2b]/[3], not claimed):** live platform API integration (real
  LINE/Meta/marketplace connections), a live TN3270 connection + EBCDIC decode,
  the physical(robot+webcam) rung, and the fleet economics of managed devices.
  `api` and green-screen rows are a *decided strategy + contract*, exercised
  against representative payloads/fixtures — not a live connection.

See [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md) and
[`ADAPTER_CONFORMANCE.md`](ADAPTER_CONFORMANCE.md).
