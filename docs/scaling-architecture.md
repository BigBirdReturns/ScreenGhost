# Scaling Architecture: Local-First Capture for High-Volume Messaging

This is the answer to the honest question raised in
[`vision-model-limitations.md`](vision-model-limitations.md): *can a
Screen-Ghost-lineage system serve high-volume, multilingual, hosted messaging
(Page365-style Live Commerce) at scale?*

It can — but not by defending "screenshot → VLM → OCR." That mechanism was
always the weakest member of the repertoire, not the whole of it. The actuation
layer can do anything a human hand can: **select, copy, paste, tap into an
element, read the clipboard, read the OS view tree.** Once you use the full
repertoire, the two hard axes — *reading fidelity* and *scale topology* —
separate cleanly and each has a real solution.

You need **both** axes won. Winning one is not winning the bet.

---

## Axis 1 — Reading fidelity: a layered extraction stack, VLM last

Stop treating the screen as an image to OCR. Read it the way the OS already
holds it. Three layers, cheapest-and-most-exact first; the VLM is the fallback,
not the front door.

### Layer 1 — Structured view tree (default)

Android `AccessibilityService` / `NotificationListenerService` expose the live
UI as nodes: the **real Unicode text**, sender, timestamp, and
content-descriptions for non-text elements. No OCR, no model.

- **Thai and mixed scripts: solved by construction.** The text was never a
  picture, so there is nothing to misread. This retires the single biggest
  objection outright — not "a better model on Thai," *no model on the text path*.
- Near-zero compute per message, so it keeps up with arrival rate.

### Layer 2 — Clipboard / copy for exact selectable text

Where the tree is thin, use the hands: long-press → *Select all* → *Copy* →
read the clipboard. You get the exact characters a human would copy.

- Perfect for **static or settled screens** (an agenda, a resolved thread).
- Caveat, stated plainly: the copy gesture is **serial and per-selection**, and
  Android 10+ restricts background clipboard reads to the focused IME/app — so
  this layer needs an IME or accessibility component to pull the clipboard back,
  and it does **not** keep up as a per-message loop on a fast stream. Use it for
  fidelity on bounded selections, not as the firehose reader.

### Layer 3 — VLM, only for genuinely-pixel content

Reserve the vision model for what is *actually* pixels and has no text behind
it: a sticker's identity, a shared map thumbnail, an image attachment's gist.
This is a small, bounded fraction of messages, run on demand — not the hot path.

**Residual, honestly:** non-text payloads (stickers, locations, attachment
*bytes*, reaction tallies) still require tapping into the element or a Layer-3
pass. Layers 1–2 do not make those free. They are a minority of volume, but they
are not zero, and the spec that pretends they are is the spec that loses.

---

## Axis 2 — Scale topology: local hands, cloud aggregates

The contradiction in `vision-model-limitations.md` §5 — "local-only hands" vs
"50k hosted users" — dissolves once you notice you were never forced to run the
hands centrally.

```
  Per seller (device they ALREADY own + keep always-on):
     ┌───────────────────────────────────────────────┐
     │  Real LINE / Facebook / Shopee app             │
     │        │  (real account, real device, real IP) │
     │        ▼                                        │
     │  On-device capture agent                        │
     │   Layer 1 view tree → Layer 2 copy → Layer 3 VLM│
     │        │  emits STRUCTURED events               │
     └────────┼───────────────────────────────────────┘
              ▼  (already-structured JSON, tiny)
  Cloud (ordinary stateless SaaS backend):
     aggregate · dedupe · order-match · dashboard · 50k sessions
```

- **The hands stay on the seller's own phone** — the device is already
  always-on and already running the app. No PC per seller. No 4GB model on a
  laptop they don't own. The precondition the critique correctly said users
  can't meet (a personal always-on computer) is **removed**, not assumed.
- **The cloud only moves already-structured events.** That is a normal
  stateless backend; 50k concurrent sessions is unremarkable for one.
- **IP / anti-bot is defeated by construction, not by evasion.** Every action
  originates from the seller's real device, real account, real IP. There is *no
  shared origin* to fingerprint because there is *no central actuation*. The
  strongest point against a datacenter phone-farm becomes the reason this design
  is safe. It is conceded ground, taken.

---

## The falsifiable claim, stated so it can be tested

> A local-first agent — on-device structured capture (view tree → clipboard →
> VLM-fallback) with actuation on each seller's own always-on phone, and a cloud
> tier that only aggregates already-structured events — can serve 50k concurrent
> phone-only Live Commerce sellers, extract Thai and mixed-script messages
> exactly, and keep up with live comment arrival rate, without a datacenter
> device farm and without a shared origin IP.

That is defensible and buildable. State it, and you are arguing from a spec.

## The two risks that can still kill it — carried in the open

A spec that hides its failure modes is a burn, not an argument. These are real
and independent of how clever the eyes are:

1. **OS-policy fragility.** Google restricts `AccessibilityService` for
   non-accessibility apps and has purged Play for exactly this pattern. Expect
   off-Play distribution (sideload / MDM) for the seller agent. This trades
   *vendor-protocol* fragility for *OS-policy* fragility — better odds, not
   immunity.
2. **Platform ToS / account bans.** Automating a platform's client can violate
   its terms; the block vector moves from *IP* to *account*. This is the risk
   that actually decides the bet, and it is not solved by any capture mechanism.
   It is a product/legal posture question (seller-consented, seller-operated
   tooling on the seller's own account), not an engineering one.

---

## What the multi-protocol history actually says

Trillian lost because it re-implemented *someone else's revocable wire format*
from the outside. This design never touches the wire format. It reads the OS's
own view of the UI on the user's own device and acts as that user. The vendor
cannot revoke the interface their paying customer uses, and cannot single out a
farm that does not exist. That is the durable position — earned with a real
extraction stack and a real topology, not asserted.
