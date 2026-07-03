# Vision-Model Limitations: Where Screen Ghost Fits, and Where It Doesn't

This document is deliberately adversarial about our own approach. The rest of
the repo argues *for* "the UI is the only stable API." This file states, as
precisely as it can, the regimes where that thesis breaks — so we stop selling
a low-volume, single-source *observer* as a hosted, high-volume *messaging
platform*. Those are different problems, and conflating them is the core error
the pitch currently makes.

The numbers below marked *(est.)* are order-of-magnitude reasoning, not
measured benchmarks. Where the repo already concedes a point in code or docs,
it is cited.

---

## 1. Two regimes, one architecture

Screen Ghost is a screenshot → VLM → tap/observe loop with a physically
attached device. Its behavior is completely different in two regimes:

| | **Regime A — Observer** | **Regime B — Hosted messaging** |
|---|---|---|
| Example | PTA/Simbli agenda (`examples/pta_agenda_observer.md`) | Page365-style Live Commerce order intake |
| Sources | one bot-protected page | thousands of live conversations |
| Volume | monthly, human-in-loop | continuous, thousands of msgs/min |
| Language | English | Thai, mixed scripts, emoji |
| Payloads | text lines | stickers, locations, files, reactions |
| Uptime need | best-effort | near-100%, revenue-critical |
| Host | the user's own phone | ??? (see §5 — this is unresolved) |

**Regime A is a genuinely good fit** and the repo should keep making that
claim. **Regime B is where the architecture does not currently work**, and the
sections below say why. `docs/index.html` markets Regime-A mechanics against a
Regime-B problem ("the Page365 bottleneck, solved"). It isn't.

---

## 2. What the VLM cannot reliably read

### 2.1 Complex scripts (Thai, and similar)

The README already concedes this: *"English UIs only — Model trained primarily
on English"* (`README.md`, Limitations). This is not a fine-tune away for a
~1.8B-parameter model:

- Thai has **no inter-word spaces**, so token/word segmentation must be
  inferred visually.
- **Stacked diacritics and tone marks** (multiple glyphs per base character)
  are exactly where small OCR heads degrade.
- Names, item codes, and quantities in an order line are the *high-cost*
  errors — a misread quantity or buyer name is a wrong shipment, not a typo.

For an order-taking workload, "usually right" is a defect: every comment must
be captured exactly once, correctly. A model whose own docs scope it to English
UIs cannot underwrite that.

### 2.2 Non-text payloads are first-class, and pixels lose them

Live Commerce and chat are not text streams. Stickers, shared locations,
attachment files, and reaction emojis carry real meaning and, in a protocol,
arrive as **structured fields**:

| Payload | Protocol/API gives you | A screenshot gives you |
|---|---|---|
| Sticker | sticker/pack ID | a rendered image to re-classify |
| Location | lat/long, place ID | a map thumbnail |
| Attachment | the file + MIME type | a thumbnail you cannot open |
| Reaction | actor + emoji + count | pixels to OCR/count |

A VLM must *re-derive* from pixels what the platform already computed. Every one
of these is a new recognition problem with its own error rate, and some
(opening an attachment's actual bytes) are **not recoverable from a screenshot
at all**. "Add stickers/locations/attachments/reactions to the mix" is not an
edge case to bolt on later; for messaging it is the workload.

### 2.3 Latency and cost per observation

A screenshot→VLM step is seconds, not milliseconds. The README states *"Slow on
CPU: 3–5 seconds per step without GPU."* Even on GPU, a full-screen VLM pass is
orders of magnitude more compute per message than parsing a structured payload
*(est.)*. For a seller receiving thousands of comments per minute during a live
stream, a per-screen VLM loop cannot keep up with arrival rate — the queue
grows without bound. Protocol parsing is effectively free at that volume.

---

## 3. Concurrency and window management

UI automation drives **one visible surface at a time**. A protocol client
multiplexes N conversations over one connection; the message carries its own
conversation ID. Screen Ghost has to *navigate to* each conversation, capture,
read, act, and navigate away.

For a seller with 50–100 concurrent buyers this means either:

- **Serial** — round-robin through chats; end-to-end latency scales with the
  number of open conversations (miss the "CF" while you're three windows away), or
- **Tiled** — many windows on screen at once, which multiplies the VLM's
  reading load per frame and shrinks each target below reliable tap/read size.

There is no configuration of a single-surface driver that matches a protocol
client's native fan-out. This is structural, not a missing feature.

---

## 4. The cost chart measures the wrong axis

`docs/index.html` plots *cumulative engineer-days lost to breaking changes* and
shows Screen Ghost flat (~3 days) while a protocol adapter climbs to ~104. Grant
the premise — UI re-derivation genuinely does absorb vendor churn better than a
brittle wire-format adapter. **That is not Page365's bottleneck.**

Page365's actual constraints are throughput, per-message accuracy, multilingual
extraction, concurrency, and uptime under load — none of which the chart plots.
Winning the maintenance-cost race says nothing about whether the system can read
10,000 Thai comments a minute correctly. The visualization is honest about the
axis it shows and silent about the axes that decide the use case.

---

## 5. The unresolved contradiction: local hands ⊥ hosted scale

This is the decisive one. The repo's safety and anti-blocking thesis is
explicit (`README.md`, "Threat Model: Local-Only Hands"):

> the hands that actually touch the UI stay on a physically attached device …
> each vendor just sees an ordinary user.

That property is **structurally incompatible** with "50k hosted concurrent
users, near-perfect uptime." There are only two places to run the hands, and
both fail the requirement:

**Option A — on each user's own device (preserves the thesis).**
Requires, per seller, an always-on computer (README: *8GB+ RAM*, *~4GB disk for
the model*) plus a tethered phone. The target users are phone-only Live Commerce
sellers who, in the words of the critique, *"don't have a PC, let alone a
personal always-on data center."* The precondition the model depends on is the
one the user base cannot meet.

**Option B — in our datacenter (enables scale).**
Now the hands are remote again: emulator/headless farms behind shared egress
IPs. That is precisely what platform anti-bot systems fingerprint and block —
the same "shared origin IP" failure the local-hands design existed to avoid. And
it reintroduces the remote actuation surface the threat model claims to have
eliminated.

You cannot have both "every action originates from the user's own
device/IP" **and** "we run it as a service in our infrastructure." The
architecture's central safety claim and its central scaling claim cancel each
other. Nothing in the current repo resolves this, and no amount of VLM
improvement does either — it's a topology problem, not a model problem.

---

## 6. What actually survives

Stated plainly, so the defensible claim isn't lost in the retraction:

- **Bot-protected, single-source, low-volume, human-in-the-loop observation**
  (the PTA/Simbli pattern) is a real, honest use of this architecture. Keep it.
- **UI-level re-derivation as churn insurance** for a *narrow* set of intents on
  a device the operator already controls is reasonable.
- **Universal, hosted, real-time, multilingual messaging at 50k-user scale is
  not** something this design reaches, and the gap is architectural (§5), not a
  roadmap item.

## 7. What would have to be true to make the big claim

Not a to-do list — a statement of how large the gap is:

1. A reader that is exact (not "usually right") on Thai and mixed scripts,
   **plus** structured recovery of stickers/locations/attachments/reactions —
   i.e. most of what a protocol already hands you, re-earned from pixels.
2. Real-time throughput at thousands of messages/minute per seller.
3. Native multi-conversation fan-out without a single visible surface as the
   bottleneck.
4. A resolution to §5 that keeps actuation on the user's own network **without**
   requiring hardware the user base does not own.

If all four are solved, the thing you have built is no longer "screenshot the
UI" — it is a protocol client with extra steps. Which is the point the
multi-protocol history actually teaches: the UI is the interface a vendor can't
*revoke*, but it was never the interface you'd choose to run a high-volume
integration on if a protocol was available.
