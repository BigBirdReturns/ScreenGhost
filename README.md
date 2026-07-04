# Screen Ghost v0.2

**Autonomous UI control + state observation. No cloud. No API. No permission.**

Screen Ghost watches any screen, understands what it sees, and either:
- **Navigator mode**: Taps to accomplish goals
- **Observer mode**: Extracts state for downstream systems

The UI is the API.

## What's New in v0.2

- **Observer mode**: Extract current screen state as structured JSON
- **Continuous watching**: Monitor screen changes over time
- **Text input**: Type text, not just tap
- **Swipe gestures**: Scroll up/down/left/right
- **State diffing**: Detect what changed between observations
- **GhostBox integration**: Events ready for tension detection

## Quick Start

```bash
# Navigate (accomplish a goal)
python screenghost.py --goal "Open Settings and turn on Dark Mode"

# Observe (extract current state)
python screenghost.py --observe

# Watch (continuous observation)
python screenghost.py --watch Settings --interval 5
```

## Observer Mode

Observer mode extracts structured state from any screen:

```bash
$ python screenghost.py --observe
```

Output:
```json
{
  "topic": "Settings.Display",
  "source": "screen_ghost",
  "source_type": "photonic",
  "data": {
    "app": "Settings",
    "screen": "Display",
    "elements": [
      {"type": "toggle", "label": "Dark Mode", "value": "off"},
      {"type": "slider", "label": "Brightness", "value": "70%"},
      {"type": "toggle", "label": "Auto-brightness", "value": "on"}
    ]
  },
  "confidence": 0.8,
  "timestamp": "2025-12-10T03:45:00Z"
}
```

### Continuous Observation

Watch for changes over time:

```bash
# Observe every 5 seconds
python screenghost.py --observe --continuous --interval 5

# Only watch when in Settings app
python screenghost.py --watch Settings --interval 2

# Save screenshots too
python screenghost.py --observe --continuous --save-screenshots
```

### Output Formats

```bash
# JSON (default, for piping to other tools)
python screenghost.py --observe --format json

# Human-readable text
python screenghost.py --observe --format text
```

## Navigator Mode

Navigator mode accomplishes goals through UI automation:

```bash
$ python screenghost.py --goal "Open Settings and turn on Dark Mode"

[step 0] Capturing screen...
[step 0] → TAP {'x': 540, 'y': 1847}: Settings app icon
[step 0] ✓ YES - settings app is now opening
[step 1] Capturing screen...
[step 1] → TAP {'x': 340, 'y': 520}: Display option
[step 1] ✓ YES - display settings visible
[step 2] Capturing screen...
[step 2] → TAP {'x': 892, 'y': 340}: Dark mode toggle
[step 2] ✓ YES - dark mode enabled

Run 1 finished. Completed: True. Steps: 3
```

### New Actions in v0.2

```bash
# Text input (for search, forms, etc.)
python screenghost.py --goal "Open Chrome and search for weather"

# Scrolling (for long lists)
python screenghost.py --goal "Scroll down and find Privacy settings"
```

## GhostBox Integration

Screen Ghost's observer mode produces events that GhostBox can ingest:

```python
from screenghost import observe

# Get current screen state
state = observe()

# Convert to GhostBox event
event = state.to_event()

# Feed to GhostBox
engine.ingest_event(event)
```

### Photonic Source

```python
# In GhostBox
from ghostbox.sources import ScreenGhostSource

source = ScreenGhostSource(device="XXXXXX")

# Single observation
event = source.capture()

# Continuous monitoring
for event in source.watch(interval=5):
    engine.ingest_event(event)
    tension = engine.compute_tension(event["topic"])
```

## State Diffing

Track what changed between observations:

```python
from screenghost import observe
import time

before = observe()
time.sleep(5)
after = observe()

diff = before.diff(after)
print(diff)
# {
#   "app_changed": False,
#   "screen_changed": False,
#   "element_changes": [
#     {"type": "changed", "label": "Battery", "old_value": "85%", "new_value": "84%"}
#   ],
#   "has_changes": True
# }
```

## The Bigger Picture

```
┌─────────────────────────────────────────────────────────────┐
│                        GHOSTBOX                             │
│                   Tension + Attention                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │  SCREEN GHOST   │         │    AXIOM-KG     │          │
│   │  (Photonic)     │         │   (Structured)  │          │
│   │                 │         │                 │          │
│   │  Screenshot     │         │  RSS Adapter    │          │
│   │  → VLM          │         │  XBRL Adapter   │          │
│   │  → State        │         │  iCal Adapter   │          │
│   │                 │         │                 │          │
│   └────────┬────────┘         └────────┬────────┘          │
│            │                           │                    │
│            └───────────┬───────────────┘                    │
│                        ▼                                    │
│              ┌─────────────────┐                            │
│              │  Semantic Space │                            │
│              │  (Coordinates)  │                            │
│              └────────┬────────┘                            │
│                       ▼                                     │
│              ┌─────────────────┐                            │
│              │ Tension Engine  │                            │
│              └─────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

**Screen Ghost** turns pixels into structured events.
**Axiom-KG** turns structured data into semantic coordinates.
**GhostBox** detects tension across both.

The screen is the universal API. Any app. Any device. No permission needed.

## Requirements

**Hardware:**
- Any Android phone (old ones work fine)
- Computer with 8GB+ RAM
- USB cable

**Software:**
- Python 3.10+
- ADB (Android Debug Bridge)
- ~4GB disk space for vision model

## Setup

### 1. Enable USB debugging

1. Settings → About Phone → Tap "Build number" 7 times
2. Settings → Developer Options → Enable "USB debugging"
3. Connect phone and tap "Allow"

### 2. Install ADB

```bash
# Ubuntu/Debian
sudo apt install android-tools-adb

# macOS
brew install android-platform-tools
```

### 3. Install Screen Ghost

```bash
git clone https://github.com/BigBirdReturns/screenghost
cd screenghost
pip install -r requirements.txt
```

### 4. Run

```bash
# First run downloads model (~3.6GB)
python screenghost.py --observe
```

## Offline Usage

```bash
# Download model once
python screenghost.py --download-model --model-path ./models/moondream2

# Run offline forever
python screenghost.py --model-path ./models/moondream2 --observe
```

## Logs

Everything is saved to `log/`:

```
log/
├── screenghost.db          # SQLite database
└── screenshots/            # Captured images
```

Query history:
```bash
# All runs
sqlite3 log/screenghost.db "SELECT * FROM runs"

# All observations
sqlite3 log/screenghost.db "SELECT * FROM observations"

# Steps for a specific run
sqlite3 log/screenghost.db "SELECT * FROM steps WHERE run_id = 1"
```

## Roadmap

- [x] v0.1: Basic navigation (tap + verify)
- [x] v0.2: Observer mode + text input + swipe
- [x] v0.3: Semantic skills foundation (label-anchored waypoints, staleness demotion)
- [ ] v0.3: Skill learning (auto-record skills from successful VLM runs)
- [ ] v0.3: Policy engine (block dangerous actions)
- [ ] v0.4: Web UI for monitoring
- [ ] v1.0: HDMI capture for non-Android devices
- [ ] v1.0: Desktop/browser automation

## Limitations

- **Model accuracy varies**: Moondream2 is small and fast but not perfect
- **English UIs only**: Model trained primarily on English
- **Slow on CPU**: 3-5 seconds per step without GPU
- **No safety rails**: No policy engine yet—don't point it at anything dangerous

These are the small-print caveats. The **architectural** limits — where the
"UI is the API" thesis holds and where it breaks (complex scripts like Thai,
non-text payloads, real-time concurrency, and the local-hands-vs-hosted-scale
contradiction) — are documented honestly in
[`docs/vision-model-limitations.md`](docs/vision-model-limitations.md). Read it
before assuming this scales to high-volume, multilingual, hosted messaging: it
does not, and the gap is structural, not a roadmap item.

## The objection matrix — closing the solution space

ScreenGhost does not ask a central scraper to win a cat-and-mouse war against
every platform. It treats **each seller's own phone as the execution boundary**.
The phone already owns the account, IP, app session, and screen state.
ScreenGhost's job is to turn that local screen state into a replayable event
ledger **without OCR on the text path**. That is the distinction the usual
critique misses: **extraction vs distribution**. One design harvests centralized
state and inherits shared-origin blocking, PC requirements, and window
multiplexing; the other lets every seller own their own local entropy.

So the critique — "Thai OCR is unreliable, vision is slow, stickers matter, 50k
concurrency, uptime, IP blocking, no user PCs" — was overbroad: it mistook one
extraction architecture for the whole design space. Rather than argue it, the
project **encodes** it. Every objection is a named row with a test, a receipt,
an *allowed* claim, and a *forbidden* claim, in
[`docs/OBJECTION_MATRIX.md`](docs/OBJECTION_MATRIX.md). Two rules keep it honest:

- **Denominator discipline** — no rate without saying total vs order-bearing
  messages vs UI nodes vs emitted events.
- **No category borrows trust** — synthetic ≠ device ≠ real-app ≠ business.

Live status: `python examples/objection_receipt.py --live`. What is *proven*
today is pipeline integrity `[1]` and rendered view-tree parity `[2a]`; real
hardware `[2b-1]`, real apps `[2b-2]`, and seller-hour business lift `[3]` are
deliberately **frozen and not claimed**.

### The ghost inherits the user's system

ScreenGhost runs on the user's own device and account, so its capture options
are the user's own access, tried in order: **api** (official platform events the
user's session already receives) → **view_tree** (exact on-screen text) →
**vision** (last resort) → **unsupported_surface** (named). Every path emits the
same candidate contract, so a strategy proven on one surface is reusable on the
next. This is how "try it on Messenger — they obfuscate everything" gets
answered by a *decision*: a seller with a Page receives messages via the
Messenger Platform webhook, so the app's obfuscation is bypassed, not fought;
the scrape path fails by name (`no_text_exposed`). Decided paths per platform:
[`docs/SURFACE_CAPABILITY_MATRIX.md`](docs/SURFACE_CAPABILITY_MATRIX.md)
(`python examples/capture_matrix.py`). Live platform API integration is a frozen
`[2b]` item — the routing and contract are proven, the live connections are not.

### App-surface differences are adapter tickets, not verdicts

ScreenGhost does not require every app to look identical. It requires every app
adapter to satisfy a candidate contract. If an app surface exposes exact text
and bounded rows, the adapter must prove it by conformance fixture. If it does
not expose text, the failure is explicit: `no_text_exposed`. The architecture is
not disproven by an adapter failure; the adapter has work to do unless the
surface withholds text entirely. The contract, fixtures, and verifier are in
[`docs/ADAPTER_CONFORMANCE.md`](docs/ADAPTER_CONFORMANCE.md) — run
`python examples/adapter_conformance.py --all`. No hardware or business proof is
claimed there either.

## License

MIT

## Author

Jonathan Sandhu / Sandhu Consulting Group

---

*The UI is the API. This is how it's done.*

## Multi-OS Foundation (v0.3 direction)

Screen Ghost now has a device-driver abstraction point so the core loop can stay OS-agnostic while transport/action backends vary by platform.

- `DeviceDriver` contract for screenshot + fast-hands actions
- `AndroidAdbDriver` as the default implementation
- Future drivers: iOS (WDA), macOS, Windows

This enables one orchestration/dashboard layer to route intents across fragmented apps while keeping UI automation as the fallback universal API.

## Execution Base Shape (v0.3)

To support multi-edge production use cases (airgapped legacy systems, fragmented consumer apps, and dashboard-first orchestration), this repo now includes foundational execution modules:

- `core/contracts.py`: canonical intent/action/plan contracts
- `core/policy.py`: app/action allowlist-denylist safety policy
- `core/executor.py`: deterministic step execution with postcondition verification
- `core/skills.py`: semantic skills — label-anchored waypoints with staleness demotion

These modules establish the orchestrator shape needed for "virtual intelligence + fast hands" beyond a single Android edge.

## Threat Model: Local-Only Hands

The point of fast hands is that the **execution surface is physical, not remote**.
Screen Ghost can watch, reason, and decide over as many digital pathways as you
like — but the hands that actually touch the UI stay on a physically attached
device. There is no network API to the actuation layer, so there is no remote
kill switch and no remote system to compromise. The attack surface collapses to
physical access.

That guarantee is now enforced in code, not just documented:

- `FastHandsExecutor` actually drives each canonical action into the live UI and
  then verifies the postcondition — the hands move, they aren't just checked.
- `AndroidAdbDriver` is **local-only by default**. ADB over TCP/IP
  (`adb connect host:port`) would quietly reintroduce a remote surface, so any
  `host:port` device target is rejected with `RemoteHandsError`. Pass
  `AndroidAdbDriver(allow_network=True)` to opt in with your eyes open.

> **See it argued visually:** [`docs/index.html`](docs/index.html) is a GitHub
> Pages one-pager with a live breaking-change simulation and the cost curves —
> the Page365/Trillian bottleneck, solved. Deployed by
> [`.github/workflows/pages.yml`](.github/workflows/pages.yml) on pushes to
> `main` that touch `docs/` (set Settings → Pages → Source to "GitHub Actions").

## Why the UI Is the Only Stable API

This project exists because of a lesson from the multi-protocol chat wars
(Trillian vs. AOL, circa 2000). Trillian integrated every chat network through
their protocols — and AOL changed OSCAR *specifically* to lock it out, over and
over, until maintaining per-service adapters became a war of attrition. The
pattern never stopped: APIs get deprecated, priced, rate-limited, or broken on
purpose. DOMs churn weekly. Standards (Matter, anyone?) require every vendor to
cooperate, so they arrive late and partial.

But there is exactly **one interface a vendor cannot revoke: the one their
paying customer uses.** They can break your protocol adapter overnight. They
cannot break the screen, because the screen *is* the product. Every service,
no matter how hostile to integration, must remain usable by a human — which
means it remains usable by anything that operates at the human interface.

Screen Ghost builds at that layer, so vendor churn has nowhere to bite:

- **No per-service adapters to maintain.** When an app redesigns, the VLM
  looks at the new screen and re-derives the path at runtime — the same way a
  human copes with an update. The maintenance cost that killed multi-protocol
  clients becomes a no-op.
- **Canonical intents stay stable forever.** `CanonicalTarget`/`CanonicalAction`
  (`hvac.Home.thermostat`, `set`, `72`) are the durable vocabulary; the per-app
  "how" is disposable and re-derivable.
- **Human speed per app, parallel across apps.** One user intent fans out to a
  `RunPlan` driving N vendor apps simultaneously — ten humans under one
  gesture. Each vendor just sees an ordinary user.

### Semantic Skills: Cache Meaning, Never Pixels

The one place the old failure mode could sneak back in is a skills system that
records tap coordinates — that's a per-service adapter rebuilt one macro at a
time, brittle to exactly the churn this architecture exists to shrug off.

So skills in Screen Ghost (`core/skills.py`) store **semantic waypoints**:
"tap the element labeled *Display*", never "tap (340, 520)". At replay time
each label is resolved against the live screen. When a redesign moves or
renames things, the skill goes stale, gets demoted after repeated failures,
and the VLM re-derives the path from the new screens.

```python
from core.skills import SemanticSkill, SkillStore, Waypoint
from core.executor import FastHandsExecutor

store = SkillStore()  # log/skills.json
store.save_skill(SemanticSkill(
    intent_key="toggle:display.Settings.Dark Mode",
    app="Settings",
    waypoints=[
        Waypoint(action="tap", label="Display", expect_screen="Display"),
        Waypoint(action="tap", label="Dark Mode"),
    ],
))

executor = FastHandsExecutor(skill_store=store)
# Known skills run as the fast path; stale skills fall back to live
# VLM derivation. Losing every skill costs speed, never capability.
```

The invariant: **the project adapts to everything except hands losing out to
evolution.** As long as the target system takes human input, Screen Ghost can
drive it — no vendor cooperation, no standard, no API contract required.

### First production consumer

This isn't hypothetical anymore.
[`axm-tools/pta-tracker`](https://github.com/BigBirdReturns/axm-tools) — a
GitHub-Actions legislation tracker for a school-district PTA — has exactly
one source its datacenter runner can never reach: the board-agenda system
behind Incapsula bot protection. Its escape hatch is an `observed.json`
drop-box that observer mode fills from a phone showing the agenda, an
interface no bot protection can revoke. The walkthrough lives in
[`examples/pta_agenda_observer.md`](examples/pta_agenda_observer.md).
