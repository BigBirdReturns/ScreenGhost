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
- [ ] v0.3: Skills system (save and replay sequences)
- [ ] v0.3: Policy engine (block dangerous actions)
- [ ] v0.4: Web UI for monitoring
- [ ] v1.0: HDMI capture for non-Android devices
- [ ] v1.0: Desktop/browser automation

## Limitations

- **Model accuracy varies**: Moondream2 is small and fast but not perfect
- **English UIs only**: Model trained primarily on English
- **Slow on CPU**: 3-5 seconds per step without GPU
- **No safety rails**: No policy engine yet—don't point it at anything dangerous

## License

MIT

## Author

Jonathan Sandhu / Sandhu Consulting Group

---

*The UI is the API. This is how it's done.*
