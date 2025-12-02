# Screen Ghost

**Autonomous Android UI control. No cloud. No API. No permission.**

Screen Ghost watches your phone's screen, understands what it sees, and taps to accomplish goals. It runs entirely on your hardware using a local vision model.

The software on your phone doesn't know you're there.

## What it does

```
$ python screenghost.py --goal "Open Settings and turn on Dark Mode"

[step 0] Capturing screen...
[step 0] → Tap (540, 1847): Settings app icon in bottom dock
[step 0] ✓ YES - settings app is now opening
[step 1] Capturing screen...
[step 1] → Tap (340, 520): Display & brightness option
[step 1] ✓ YES - display settings menu is visible
[step 2] Capturing screen...
[step 2] → Tap (892, 340): Dark mode toggle switch
[step 2] ✓ YES - dark mode is now enabled

Run 1 finished. Completed: True. Steps: 3
```

## Why this matters

Every app on your phone is a walled garden. Each one wants you locked into their ecosystem. They control the API—if they offer one at all.

Screen Ghost doesn't ask for API access. It operates at the layer they can't control: the screen itself.

- **Any app.** If you can see it and tap it, so can Screen Ghost.
- **No cloud.** The vision model runs locally. Your screenshots never leave your machine.
- **No permission.** You're not using their API. You're using their UI. Like a human would.
- **Full audit trail.** Every action logged with screenshots.

## Limitations

**This is v0.1. It will not work perfectly.**

- **Model accuracy varies.** Moondream2 is small and fast but sometimes picks wrong coordinates or misreads UI elements. Complex UIs or unusual layouts may confuse it.
- **No text input yet.** Screen Ghost can tap but cannot type. Text input is planned for v0.2.
- **No scrolling yet.** If the target element is off-screen, it won't find it.
- **No retry intelligence.** If it gets stuck tapping the same wrong spot, it doesn't know to try something else.
- **English UIs only.** The model was trained primarily on English text.
- **Slow on CPU.** Expect 3-5 seconds per step without a GPU. With a GPU, under 1 second.
- **No safety rails.** There's no policy engine preventing it from tapping "Delete" or "Send". Don't point it at anything dangerous.

**This is a proof of concept, not production software.** It demonstrates that local, sovereign UI automation is possible. Making it reliable is future work.

## Offline Usage

Screen Ghost can run completely offline after a one-time model download:

```bash
# Download model once (requires internet)
python screenghost.py --download-model --model-path ./models/moondream2

# Run offline forever (no internet required)
python screenghost.py --model-path ./models/moondream2 --goal "Open Settings"
```

The model is ~3.6GB. Once downloaded, Screen Ghost never needs network access again.

## Requirements

**Hardware:**
- Any Android phone (old ones work fine)
- Computer with 8GB+ RAM (GPU helps but not required)
- USB cable

**Software:**
- Python 3.10+
- ADB (Android Debug Bridge)
- ~4GB disk space for the vision model

## Setup

### 1. Enable USB debugging on your phone

1. Go to Settings → About Phone
2. Tap "Build number" 7 times to enable Developer Options
3. Go to Settings → Developer Options
4. Enable "USB debugging"
5. Connect phone via USB and tap "Allow" when prompted

### 2. Install ADB on your computer

```bash
# Ubuntu/Debian
sudo apt install android-tools-adb

# macOS
brew install android-platform-tools

# Windows: Download from https://developer.android.com/tools/releases/platform-tools
```

### 3. Verify connection

```bash
adb devices
# Should show your device
```

### 4. Install Screen Ghost

```bash
git clone https://github.com/yourname/screenghost
cd screenghost
pip install -r requirements.txt
```

### 5. Run

```bash
python screenghost.py --goal "Open Settings and turn on Dark Mode"
```

First run downloads the vision model (~3.6GB). Subsequent runs start immediately.

## Usage

```bash
# Basic
python screenghost.py --goal "Your goal here"

# More attempts for complex tasks
python screenghost.py --goal "Send a text to Mom saying hi" --max-steps 20

# Slower for laggy phones
python screenghost.py --goal "Open camera" --delay 2.0

# Multiple phones connected
python screenghost.py --list-devices
python screenghost.py --goal "Open Settings" --device XXXXXX
```

## Examples

```bash
# Smart home control
python screenghost.py --goal "Open Google Home and set living room lights to 50%"
python screenghost.py --goal "Open Nest app and set thermostat to 72"

# Productivity
python screenghost.py --goal "Open Chrome and search for weather"
python screenghost.py --goal "Open Calendar and create event for tomorrow at 3pm"

# Settings
python screenghost.py --goal "Open Settings and turn on WiFi"
python screenghost.py --goal "Open Settings and check battery percentage"
```

## How it works

```
┌─────────────────────────────────────────┐
│          Your goal (text)               │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Screenshot ← ADB ← Phone              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Vision Model (moondream2, local)      │
│   "What should I tap to reach goal?"    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Tap → ADB → Phone                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Verify: Did screen change correctly?  │
│   If no: try different action           │
│   If yes: continue toward goal          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Log everything to SQLite              │
└─────────────────────────────────────────┘
```

## Logs

Everything is saved to `log/`:

```
log/
├── screenghost.db          # SQLite database of all runs
└── screenshots/
    ├── run1_step0_before.png
    ├── run1_step0_after.png
    └── ...
```

Query your history:

```bash
sqlite3 log/screenghost.db "SELECT * FROM runs"
sqlite3 log/screenghost.db "SELECT * FROM steps WHERE run_id = 1"
```

## Roadmap

- [ ] v0.2: Text input (`adb shell input text`)
- [ ] v0.2: Swipe gestures for scrolling
- [ ] v0.3: Skills system (save and replay successful sequences)
- [ ] v0.3: Policy engine (block certain actions)
- [ ] v0.4: Web UI for monitoring
- [ ] v1.0: HDMI capture for non-Android devices

## The bigger picture

This is a proof of concept for **the autonomy layer**: the idea that any software with a screen can be automated without permission from the vendor.

Your smart home has 5 apps that don't talk to each other? Screen Ghost doesn't care.

Your company has legacy software with no API? Screen Ghost doesn't care.

Your bank's website fights automation? Screen Ghost just sees pixels.

The screen is the universal API. This is what it looks like to use it.

## License

MIT. Do what you want.

## Contributing

PRs welcome. Especially:
- Better prompts for the vision model
- Support for other VLMs (Qwen2-VL, PaliGemma)
- Example skills for common apps
- Documentation and tutorials
