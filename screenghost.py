#!/usr/bin/env python3
"""
Screen Ghost v0.1
Minimal Android autonomy layer with Moondream2 vision model.

See screen. Understand state. Tap. Verify. Log.
No cloud. No API keys. No permission.

Usage:
    python screenghost.py --goal "Open Settings and turn on Dark Mode"

Requirements:
    - Android phone with USB debugging enabled
    - ADB installed (android-platform-tools)
    - pip install transformers torch pillow

Hardware:
    - Any laptop/desktop with 8GB+ RAM, or
    - Raspberry Pi 5 with 8GB RAM (slower but works)

Offline usage:
    # Download model once (internet required)
    python screenghost.py --download-model --model-path ./models/moondream2
    
    # Run offline forever
    python screenghost.py --model-path ./models/moondream2 --goal "..."
"""

import argparse
import io
import json
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

# Model loading is deferred to avoid slow startup when just checking --help
_model = None
_processor = None
_model_path = None  # Set via CLI for offline usage


# -------------------------
# ADB helpers
# -------------------------

def adb_available() -> bool:
    """Check if ADB is installed and accessible."""
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def adb_devices() -> list[str]:
    """List connected ADB devices."""
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")[1:]  # Skip header
    devices = []
    for line in lines:
        if "\tdevice" in line:
            devices.append(line.split("\t")[0])
    return devices


def adb_screencap(device: Optional[str] = None) -> Image.Image:
    """Capture a PNG screenshot from the device and return a PIL Image."""
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["exec-out", "screencap", "-p"])
    
    proc = subprocess.run(cmd, capture_output=True, timeout=10)
    if proc.returncode != 0:
        raise RuntimeError(f"adb screencap failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    
    img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    return img


def adb_tap(x: int, y: int, device: Optional[str] = None) -> None:
    """Send a single tap at (x, y)."""
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["shell", "input", "tap", str(int(x)), str(int(y))])
    
    proc = subprocess.run(cmd, capture_output=True, timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(f"adb tap failed: {proc.stderr.decode('utf-8', errors='ignore')}")


def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, device: Optional[str] = None) -> None:
    """Swipe from (x1, y1) to (x2, y2)."""
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
    
    proc = subprocess.run(cmd, capture_output=True, timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(f"adb swipe failed: {proc.stderr.decode('utf-8', errors='ignore')}")


# -------------------------
# Model: Moondream2
# -------------------------

def download_model(path: Path) -> None:
    """Download moondream2 to a local directory for offline use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "vikhyatk/moondream2"
    revision = "2025-01-09"
    
    print(f"[model] Downloading moondream2 to {path}...")
    print("[model] This will download ~3.6GB. Please wait...")
    
    path.mkdir(parents=True, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model.save_pretrained(path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.save_pretrained(path)
    
    print(f"[model] Saved to {path}")
    print(f"[model] You can now run offline with: --model-path {path}")


def load_model(model_path: Optional[Path] = None):
    """Load Moondream2 model (lazy, cached)."""
    global _model, _processor, _model_path
    
    if _model is not None:
        return _model, _processor
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use provided path, or global path set via CLI, or download from HF
    path = model_path or _model_path
    
    if path and path.exists():
        print(f"[model] Loading moondream2 from {path}...")
        source = str(path)
        revision = None
    else:
        print("[model] Loading moondream2 (first run downloads ~3.6GB)...")
        source = "vikhyatk/moondream2"
        revision = "2025-01-09"
    
    _model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    _processor = AutoTokenizer.from_pretrained(source, revision=revision)
    
    print("[model] Loaded.")
    return _model, _processor


def ask_model(img: Image.Image, prompt: str) -> str:
    """Ask moondream2 a question about an image."""
    model, _ = load_model()
    
    # Resize large images to save memory/time
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Moondream2's API
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, prompt, tokenizer=_processor)
    
    return answer.strip()


# -------------------------
# Actions
# -------------------------

@dataclass
class TapAction:
    x: int
    y: int
    reason: str


def pick_tap(img: Image.Image, goal: str) -> TapAction:
    """Ask the model where to tap to make progress toward the goal."""
    width, height = img.size
    
    prompt = f"""You are a UI automation assistant. 

GOAL: {goal}

Look at this Android screenshot. Identify ONE element to tap that moves toward the goal.

Reply with ONLY a JSON object in this exact format:
{{"x": <number>, "y": <number>, "reason": "<brief explanation>"}}

The screen is {width}x{height} pixels. x=0 is left edge, y=0 is top edge.
Be precise with coordinates - aim for the center of the element to tap."""

    response = ask_model(img, prompt)
    
    # Parse JSON from response
    try:
        # Try to find JSON in the response
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            return TapAction(
                x=int(data.get("x", width // 2)),
                y=int(data.get("y", height // 2)),
                reason=str(data.get("reason", "no reason given"))
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    # Fallback: try to extract numbers
    numbers = re.findall(r'\d+', response)
    if len(numbers) >= 2:
        return TapAction(
            x=int(numbers[0]),
            y=int(numbers[1]),
            reason=f"parsed from: {response[:100]}"
        )
    
    raise ValueError(f"Could not parse tap coordinates from model response: {response}")


def check_progress(before: Image.Image, after: Image.Image, goal: str) -> Tuple[bool, str]:
    """Ask the model if the tap made progress toward the goal."""
    prompt = f"""Compare these two Android screenshots (before and after a tap).

GOAL: {goal}

Did the screen change in a way that moves toward the goal?
Reply with ONLY: YES or NO, followed by a brief reason.

Example: "YES - settings menu is now open"
Example: "NO - nothing changed, tap may have missed" """

    # Combine images side by side for comparison
    width = before.width + after.width
    height = max(before.height, after.height)
    combined = Image.new('RGB', (width, height))
    combined.paste(before, (0, 0))
    combined.paste(after, (before.width, 0))
    
    response = ask_model(combined, prompt)
    
    improved = response.upper().startswith("YES")
    return improved, response


def check_goal_complete(img: Image.Image, goal: str) -> Tuple[bool, str]:
    """Ask the model if the goal has been achieved."""
    prompt = f"""Look at this Android screenshot.

GOAL: {goal}

Has this goal been fully achieved? Look carefully at the current screen state.
Reply with ONLY: DONE or NOT_DONE, followed by a brief reason.

Example: "DONE - dark mode toggle is now ON"
Example: "NOT_DONE - still on home screen, need to open settings" """

    response = ask_model(img, prompt)
    
    done = response.upper().startswith("DONE")
    return done, response


# -------------------------
# Logging
# -------------------------

class RunLogger:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                goal TEXT,
                completed INTEGER DEFAULT 0,
                steps_taken INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                step_index INTEGER,
                ts TEXT,
                tap_x INTEGER,
                tap_y INTEGER,
                tap_reason TEXT,
                improved INTEGER,
                improve_reason TEXT,
                before_path TEXT,
                after_path TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
        """)
        self._conn.commit()

    def start_run(self, goal: str) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO runs (started_at, goal) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), goal),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_step(self, run_id: int, step_index: int, tap: TapAction,
                 improved: bool, improve_reason: str,
                 before_path: Path, after_path: Path) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO steps (
                run_id, step_index, ts, tap_x, tap_y, tap_reason,
                improved, improve_reason, before_path, after_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, step_index, datetime.utcnow().isoformat(),
            tap.x, tap.y, tap.reason,
            int(improved), improve_reason,
            str(before_path), str(after_path),
        ))
        self._conn.commit()

    def finish_run(self, run_id: int, completed: bool, steps: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET completed = ?, steps_taken = ? WHERE id = ?",
            (int(completed), steps, run_id)
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# -------------------------
# Main loop
# -------------------------

def save_screenshot(img: Image.Image, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:20]
    path = out_dir / f"{prefix}_{ts}.png"
    img.save(path)
    return path


def run_ghost(goal: str, device: Optional[str], max_steps: int,
              delay: float, out_dir: Path, db_path: Path) -> bool:
    """Main automation loop. Returns True if goal completed."""
    
    logger = RunLogger(db_path)
    run_id = logger.start_run(goal)
    
    print(f"\n{'='*60}")
    print(f"SCREEN GHOST v0.1")
    print(f"{'='*60}")
    print(f"Goal: {goal}")
    print(f"Device: {device or 'default'}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    completed = False
    step = 0
    
    try:
        for step in range(max_steps):
            print(f"[step {step}] Capturing screen...")
            before = adb_screencap(device)
            before_path = save_screenshot(before, out_dir, f"run{run_id}_step{step}_before")
            
            # Check if already done
            try:
                done, done_reason = check_goal_complete(before, goal)
                if done:
                    print(f"[step {step}] ✓ Goal complete: {done_reason}")
                    completed = True
                    break
            except Exception as e:
                print(f"[step {step}] Warning: Could not check goal completion: {e}")
            
            print(f"[step {step}] Asking model for next tap...")
            try:
                tap = pick_tap(before, goal)
            except ValueError as e:
                print(f"[step {step}] ✗ Model returned unusable response: {e}")
                print(f"[step {step}] Trying again with fresh screenshot...")
                time.sleep(delay)
                continue
            except Exception as e:
                print(f"[step {step}] ✗ Model error: {e}")
                print(f"[step {step}] Stopping run.")
                break
                
            print(f"[step {step}] → Tap ({tap.x}, {tap.y}): {tap.reason}")
            
            print(f"[step {step}] Executing tap...")
            adb_tap(tap.x, tap.y, device)
            
            time.sleep(delay)
            
            print(f"[step {step}] Capturing result...")
            after = adb_screencap(device)
            after_path = save_screenshot(after, out_dir, f"run{run_id}_step{step}_after")
            
            print(f"[step {step}] Checking progress...")
            try:
                improved, improve_reason = check_progress(before, after, goal)
                status = "✓" if improved else "✗"
                print(f"[step {step}] {status} {improve_reason}")
            except Exception as e:
                print(f"[step {step}] Warning: Could not verify progress: {e}")
                improved, improve_reason = True, "verification failed, assuming progress"
            
            logger.log_step(run_id, step, tap, improved, improve_reason, before_path, after_path)
            
            if not improved:
                print(f"[step {step}] No progress detected, will try different action...")
            
            print()
        
        else:
            print(f"[!] Reached max steps ({max_steps}) without completing goal.")
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    
    finally:
        logger.finish_run(run_id, completed, step + 1)
        logger.close()
    
    print(f"\n{'='*60}")
    print(f"Run {run_id} finished. Completed: {completed}. Steps: {step + 1}")
    print(f"Screenshots: {out_dir}")
    print(f"Log: {db_path}")
    print(f"{'='*60}\n")
    
    return completed


# -------------------------
# CLI
# -------------------------

def main():
    global _model_path
    
    parser = argparse.ArgumentParser(
        description="Screen Ghost v0.1 - Autonomous Android UI control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --goal "Open Settings and turn on Dark Mode"
  %(prog)s --goal "Open Chrome and search for weather"
  %(prog)s --goal "Set alarm for 7am" --max-steps 15

Offline usage:
  %(prog)s --download-model --model-path ./models/moondream2
  %(prog)s --model-path ./models/moondream2 --goal "..."

Requirements:
  - Android phone with USB debugging enabled
  - ADB installed: sudo apt install android-tools-adb
  - Python packages: pip install transformers torch pillow
        """
    )
    parser.add_argument("--goal", help="What you want the phone to do")
    parser.add_argument("--device", help="ADB device serial (if multiple connected)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max taps before giving up (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait after each tap (default: 1.0)")
    parser.add_argument("--out-dir", type=Path, default=Path("log/screenshots"), help="Where to save screenshots")
    parser.add_argument("--db", type=Path, default=Path("log/screenghost.db"), help="SQLite log path")
    parser.add_argument("--list-devices", action="store_true", help="List connected devices and exit")
    parser.add_argument("--model-path", type=Path, help="Path to local moondream2 model (for offline use)")
    parser.add_argument("--download-model", action="store_true", help="Download model to --model-path and exit")
    
    args = parser.parse_args()
    
    # Handle model download
    if args.download_model:
        if not args.model_path:
            print("ERROR: --download-model requires --model-path", file=sys.stderr)
            sys.exit(1)
        download_model(args.model_path)
        sys.exit(0)
    
    # Set global model path for offline usage
    if args.model_path:
        _model_path = args.model_path
    
    # Check ADB
    if not adb_available():
        print("ERROR: ADB not found. Install with: sudo apt install android-tools-adb", file=sys.stderr)
        sys.exit(1)
    
    if args.list_devices:
        devices = adb_devices()
        if devices:
            print("Connected devices:")
            for d in devices:
                print(f"  {d}")
        else:
            print("No devices connected.")
        sys.exit(0)
    
    # Require --goal for actual runs
    if not args.goal:
        parser.print_help()
        print("\nERROR: --goal is required", file=sys.stderr)
        sys.exit(1)
    
    # Check device connection
    devices = adb_devices()
    if not devices:
        print("ERROR: No Android device connected.", file=sys.stderr)
        print("1. Enable USB debugging on your phone (Settings → Developer Options)", file=sys.stderr)
        print("2. Connect via USB and authorize the computer", file=sys.stderr)
        sys.exit(1)
    
    if args.device and args.device not in devices:
        print(f"ERROR: Device {args.device} not found. Available: {devices}", file=sys.stderr)
        sys.exit(1)
    
    if len(devices) > 1 and not args.device:
        print(f"Multiple devices connected: {devices}", file=sys.stderr)
        print("Specify one with --device", file=sys.stderr)
        sys.exit(1)
    
    # Run
    success = run_ghost(
        goal=args.goal,
        device=args.device or devices[0],
        max_steps=args.max_steps,
        delay=args.delay,
        out_dir=args.out_dir,
        db_path=args.db,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
