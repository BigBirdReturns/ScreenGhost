#!/usr/bin/env python3
"""
Screen Ghost v0.2
Autonomous UI control + state observation. No cloud. No API. No permission.

Navigator mode: See screen → Understand → Tap → Verify → Log
Observer mode:  See screen → Extract state → Emit event

The UI is the API.

Usage:
    # Navigate (existing)
    python screenghost.py --goal "Open Settings and turn on Dark Mode"
    
    # Observe (new)
    python screenghost.py --observe
    python screenghost.py --observe --continuous --interval 5
    
    # Watch for changes
    python screenghost.py --watch "Settings"

Requirements:
    - Android phone with USB debugging enabled
    - ADB installed (android-platform-tools)
    - pip install transformers torch pillow

Offline usage:
    python screenghost.py --download-model --model-path ./models/moondream2
    python screenghost.py --model-path ./models/moondream2 --observe
"""

import argparse
import io
import json
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Generator

from PIL import Image

# Model loading is deferred to avoid slow startup when just checking --help
_model = None
_processor = None
_model_path = None


# =============================================================================
# SCREEN STATE (New in v0.2)
# =============================================================================

@dataclass
class UIElement:
    """A single UI element extracted from screen."""
    type: str           # toggle, button, text, slider, input, icon
    label: str          # Human-readable label
    value: Optional[str] = None  # Current value if applicable
    bounds: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type, "label": self.label}
        if self.value is not None:
            d["value"] = self.value
        if self.bounds is not None:
            d["bounds"] = self.bounds
        return d


@dataclass
class ScreenState:
    """
    Observed state from a screenshot.
    
    This is the bridge between Screen Ghost (photonic) and GhostBox (semantic).
    """
    app: str                              # App name or "home"
    screen: str                           # Screen/page name
    elements: List[UIElement] = field(default_factory=list)
    screenshot_path: Optional[Path] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.8
    raw_response: str = ""                # VLM response for debugging
    device: Optional[str] = None
    
    # Screen dimensions
    width: int = 0
    height: int = 0
    
    @classmethod
    def from_vlm_response(
        cls, 
        response: str, 
        img: Image.Image,
        device: Optional[str] = None,
        screenshot_path: Optional[Path] = None,
    ) -> "ScreenState":
        """Parse VLM response into ScreenState."""
        width, height = img.size
        
        # Try to extract JSON from response
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                
                elements = []
                for el in data.get("elements", []):
                    elements.append(UIElement(
                        type=el.get("type", "unknown"),
                        label=el.get("label", ""),
                        value=el.get("value"),
                        bounds=tuple(el["bounds"]) if "bounds" in el else None,
                    ))
                
                return cls(
                    app=data.get("app", "unknown"),
                    screen=data.get("screen", "unknown"),
                    elements=elements,
                    screenshot_path=screenshot_path,
                    timestamp=datetime.utcnow(),
                    confidence=0.8,
                    raw_response=response,
                    device=device,
                    width=width,
                    height=height,
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass
        
        # Fallback: return minimal state
        return cls(
            app="unknown",
            screen="unknown",
            elements=[],
            screenshot_path=screenshot_path,
            timestamp=datetime.utcnow(),
            confidence=0.3,
            raw_response=response,
            device=device,
            width=width,
            height=height,
        )
    
    def to_event(self) -> Dict[str, Any]:
        """
        Convert to GhostBox-compatible event.
        
        This is the contract between Screen Ghost and GhostBox.
        """
        return {
            "topic": f"{self.app}.{self.screen}",
            "source": "screen_ghost",
            "source_type": "photonic",
            "data": {
                "app": self.app,
                "screen": self.screen,
                "elements": [el.to_dict() for el in self.elements],
                "dimensions": {"width": self.width, "height": self.height},
            },
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "device": self.device,
            "screenshot": str(self.screenshot_path) if self.screenshot_path else None,
        }
    
    def get_value(self, label: str) -> Optional[str]:
        """Get value of element by label (case-insensitive)."""
        label_lower = label.lower()
        for el in self.elements:
            if el.label.lower() == label_lower:
                return el.value
        return None
    
    def get_element(self, label: str) -> Optional[UIElement]:
        """Get element by label (case-insensitive)."""
        label_lower = label.lower()
        for el in self.elements:
            if el.label.lower() == label_lower:
                return el
        return None
    
    def has_element(self, label: str) -> bool:
        """Check if element exists."""
        return self.get_element(label) is not None
    
    def diff(self, other: "ScreenState") -> Dict[str, Any]:
        """
        Compare two states and return differences.
        
        This is key for tension detection - what changed?
        """
        changes = {
            "app_changed": self.app != other.app,
            "screen_changed": self.screen != other.screen,
            "element_changes": [],
        }
        
        # Build lookup for other state
        other_elements = {el.label.lower(): el for el in other.elements}
        self_elements = {el.label.lower(): el for el in self.elements}
        
        # Check for changed/removed elements
        for label, el in self_elements.items():
            if label not in other_elements:
                changes["element_changes"].append({
                    "type": "removed",
                    "label": el.label,
                    "old_value": el.value,
                })
            elif el.value != other_elements[label].value:
                changes["element_changes"].append({
                    "type": "changed",
                    "label": el.label,
                    "old_value": el.value,
                    "new_value": other_elements[label].value,
                })
        
        # Check for new elements
        for label, el in other_elements.items():
            if label not in self_elements:
                changes["element_changes"].append({
                    "type": "added",
                    "label": el.label,
                    "new_value": el.value,
                })
        
        changes["has_changes"] = (
            changes["app_changed"] or 
            changes["screen_changed"] or 
            len(changes["element_changes"]) > 0
        )
        
        return changes
    
    def __str__(self) -> str:
        elements_str = ", ".join(
            f"{el.label}={el.value}" if el.value else el.label 
            for el in self.elements[:5]
        )
        if len(self.elements) > 5:
            elements_str += f", ... (+{len(self.elements) - 5} more)"
        return f"ScreenState({self.app}.{self.screen}: {elements_str})"


# =============================================================================
# ADB HELPERS
# =============================================================================

def adb_available() -> bool:
    """Check if ADB is installed and accessible."""
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def adb_devices() -> List[str]:
    """List connected ADB devices."""
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")[1:]
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


def adb_text(text: str, device: Optional[str] = None) -> None:
    """Type text (new in v0.2)."""
    # Escape special characters for shell
    escaped = text.replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
    
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["shell", "input", "text", escaped])
    
    proc = subprocess.run(cmd, capture_output=True, timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(f"adb text failed: {proc.stderr.decode('utf-8', errors='ignore')}")


def adb_keyevent(keycode: int, device: Optional[str] = None) -> None:
    """Send key event (e.g., 4=back, 3=home, 66=enter)."""
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["shell", "input", "keyevent", str(keycode)])
    
    proc = subprocess.run(cmd, capture_output=True, timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(f"adb keyevent failed: {proc.stderr.decode('utf-8', errors='ignore')}")


# =============================================================================
# MODEL: MOONDREAM2
# =============================================================================

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
    
    path = model_path or _model_path
    
    if path and path.exists():
        print(f"[model] Loading moondream2 from {path}...", file=sys.stderr)
        source = str(path)
        revision = None
    else:
        print("[model] Loading moondream2 (first run downloads ~3.6GB)...", file=sys.stderr)
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
    
    print("[model] Loaded.", file=sys.stderr)
    return _model, _processor


def ask_model(img: Image.Image, prompt: str) -> str:
    """Ask moondream2 a question about an image."""
    model, processor = load_model()
    
    # Resize large images to save memory/time
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, prompt, tokenizer=processor)
    
    return answer.strip()


# =============================================================================
# OBSERVER MODE (New in v0.2)
# =============================================================================

OBSERVE_PROMPT = """Look at this Android screenshot.

Extract the current UI state as JSON:
{
  "app": "<app name or 'home' or 'launcher'>",
  "screen": "<screen/page name, e.g. 'main', 'settings', 'display'>",
  "elements": [
    {"type": "toggle", "label": "Dark Mode", "value": "on"},
    {"type": "toggle", "label": "WiFi", "value": "off"},
    {"type": "button", "label": "Save"},
    {"type": "text", "label": "Battery", "value": "85%"},
    {"type": "slider", "label": "Brightness", "value": "70%"},
    {"type": "input", "label": "Search", "value": ""},
    {"type": "icon", "label": "Settings"}
  ]
}

Rules:
- List ALL visible interactive elements (buttons, toggles, inputs, icons)
- For toggles/switches, value is "on" or "off"
- For text displays, include the current value
- For buttons/icons, no value needed
- Be precise with labels - use the exact text shown
- Return ONLY the JSON, no other text"""


def observe(
    device: Optional[str] = None,
    save_screenshot: bool = False,
    out_dir: Path = Path("log/screenshots"),
) -> ScreenState:
    """
    Extract current screen state without acting.
    
    This is the core of observer mode - turn pixels into structured state.
    """
    img = adb_screencap(device)
    
    # Optionally save screenshot
    screenshot_path = None
    if save_screenshot:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:20]
        screenshot_path = out_dir / f"observe_{ts}.png"
        img.save(screenshot_path)
    
    response = ask_model(img, OBSERVE_PROMPT)
    
    return ScreenState.from_vlm_response(
        response=response,
        img=img,
        device=device,
        screenshot_path=screenshot_path,
    )


def watch(
    device: Optional[str] = None,
    interval: float = 2.0,
    app_filter: Optional[str] = None,
    save_screenshots: bool = False,
    out_dir: Path = Path("log/screenshots"),
) -> Generator[ScreenState, None, None]:
    """
    Continuously observe screen and yield states.
    
    Optionally filter to only yield when in a specific app.
    """
    last_state = None
    
    while True:
        try:
            state = observe(
                device=device,
                save_screenshot=save_screenshots,
                out_dir=out_dir,
            )
            
            # Apply app filter
            if app_filter and app_filter.lower() not in state.app.lower():
                time.sleep(interval)
                continue
            
            # Only yield if state changed
            if last_state is None or state.diff(last_state)["has_changes"]:
                yield state
                last_state = state
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[watch] Error: {e}", file=sys.stderr)
            time.sleep(interval)


# =============================================================================
# NAVIGATOR MODE (Original, with v0.2 enhancements)
# =============================================================================

@dataclass
class TapAction:
    x: int
    y: int
    reason: str


@dataclass 
class TypeAction:
    text: str
    reason: str


@dataclass
class Action:
    """Generic action that can be tap, type, swipe, or key."""
    action_type: str  # tap, type, swipe, key
    params: Dict[str, Any]
    reason: str


def pick_action(img: Image.Image, goal: str) -> Action:
    """Ask the model what action to take to make progress toward the goal."""
    width, height = img.size
    
    prompt = f"""You are a UI automation assistant.

GOAL: {goal}

Look at this Android screenshot. Decide ONE action to take.

Available actions:
1. TAP: {{"action": "tap", "x": <number>, "y": <number>, "reason": "..."}}
2. TYPE: {{"action": "type", "text": "<text to type>", "reason": "..."}}
3. SWIPE: {{"action": "swipe", "direction": "up/down/left/right", "reason": "..."}}
4. BACK: {{"action": "back", "reason": "..."}}

The screen is {width}x{height} pixels. x=0 is left, y=0 is top.

Reply with ONLY the JSON object for your chosen action."""

    response = ask_model(img, prompt)
    
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            action_type = data.get("action", "tap")
            reason = data.get("reason", "no reason")
            
            if action_type == "tap":
                return Action(
                    action_type="tap",
                    params={"x": int(data.get("x", width // 2)), "y": int(data.get("y", height // 2))},
                    reason=reason,
                )
            elif action_type == "type":
                return Action(
                    action_type="type",
                    params={"text": data.get("text", "")},
                    reason=reason,
                )
            elif action_type == "swipe":
                return Action(
                    action_type="swipe",
                    params={"direction": data.get("direction", "up")},
                    reason=reason,
                )
            elif action_type == "back":
                return Action(
                    action_type="back",
                    params={},
                    reason=reason,
                )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    # Fallback to tap parsing (v0.1 compatibility)
    numbers = re.findall(r'\d+', response)
    if len(numbers) >= 2:
        return Action(
            action_type="tap",
            params={"x": int(numbers[0]), "y": int(numbers[1])},
            reason=f"parsed from: {response[:100]}",
        )
    
    raise ValueError(f"Could not parse action from model response: {response}")


def execute_action(action: Action, device: Optional[str] = None, screen_size: Tuple[int, int] = (1080, 1920)) -> None:
    """Execute an action on the device."""
    width, height = screen_size
    
    if action.action_type == "tap":
        adb_tap(action.params["x"], action.params["y"], device)
    
    elif action.action_type == "type":
        adb_text(action.params["text"], device)
    
    elif action.action_type == "swipe":
        direction = action.params.get("direction", "up")
        cx, cy = width // 2, height // 2
        
        if direction == "up":
            adb_swipe(cx, cy + 300, cx, cy - 300, device=device)
        elif direction == "down":
            adb_swipe(cx, cy - 300, cx, cy + 300, device=device)
        elif direction == "left":
            adb_swipe(cx + 300, cy, cx - 300, cy, device=device)
        elif direction == "right":
            adb_swipe(cx - 300, cy, cx + 300, cy, device=device)
    
    elif action.action_type == "back":
        adb_keyevent(4, device)  # KEYCODE_BACK


def check_progress(before: Image.Image, after: Image.Image, goal: str) -> Tuple[bool, str]:
    """Ask the model if the action made progress toward the goal."""
    prompt = f"""Compare these two Android screenshots (before tap on left, after tap on right).

GOAL: {goal}

Did the action move toward the goal?
Reply with ONLY: YES or NO, followed by a brief reason.

Example: "YES - settings menu is now open"
Example: "NO - nothing changed, action may have missed" """

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


# =============================================================================
# LOGGING
# =============================================================================

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
                action_type TEXT,
                action_params TEXT,
                action_reason TEXT,
                improved INTEGER,
                improve_reason TEXT,
                before_path TEXT,
                after_path TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                device TEXT,
                app TEXT,
                screen TEXT,
                state_json TEXT,
                screenshot_path TEXT
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

    def log_step(self, run_id: int, step_index: int, action: Action,
                 improved: bool, improve_reason: str,
                 before_path: Path, after_path: Path) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO steps (
                run_id, step_index, ts, action_type, action_params, action_reason,
                improved, improve_reason, before_path, after_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, step_index, datetime.utcnow().isoformat(),
            action.action_type, json.dumps(action.params), action.reason,
            int(improved), improve_reason,
            str(before_path), str(after_path),
        ))
        self._conn.commit()

    def log_observation(self, state: ScreenState) -> int:
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO observations (ts, device, app, screen, state_json, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            state.timestamp.isoformat(),
            state.device,
            state.app,
            state.screen,
            json.dumps(state.to_event()),
            str(state.screenshot_path) if state.screenshot_path else None,
        ))
        self._conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, completed: bool, steps: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET completed = ?, steps_taken = ? WHERE id = ?",
            (int(completed), steps, run_id)
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# =============================================================================
# MAIN LOOPS
# =============================================================================

def save_screenshot(img: Image.Image, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:20]
    path = out_dir / f"{prefix}_{ts}.png"
    img.save(path)
    return path


def run_navigator(goal: str, device: Optional[str], max_steps: int,
                  delay: float, out_dir: Path, db_path: Path) -> bool:
    """Navigator mode: accomplish a goal through UI automation."""
    
    logger = RunLogger(db_path)
    run_id = logger.start_run(goal)
    
    print(f"\n{'='*60}")
    print(f"SCREEN GHOST v0.2 - Navigator Mode")
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
            
            print(f"[step {step}] Asking model for next action...")
            try:
                action = pick_action(before, goal)
            except ValueError as e:
                print(f"[step {step}] ✗ Model returned unusable response: {e}")
                time.sleep(delay)
                continue
            
            print(f"[step {step}] → {action.action_type.upper()} {action.params}: {action.reason}")
            
            print(f"[step {step}] Executing action...")
            execute_action(action, device, (before.width, before.height))
            
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
            
            logger.log_step(run_id, step, action, improved, improve_reason, before_path, after_path)
            
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


def run_observer(
    device: Optional[str],
    continuous: bool,
    interval: float,
    app_filter: Optional[str],
    save_screenshots: bool,
    out_dir: Path,
    db_path: Path,
    output_format: str,
) -> None:
    """Observer mode: extract and emit screen state."""
    
    if continuous:
        print(f"[observer] Starting continuous observation (interval={interval}s)...", file=sys.stderr)
        print(f"[observer] Press Ctrl+C to stop.", file=sys.stderr)
        
        logger = RunLogger(db_path)
        
        try:
            for state in watch(
                device=device,
                interval=interval,
                app_filter=app_filter,
                save_screenshots=save_screenshots,
                out_dir=out_dir,
            ):
                logger.log_observation(state)
                
                if output_format == "json":
                    print(json.dumps(state.to_event()))
                else:
                    print(state)
                
                sys.stdout.flush()
        
        finally:
            logger.close()
    
    else:
        # Single observation
        state = observe(
            device=device,
            save_screenshot=save_screenshots,
            out_dir=out_dir,
        )
        
        if output_format == "json":
            print(json.dumps(state.to_event(), indent=2))
        else:
            print(state)
            print(f"\nElements ({len(state.elements)}):")
            for el in state.elements:
                if el.value:
                    print(f"  [{el.type}] {el.label} = {el.value}")
                else:
                    print(f"  [{el.type}] {el.label}")


# =============================================================================
# CLI
# =============================================================================

def main():
    global _model_path
    
    parser = argparse.ArgumentParser(
        description="Screen Ghost v0.2 - Autonomous UI control + observation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Navigate (accomplish a goal)
  %(prog)s --goal "Open Settings and turn on Dark Mode"
  %(prog)s --goal "Search for weather" --max-steps 15

  # Observe (extract current state)
  %(prog)s --observe
  %(prog)s --observe --format json
  %(prog)s --observe --continuous --interval 5
  %(prog)s --watch Settings

  # Offline usage
  %(prog)s --download-model --model-path ./models/moondream2
  %(prog)s --model-path ./models/moondream2 --observe
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--goal", help="Navigator mode: goal to accomplish")
    mode.add_argument("--observe", action="store_true", help="Observer mode: extract current state")
    mode.add_argument("--watch", metavar="APP", help="Watch mode: continuous observation filtered to app")
    
    # Navigator options
    parser.add_argument("--max-steps", type=int, default=10, help="Max actions before giving up (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait after each action (default: 1.0)")
    
    # Observer options
    parser.add_argument("--continuous", action="store_true", help="Continuous observation (with --observe)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between observations (default: 2.0)")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Output format (default: json)")
    parser.add_argument("--save-screenshots", action="store_true", help="Save screenshots during observation")
    
    # Common options
    parser.add_argument("--device", help="ADB device serial (if multiple connected)")
    parser.add_argument("--out-dir", type=Path, default=Path("log/screenshots"), help="Screenshot directory")
    parser.add_argument("--db", type=Path, default=Path("log/screenghost.db"), help="SQLite log path")
    parser.add_argument("--list-devices", action="store_true", help="List connected devices and exit")
    parser.add_argument("--model-path", type=Path, help="Path to local moondream2 model")
    parser.add_argument("--download-model", action="store_true", help="Download model to --model-path and exit")
    
    args = parser.parse_args()
    
    # Handle model download
    if args.download_model:
        if not args.model_path:
            print("ERROR: --download-model requires --model-path", file=sys.stderr)
            sys.exit(1)
        download_model(args.model_path)
        sys.exit(0)
    
    # Set global model path
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
    
    # Check device connection
    devices = adb_devices()
    if not devices:
        print("ERROR: No Android device connected.", file=sys.stderr)
        print("1. Enable USB debugging on your phone", file=sys.stderr)
        print("2. Connect via USB and authorize the computer", file=sys.stderr)
        sys.exit(1)
    
    if args.device and args.device not in devices:
        print(f"ERROR: Device {args.device} not found. Available: {devices}", file=sys.stderr)
        sys.exit(1)
    
    if len(devices) > 1 and not args.device:
        print(f"Multiple devices connected: {devices}", file=sys.stderr)
        print("Specify one with --device", file=sys.stderr)
        sys.exit(1)
    
    device = args.device or devices[0]
    
    # Run appropriate mode
    if args.goal:
        success = run_navigator(
            goal=args.goal,
            device=device,
            max_steps=args.max_steps,
            delay=args.delay,
            out_dir=args.out_dir,
            db_path=args.db,
        )
        sys.exit(0 if success else 1)
    
    elif args.observe or args.watch:
        run_observer(
            device=device,
            continuous=args.continuous or bool(args.watch),
            interval=args.interval,
            app_filter=args.watch,
            save_screenshots=args.save_screenshots,
            out_dir=args.out_dir,
            db_path=args.db,
            output_format=args.format,
        )
        sys.exit(0)
    
    else:
        parser.print_help()
        print("\nERROR: Specify --goal, --observe, or --watch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
