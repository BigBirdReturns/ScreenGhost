#!/usr/bin/env python3
"""Real Chromium DPR-2 and animated-surface receipt, with zero input actions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from playwright.sync_api import sync_playwright

from core.surface_alignment import AlignmentNode, FrameObservation, certify_alignment, stage_alignment


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def image_size(data: bytes):
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=args.chromium,
            args=["--no-sandbox", "--disable-background-networking", "--disable-sync"],
        )
        browser_version = browser.version

        dpr_context = browser.new_context(
            viewport={"width": 240, "height": 160},
            device_scale_factor=2,
        )
        dpr_requests = []
        dpr_context.on("request", lambda request: dpr_requests.append(request.url))
        dpr_context.route("**/*", lambda route: route.abort())
        dpr_page = dpr_context.new_page()
        dpr_page.set_content(
            """
            <style>
              html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; }
              #full { display: block; width: 100%; height: 100%; border: 0; }
            </style>
            <button id="full" aria-label="Full viewport control">Full viewport control</button>
            """
        )
        dpr_png = dpr_page.screenshot(type="png", scale="device")
        dpr_record = dpr_page.evaluate(
            """() => {
              const el = document.querySelector('#full');
              const r = el.getBoundingClientRect();
              return {
                device_pixel_ratio: devicePixelRatio,
                css_viewport: [innerWidth, innerHeight],
                device_bounds: [
                  r.x * devicePixelRatio,
                  r.y * devicePixelRatio,
                  (r.x + r.width) * devicePixelRatio,
                  (r.y + r.height) * devicePixelRatio
                ]
              };
            }"""
        )
        width, height = image_size(dpr_png)
        x1, y1, x2, y2 = dpr_record["device_bounds"]
        normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
        if (width, height) != (480, 320) or normalized != [0.0, 0.0, 1.0, 1.0]:
            raise SystemExit(
                f"DPR receipt failed: screenshot={(width, height)} bounds={dpr_record['device_bounds']} "
                f"normalized={normalized}"
            )
        (out / "dpr2_surface.png").write_bytes(dpr_png)
        dpr_context.close()

        motion_context = browser.new_context(
            viewport={"width": 360, "height": 720},
            device_scale_factor=2,
        )
        motion_requests = []
        motion_context.on("request", lambda request: motion_requests.append(request.url))
        motion_context.route("**/*", lambda route: route.abort())
        page = motion_context.new_page()
        page.set_content(
            """
            <style>
              html, body { margin: 0; width: 100%; height: 100%; background: #fafafa; font-family: sans-serif; }
              header { height: 64px; background: #e8e8e8; display: flex; align-items: center; padding: 0 18px; }
              #clock { margin-left: auto; width: 48px; text-align: right; }
              #dark { position: absolute; left: 20px; top: 130px; width: 320px; height: 64px; }
              #save { position: absolute; left: 210px; top: 610px; width: 120px; height: 58px; }
            </style>
            <header><strong>Settings</strong><span id="clock">0</span></header>
            <button id="dark" role="switch" aria-checked="false">Dark mode</button>
            <button id="save">Save</button>
            <script>
              let count = 0;
              setInterval(() => { document.querySelector('#clock').textContent = String(++count); }, 70);
            </script>
            """
        )
        observations = []
        capture_records = []
        for index in range(3):
            png = page.screenshot(type="png", scale="device")
            record = page.evaluate(
                """() => {
                  const read = (selector) => {
                    const el = document.querySelector(selector);
                    const r = el.getBoundingClientRect();
                    return {
                      bounds: [r.x * devicePixelRatio, r.y * devicePixelRatio,
                               (r.x + r.width) * devicePixelRatio,
                               (r.y + r.height) * devicePixelRatio],
                      text: el.textContent.trim()
                    };
                  };
                  return { dpr: devicePixelRatio, clock: read('#clock'), dark: read('#dark'), save: read('#save') };
                }"""
            )
            nodes = (
                AlignmentNode("clock", "text", tuple(record["clock"]["bounds"]), label=record["clock"]["text"], dynamic=True),
                AlignmentNode("dark", "switch", tuple(record["dark"]["bounds"]), label="Dark mode", interactive=True,
                              states=(("checked", "false"),)),
                AlignmentNode("save", "button", tuple(record["save"]["bounds"]), label="Save", interactive=True),
            )
            observations.append(
                FrameObservation(
                    png_bytes=png,
                    nodes=nodes,
                    observed_monotonic_ms=time.monotonic() * 1000.0,
                    source_digest=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                    event_idle=True,
                    note=f"real-chromium-{index}",
                )
            )
            capture_records.append(record)
            if index < 2:
                page.wait_for_timeout(120)
        alignment = certify_alignment(observations)
        stage_alignment(alignment, out / "animated_alignment")
        motion_context.close()
        browser.close()

    receipt = {
        "schema": "surface_teacher_real_playwright_receipt_v1",
        "browser": "system chromium through Playwright",
        "browser_version": browser_version,
        "playwright_version": importlib.metadata.version("playwright"),
        "page_network_requests": dpr_requests + motion_requests,
        "network_used": bool(dpr_requests or motion_requests),
        "input_actions": 0,
        "dpr": 2,
        "css_viewport": dpr_record["css_viewport"],
        "screenshot_pixels": [width, height],
        "device_bounds": dpr_record["device_bounds"],
        "normalized_bounds": normalized,
        "dpr_geometry_pass": True,
        "animated_surface_certificate_id": alignment.certificate.certificate_id,
        "animated_surface_dynamic_fraction": alignment.certificate.dynamic_pixel_fraction,
        "animated_surface_static_mean_difference": alignment.certificate.static_mean_difference,
        "animated_surface_pass": True,
        "capture_clock_values": [record["clock"]["text"] for record in capture_records],
    }
    if receipt["network_used"]:
        raise SystemExit(f"unexpected page network requests: {receipt['page_network_requests']}")
    (out / "receipt.json").write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
