from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw


def png_frame(*, spinner: int = 0, background=(245, 245, 245), button_x: int = 20) -> bytes:
    image = Image.new("RGB", (200, 320), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 188, 48), fill=(225, 225, 225))
    draw.text((20, 22), "Settings", fill=(0, 0, 0))
    draw.rectangle((button_x, 80, button_x + 160, 126), outline=(0, 0, 0), width=2)
    draw.text((button_x + 10, 95), "Dark mode", fill=(0, 0, 0))
    draw.rectangle((148, 92, 170, 114), fill=(80, 80, 80))
    # Deliberately volatile clock/spinner region.
    draw.rectangle((150, 20, 180, 40), fill=(225, 225, 225))
    draw.text((154, 24), str(spinner), fill=(0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def runtime_projection(screen: str, *, surface: str = "demo.settings", content: str = "c1") -> dict[str, Any]:
    return {
        "schema": "surface_teacher_runtime_v0",
        "lesson_id": f"lesson-{screen}-{content}",
        "surface_id": surface,
        "screen_key": screen,
        "grammar_hash": f"grammar-{screen}",
        "control_hash": f"controls-{screen}",
        "content_hash": content,
        "explanation": f"Known screen {screen}",
        "elements": [
            {
                "element_id": f"{screen}-title",
                "role": "heading",
                "label": "Settings",
                "normalized_bounds": [0.06, 0.04, 0.70, 0.15],
                "interactive": False,
                "enabled": True,
                "states": {},
                "parent_element_id": None,
                "pixel_crop_sha256": "a" * 64,
                "sensitive": False,
            },
            {
                "element_id": f"{screen}-dark",
                "role": "switch",
                "label": "Dark mode",
                "normalized_bounds": [0.10, 0.25, 0.90, 0.40],
                "interactive": True,
                "enabled": True,
                "states": {"checked": "false"},
                "parent_element_id": None,
                "pixel_crop_sha256": "b" * 64,
                "sensitive": False,
            },
            {
                "element_id": f"{screen}-save",
                "role": "button",
                "label": "Save",
                "normalized_bounds": [0.55, 0.75, 0.90, 0.88],
                "interactive": True,
                "enabled": True,
                "states": {},
                "parent_element_id": None,
                "pixel_crop_sha256": "c" * 64,
                "sensitive": False,
            },
            {
                "element_id": f"{screen}-password",
                "role": "text_field",
                "label": "Password",
                "normalized_bounds": [0.10, 0.50, 0.90, 0.62],
                "interactive": True,
                "enabled": True,
                "states": {},
                "parent_element_id": None,
                "pixel_crop_sha256": "d" * 64,
                "sensitive": True,
            },
        ],
    }
