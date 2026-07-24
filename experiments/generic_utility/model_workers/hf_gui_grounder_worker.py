#!/usr/bin/env python3
"""Run-scoped Hugging Face GUI grounder worker.

The process loads one image-text-to-text model, emits a ready record, accepts
JSONL requests over stdin, and exits when stdin closes or a shutdown record
arrives.  It binds no socket.  The worker is intentionally generic and records
its raw model text; ScreenGhost's benchmark parser owns coordinate extraction.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def extract_point(text: str) -> list[float] | None:
    candidates: list[tuple[float, float]] = []
    for match in re.finditer(
        r'(?i)["\']?x["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*[,}\s]+["\']?y["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)',
        text,
    ):
        candidates.append((float(match.group(1)), float(match.group(2))))
    for match in re.finditer(r'[\[(]\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\s*[\])]', text):
        candidates.append((float(match.group(1)), float(match.group(2))))
    if not candidates:
        return None
    x, y = candidates[-1]
    if max(abs(x), abs(y)) > 1.5:
        x, y = x / 1000.0, y / 1000.0
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return [x, y]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    loaded_at = time.perf_counter()
    try:
        import torch
        from transformers import pipeline

        dtype: Any = "auto"
        if args.dtype not in {"auto", ""}:
            dtype = getattr(torch, args.dtype)
        pipe = pipeline(
            "image-text-to-text",
            model=args.model,
            device_map=args.device_map,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        emit({"type": "error", "error": f"model load failed: {type(exc).__name__}: {exc}"})
        return 2
    load_ms = (time.perf_counter() - loaded_at) * 1000.0
    emit({"type": "ready", "load_ms": load_ms, "model": args.model})

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "error": f"invalid request json: {exc}"})
            continue
        if request.get("type") == "shutdown":
            return 0
        request_id = str(request.get("request_id") or "")
        payload = request.get("payload") or {}
        image_path = Path(str(payload.get("image_path") or ""))
        prompt = str(payload.get("prompt") or payload.get("instruction") or "")
        if not image_path.is_file() or not prompt:
            emit({"type": "error", "request_id": request_id, "error": "image_path and prompt are required"})
            continue
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            started = time.perf_counter()
            output = pipe(messages, max_new_tokens=args.max_new_tokens, do_sample=False)
            inference_ms = (time.perf_counter() - started) * 1000.0
            generated = output
            if isinstance(output, list) and output:
                generated = output[0].get("generated_text", output[0]) if isinstance(output[0], dict) else output[0]
            if isinstance(generated, list) and generated:
                last = generated[-1]
                text = str(last.get("content", last)) if isinstance(last, dict) else str(last)
            else:
                text = str(generated)
            emit(
                {
                    "type": "response",
                    "request_id": request_id,
                    "inference_ms": inference_ms,
                    "result": {"point": extract_point(text), "raw_text": text},
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"inference failed: {type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
