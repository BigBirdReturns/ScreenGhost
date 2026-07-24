#!/usr/bin/env python3
"""Deterministic attached-worker fixture for lifecycle and timeout gates."""
from __future__ import annotations

import argparse
import json
import sys
import time


def emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("echo", "oracle", "hang", "error"), default="echo")
    parser.add_argument("--load-ms", type=float, default=5.0)
    args = parser.parse_args()
    time.sleep(max(0.0, args.load_ms) / 1000.0)
    emit({"type": "ready", "load_ms": args.load_ms, "provider": "fake_grounder"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "error": f"invalid json: {exc}"})
            continue
        if request.get("type") == "shutdown":
            return 0
        request_id = request.get("request_id")
        payload = request.get("payload") or {}
        started = time.perf_counter()
        if args.mode == "hang":
            time.sleep(3600)
        if args.mode == "error":
            emit({"type": "error", "request_id": request_id, "error": "fixture error"})
            continue
        if args.mode == "oracle":
            point = payload.get("fixture_point") or [0.5, 0.5]
            result = {"point": point, "raw_text": json.dumps(point)}
        else:
            result = {"echo": payload, "point": payload.get("point", [0.5, 0.5])}
        emit(
            {
                "type": "response",
                "request_id": request_id,
                "inference_ms": (time.perf_counter() - started) * 1000.0,
                "result": result,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
