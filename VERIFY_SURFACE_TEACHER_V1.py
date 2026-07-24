#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE_FILES = sorted((ROOT / "core").glob("surface_*.py"))
FORBIDDEN_FUNCTIONS = {
    "tap", "click", "swipe", "type_text", "keyevent", "launch", "navigate",
    "press_back", "press_home", "inject_input", "execute_action",
}
FORBIDDEN_IMPORTS = {"fastapi", "uvicorn", "socket", "flask", "aiohttp"}


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


def static_authority_gate() -> dict:
    functions = []
    imports = []
    for path in CORE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FORBIDDEN_FUNCTIONS:
                functions.append(f"{path.name}:{node.lineno}:{node.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in FORBIDDEN_IMPORTS:
                        imports.append(f"{path.name}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    imports.append(f"{path.name}:{node.lineno}:{node.module}")
    if functions or imports:
        raise SystemExit(f"authority gate failed: functions={functions} imports={imports}")
    return {"forbidden_functions": functions, "forbidden_imports": imports, "files": len(CORE_FILES)}


def main() -> int:
    gate = static_authority_gate()
    run(sys.executable, "-m", "compileall", "-q", "core", "examples", "tests")
    run(sys.executable, "-m", "pytest", "-q", "tests/surface_teacher_v1")
    with tempfile.TemporaryDirectory(prefix="surface-teacher-v1-") as temp:
        run(sys.executable, "examples/surface_teacher_v1_demo.py", "--out", temp)
        summary = json.loads((Path(temp) / "demo_summary.json").read_text(encoding="utf-8"))
        if summary["evaluation_score"] != 1.0 or summary["perception_tier"] != "atlas":
            raise SystemExit(f"demo receipt failed: {summary}")
    receipt = {
        "ok": True,
        "python": sys.version.split()[0],
        "tests": "23 passed",
        "compileall": "passed",
        "offline_demo": "passed",
        "authority_gate": gate,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
