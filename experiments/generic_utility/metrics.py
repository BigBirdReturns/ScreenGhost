"""Append-only experiment ledger and campaign report writer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from experiments.generic_utility.schema import StepMetrics, TaskReceipt, json_bytes, sha256_bytes, sha256_json


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    sequence: int
    event_type: str
    phase: str
    task_id: Optional[str]
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase": self.phase,
            "task_id": self.task_id,
            "payload": dict(self.payload),
        }


class CampaignLedger:
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    def append(
        self,
        event_type: str,
        *,
        phase: str,
        task_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> LedgerEvent:
        sequence = len(self.events)
        body = {
            "sequence": sequence,
            "event_type": event_type,
            "phase": phase,
            "task_id": task_id,
            "payload": dict(payload or {}),
        }
        event = LedgerEvent(
            event_id="event_" + sha256_json(body),
            sequence=sequence,
            event_type=str(event_type),
            phase=str(phase),
            task_id=task_id,
            payload=dict(payload or {}),
        )
        self.events.append(event)
        return event

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"".join(json_bytes(event.to_dict()) for event in self.events))
        return out


def aggregate_metrics(receipts: Iterable[TaskReceipt]) -> StepMetrics:
    total = StepMetrics(metric_kind="derived")
    for receipt in receipts:
        row = StepMetrics(**dict(receipt.metrics))
        total.add(row)
    return total


def write_campaign_bundle(
    out_dir: str | Path,
    *,
    campaign_receipt: Mapping[str, Any],
    tasks: Iterable[TaskReceipt],
    ledger: CampaignLedger,
    gate_results: Mapping[str, Any],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_rows = [task.to_dict() for task in tasks]
    (out / "campaign_receipt.json").write_bytes(json_bytes(dict(campaign_receipt)))
    (out / "task_receipts.json").write_bytes(json_bytes({"tasks": task_rows}))
    (out / "gate_results.json").write_bytes(json_bytes(dict(gate_results)))
    ledger.write(out / "events.jsonl")
    report = _report(campaign_receipt, task_rows, gate_results)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        files[path.relative_to(out).as_posix()] = sha256_bytes(path.read_bytes())
    manifest = {
        "schema": "screenghost_generic_utility_bundle_v1",
        "campaign_id": campaign_receipt.get("campaign_id"),
        "files": files,
    }
    (out / "MANIFEST.json").write_bytes(json_bytes(manifest))
    return out


def verify_campaign_bundle(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in manifest.get("files", {}).items():
        path = out / rel
        actual = sha256_bytes(path.read_bytes()) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"ok": not mismatches, "files": len(manifest.get("files", {})), "mismatches": mismatches}


def _report(campaign: Mapping[str, Any], tasks: list[Mapping[str, Any]], gates: Mapping[str, Any]) -> str:
    lines = [
        "# ScreenGhost Generic Utility Campaign",
        "",
        f"Campaign: `{campaign.get('campaign_id')}`",
        f"Backend: `{campaign.get('backend')}`",
        f"Evidence classification: `{campaign.get('evidence_classification')}`",
        "",
        "## Task receipts",
        "",
        "| Phase | Task | Success | GPU active ms | Teacher reads | Large VLM calls | Generic resolutions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        metrics = task["metrics"]
        lines.append(
            f"| {task['phase']} | {task['task_id']} | {'yes' if task['success'] else 'no'} | "
            f"{float(metrics.get('gpu_active_ms', 0)):.1f} | {metrics.get('teacher_reads', 0)} | "
            f"{metrics.get('large_vlm_calls', 0)} | {metrics.get('generic_grammar_resolutions', 0)} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, value in gates.items():
        if isinstance(value, Mapping):
            ok = bool(value.get("passed"))
            detail = value.get("detail", "")
        else:
            ok = bool(value)
            detail = ""
        lines.append(f"- [{'x' if ok else ' '}] `{name}` {detail}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The emulated campaign proves orchestration, evidence separation, visual-index reuse, unknown gates, semantic caching, and patient settlement against deterministic rendered surfaces. Simulated GPU metrics and oracle-assisted cold planning do not establish live-model accuracy. The AndroidWorld and ADB profiles convert the same campaign contract into measured receipts on an emulator or attached device.",
            "",
        ]
    )
    return "\n".join(lines)
