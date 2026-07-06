"""Seal an interface-procedure trace through genesis; verify with an
out-of-band key.

The trace shard is the proof object for one bounded interface operation:

    content/
      before.png · after.png · drift.png     (whichever the run produced, VERBATIM)
      procedure.json                          (the approved procedure, exactly)
      trace.json                              (steps, outcome, drift reason, hashes)
      interface_trace_manifest.json           (tier + limits + image manifests)
      source.txt (+ graph claims)             (genesis candidates citing the run)

Reuses Pixel Evidence v0 unchanged: every screenshot's manifest is built by
``core.pixel_evidence.build_capture`` (tier ``pixel_capture``, bytes hashed and
carried verbatim, never rewritten), and detached verification reuses
``core.pixel_exit_test.verify_detached``. Custody stays genesis's: ``shard_id``
is the genesis-derived ``sh1_``; verification uses an out-of-band key and the
frozen PASS / FAIL / MALFORMED / NO_TRUSTED_KEY taxonomy.

A DRIFT trace seals exactly like a COMPLETED one -- drift is evidence, not an
error to discard.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.interface_procedure import (
    TRACE_TIER,
    TRACE_TIER_LIMITS,
    InterfaceProcedure,
    ProcedureTrace,
)
from core.pixel_evidence import build_capture, sha256_hex
from core.pixel_seal import AXM_BUILD, VerifyStatus, kernel_available  # same frozen taxonomy

TRACE_MANIFEST_NAME = "interface_trace_manifest.json"


@dataclass(frozen=True)
class SealedTraceShard:
    shard_id: str            # genesis-derived sh1_, the ONLY custody identity
    shard_dir: str
    trusted_key_path: str    # out-of-band publisher pub (sibling to the shard)
    suite: str
    outcome: str             # completed | drift -- carried for the receipt


def _procedure_json(procedure: InterfaceProcedure) -> str:
    return json.dumps(asdict(procedure), sort_keys=True, indent=2) + "\n"


def build_trace_bundle(trace: ProcedureTrace, out_dir: str | Path) -> Path:
    """Write the pre-seal trace bundle. Screenshot bytes are copied VERBATIM;
    their manifests come from the unchanged Pixel Evidence v0 path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = trace.procedure

    images: Dict[str, Dict[str, Any]] = {}
    for name, png in (
        ("before.png", trace.before_png),
        ("after.png", trace.after_png),
        ("drift.png", trace.drift_png),
    ):
        if png is None:
            continue
        (out / name).write_bytes(png)  # verbatim, no rewrite
        capture = build_capture(
            png,
            capture_method="browser_screenshot",
            source_label=proc.surface_label,
        )
        images[name] = capture.to_manifest()

    procedure_text = _procedure_json(proc)
    (out / "procedure.json").write_text(procedure_text, encoding="utf-8")

    trace_doc = {
        "procedure_id": proc.procedure_id,
        "surface_label": proc.surface_label,
        "outcome": trace.outcome.value,
        "drift_reason": trace.drift_reason.value if trace.drift_reason else None,
        "clicked": trace.clicked,
        "click_point": list(trace.click_point) if trace.click_point else None,
        "steps": [asdict(s) for s in trace.steps],
        "image_sha256": {name: m["image_sha256"] for name, m in images.items()},
    }
    (out / "trace.json").write_text(json.dumps(trace_doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "evidence_tier": TRACE_TIER,
        "evidence_tier_limits": list(TRACE_TIER_LIMITS),
        "procedure_id": proc.procedure_id,
        "surface_label": proc.surface_label,
        "outcome": trace.outcome.value,
        "drift_reason": trace_doc["drift_reason"],
        "procedure_sha256": sha256_hex(procedure_text.encode("utf-8")),
        "verification": {
            "selector": proc.verify_selector,
            "expected_text": proc.verify_expected_text,
            "mechanism": "rendered on-screen text at an approved locator (view-tree rung); no OCR",
        },
        "images": images,  # each is a full pixel_capture manifest (tier + hash)
    }
    (out / TRACE_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return out


def _candidates_and_source(trace: ProcedureTrace, namespace: str) -> Tuple[List[dict], str]:
    proc = trace.procedure
    pid, surface = proc.procedure_id, proc.surface_label
    outcome = trace.outcome.value

    def ent(label: str, etype: str) -> dict:
        return {"type": "entity", "namespace": namespace, "label": label, "entity_type": etype}

    entities = {
        pid: ent(pid, "interface_procedure"),
        surface: ent(surface, "rendered_surface"),
        outcome: ent(outcome, "procedure_outcome"),
    }
    statements = [
        (f"{pid} was executed against {surface}",
         {"subject_label": pid, "predicate": "executed_against", "object_label": surface}),
        (f"{pid} had outcome {outcome}",
         {"subject_label": pid, "predicate": "had_outcome", "object_label": outcome}),
    ]
    if trace.drift_reason:
        reason = trace.drift_reason.value
        entities[reason] = ent(reason, "drift_reason")
        statements.append(
            (f"{pid} recorded drift {reason}",
             {"subject_label": pid, "predicate": "recorded_drift", "object_label": reason})
        )

    claims: List[dict] = []
    source = ""
    for stmt, base in statements:
        start = len(source.encode("utf-8"))
        source += stmt
        end = len(source.encode("utf-8"))
        source += "\n"
        claims.append(
            {
                "type": "claim",
                "subject_label": base["subject_label"],
                "predicate": base["predicate"],
                "object_label": base["object_label"],
                "object_type": "entity",
                "tier": 1,
                "evidence": {"source_file": "source.txt", "byte_start": start, "byte_end": end, "text": stmt},
            }
        )
    return list(entities.values()) + claims, source


def seal_trace(
    trace: ProcedureTrace,
    out_shard_dir: str | Path,
    *,
    work_dir: Optional[str | Path] = None,
    namespace: str = "screenghost/interface",
    title: str = "ScreenGhost interface procedure trace",
    created_at: str = "2026-07-04T00:00:00Z",
) -> SealedTraceShard:
    """Seal the full trace (pixels + procedure + steps) as one axm-hybrid1 shard."""
    out_shard_dir = Path(out_shard_dir)
    work = Path(work_dir) if work_dir else out_shard_dir.parent
    content_dir = work / "_content"
    key_dir = work / "keys"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    content_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    build_trace_bundle(trace, content_dir)
    candidates, source_text = _candidates_and_source(trace, namespace)
    (content_dir / "source.txt").write_text(source_text, encoding="utf-8")
    candidates_path = work / "candidates.jsonl"
    candidates_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")

    key_path = key_dir / "publisher.key"
    pub_path = key_dir / "publisher.pub"
    if not (key_path.exists() and pub_path.exists()):
        subprocess.run([AXM_BUILD, "keygen", str(key_dir), "--name", "publisher"], check=True, capture_output=True, text=True)

    subprocess.run(
        [
            AXM_BUILD, "compile", str(candidates_path), str(content_dir), str(out_shard_dir),
            "--private-key", str(key_path),
            "--namespace", namespace, "--title", title, "--created-at", created_at,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_bytes = (out_shard_dir / "manifest.json").read_bytes()
    m = json.loads(manifest_bytes)
    from axm_verify.crypto import derive_shard_id  # genesis owns custody id derivation

    return SealedTraceShard(
        shard_id=derive_shard_id(manifest_bytes),
        shard_dir=str(out_shard_dir),
        trusted_key_path=str(pub_path),
        suite=m.get("suite", "axm-hybrid1"),
        outcome=trace.outcome.value,
    )


def verify_trace(shard_dir: str | Path, trusted_key: Optional[str | Path]) -> VerifyStatus:
    """Verify through genesis with an out-of-band key (frozen taxonomy).
    Reuses the Pixel Evidence v0 mapping unchanged."""
    from core.pixel_seal import verify_pixel_evidence

    return verify_pixel_evidence(shard_dir, trusted_key)
