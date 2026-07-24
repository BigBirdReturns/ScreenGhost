"""End-to-end semantic multibox experiment.

The campaign records a coordinate macro on one leader, distills its meaning under
Surface Teacher, measures raw coordinate replay across divergent clones, and then
replays the semantic procedure independently in every instance with teacher access
disabled at runtime.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.surface_alignment import AlignmentPolicy
from core.surface_temporal_teacher import teach_temporal_surface
from experiments.emulator_fleet.distill import distill_macro
from experiments.emulator_fleet.macro import parse_ldplayer_macro
from experiments.emulator_fleet.replay import ReplayPolicy, run_coordinate_baseline, run_semantic_procedure
from experiments.emulator_fleet.schema import FleetComparison, InstanceRunReceipt, write_json
from experiments.emulator_fleet.simulated import SimulatedFleet, SimulatedFleetSpec, SimulatedInstanceAdapter
from experiments.generic_utility.phone_grammar import PhoneGrammar
from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorldTemporalSource
from experiments.generic_utility.schema import SemanticGoal, StudentObservation, VisibleElement, sha256_bytes
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex


CAMPAIGN_SCHEMA = "semantic_multibox_campaign_v1"


def _teacher_observation(projection: Mapping[str, Any]) -> StudentObservation:
    elements = tuple(VisibleElement.from_mapping(row) for row in projection.get("elements", []))
    return StudentObservation(
        observation_id="teacher_observation_" + str(projection.get("lesson_id") or projection.get("screen_key")),
        screen_key=str(projection.get("screen_key")),
        surface_id=str(projection.get("surface_id")),
        app_family=str(projection.get("app_family")),
        confidence=1.0,
        unknown=False,
        elements=elements,
        evidence_sources=("teacher",),
        explanation=str(projection.get("explanation") or ""),
        match_detail={"screen_name": projection.get("screen_name")},
    )


def record_coordinate_demonstration(
    leader: SimulatedInstanceAdapter,
    *,
    task_id: str = "settings_dark_mode",
) -> tuple[Any, tuple[Mapping[str, Any], ...]]:
    """Record a deterministic LDPlayer-style macro from a semantic leader run."""

    task = leader.start_task(task_id)
    grammar = PhoneGrammar()
    width, height = leader.world.size
    lines = [f"size {width} {height}"]
    snapshots: list[Mapping[str, Any]] = []
    for goal in task.goals:
        snapshots.append(leader.snapshot_state())
        projection = leader.capture_teacher()
        observation = _teacher_observation(projection)
        resolution = grammar.resolve(goal, observation, app_specific_memory=True)
        if not resolution.resolved or resolution.action is None or resolution.action.normalized_point is None:
            raise RuntimeError(f"leader could not record goal {goal.goal_id}: {resolution.reason}")
        x = int(round(resolution.action.normalized_point[0] * width))
        y = int(round(resolution.action.normalized_point[1] * height))
        lines.append(f"touch {x} {y}")
        lines.append("wait 700")
        result = leader.tap_normalized(*resolution.action.normalized_point)
        if not result.injected:
            raise RuntimeError(f"leader demonstration action refused: {result.reason}")
        while leader.pending:
            leader.advance(80)
        leader.advance(180)
    snapshots.append(leader.snapshot_state())
    text = "\n".join(lines) + "\n"
    macro = parse_ldplayer_macro(text, name="Settings dark-mode coordinate baseline")
    return macro, tuple(snapshots)


def _teach_variant(
    adapter: SimulatedInstanceAdapter,
    snapshots: Sequence[Mapping[str, Any]],
    index: VisualStateIndex,
    *,
    variant_label: str,
) -> list[dict[str, Any]]:
    receipts = []
    for position, snapshot in enumerate(snapshots):
        adapter.restore_state(snapshot)
        source = PhoneWorldTemporalSource(adapter.world)
        result = teach_temporal_surface(
            source,
            lambda png, compiler_nodes, burst_digest, certificate: {
                "projection": dict(compiler_nodes),
                "burst_digest": burst_digest,
                "certificate": dict(certificate),
            },
            burst_count=3,
            interval_ms=90,
            alignment_policy=AlignmentPolicy(
                minimum_samples=3,
                minimum_duration_ms=160,
                pixel_change_threshold=10,
                max_dynamic_fraction=0.28,
                max_static_mean_difference=0.45,
                max_interactive_shift_px=2.0,
            ),
            monotonic_ms=lambda: adapter.world.tick_ms,
            sleep=lambda seconds: adapter.advance(seconds * 1000.0),
        )
        projection = dict(result.compiled_lesson["projection"])
        family_id = f"{projection.get('app_family')}:{projection.get('screen_name')}"
        variant = index.add(
            result.alignment.representative_png,
            projection,
            dynamic_mask_png=result.alignment.volatility_mask_png,
            family_id=family_id,
            metadata={
                "variant_label": variant_label,
                "snapshot_position": position,
                "alignment_certificate_id": result.alignment.certificate.certificate_id,
            },
        )
        receipts.append(
            {
                "variant_id": variant.variant_id,
                "family_id": variant.family_id,
                "screen_name": projection.get("screen_name"),
                "dynamic_pixel_fraction": result.alignment.certificate.dynamic_pixel_fraction,
                "certificate_id": result.alignment.certificate.certificate_id,
            }
        )
    return receipts


def _fleet_specs() -> list[SimulatedFleetSpec]:
    return [
        SimulatedFleetSpec(
            "fleet:exact",
            "Exact clone",
            DisplayVariant(variant_id="exact"),
            expected_baseline_success=True,
            expected_semantic_success=True,
            tags=("exact_clone",),
        ),
        SimulatedFleetSpec(
            "fleet:account",
            "Account-diverse clone",
            DisplayVariant(variant_id="account"),
            account_label="Account B",
            expected_baseline_success=True,
            expected_semantic_success=True,
            tags=("account_diverse",),
        ),
        SimulatedFleetSpec(
            "fleet:dark",
            "Dark-theme clone",
            DisplayVariant(theme="dark", variant_id="dark"),
            expected_baseline_success=True,
            expected_semantic_success=True,
            tags=("theme_variant",),
        ),
        SimulatedFleetSpec(
            "fleet:density",
            "High-density clone",
            DisplayVariant(density=1.25, variant_id="density"),
            expected_baseline_success=False,
            expected_semantic_success=True,
            tags=("geometry_variant",),
        ),
        SimulatedFleetSpec(
            "fleet:landscape",
            "Landscape clone",
            DisplayVariant(orientation="landscape", variant_id="landscape"),
            expected_baseline_success=False,
            expected_semantic_success=True,
            tags=("orientation_variant",),
        ),
        SimulatedFleetSpec(
            "fleet:moved",
            "Moved-control clone",
            DisplayVariant(move_controls=True, variant_id="moved"),
            expected_baseline_success=False,
            expected_semantic_success=True,
            tags=("layout_drift",),
        ),
        SimulatedFleetSpec(
            "fleet:renamed",
            "Renamed-control clone",
            DisplayVariant(rename_control=True, variant_id="renamed"),
            expected_baseline_success=True,
            expected_semantic_success=True,
            tags=("control_copy_drift",),
        ),
        SimulatedFleetSpec(
            "fleet:overlay",
            "Obscured-target clone",
            DisplayVariant(overlay_target=True, variant_id="overlay"),
            expected_baseline_success=False,
            expected_semantic_success=False,
            tags=("safe_abort", "occlusion"),
        ),
    ]


def _report_markdown(
    comparison: FleetComparison,
    *,
    distillation: Mapping[str, Any],
    negative_controls: Mapping[str, Any],
) -> str:
    baseline = {row.instance_id: row for row in comparison.baseline}
    semantic = {row.instance_id: row for row in comparison.semantic}
    lines = [
        "# Semantic Multibox Campaign",
        "",
        "One coordinate demonstration was distilled into a semantic procedure and replayed independently across a divergent emulator fleet.",
        "",
        f"- Procedure: `{comparison.procedure_id}`",
        f"- Distillation teacher reads: {distillation['teacher_reads']}",
        f"- Semantic advantages over coordinate replay: {comparison.metrics['semantic_advantages']}",
        f"- Runtime teacher reads: {comparison.metrics['runtime_teacher_reads']}",
        f"- Large-model calls: {comparison.metrics['large_model_calls']}",
        "",
        "| Instance | Coordinate baseline | Semantic replay | Expected semantic outcome |",
        "|---|---:|---:|---:|",
    ]
    for instance_id in sorted(baseline):
        b = baseline[instance_id]
        s = semantic[instance_id]
        lines.append(
            f"| `{instance_id}` | {'PASS' if b.success else 'FAIL'} | {'PASS' if s.success else 'SAFE ABORT'} | {'PASS' if s.expected_success else 'SAFE ABORT'} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in sorted(comparison.gates.items()):
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## Negative controls",
            "",
            f"- Unknown canvas confidently matched: {negative_controls['unknown_known']}",
            f"- Deceptive look-alike confidently matched: {negative_controls['lookalike_known']}",
            "",
            "The coordinate baseline intentionally reuses leader pixels. The semantic path resolves each current target from the instance's current visual state, issues one action, waits, and verifies the visible postcondition before proceeding.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bundle_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"schema": "semantic_multibox_bundle_v1", "files": files}
    payload["bundle_id"] = "multibox_bundle_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_bundle(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest_path = base / "MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in payload.get("files", {}).items():
        path = base / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"ok": not mismatches, "mismatches": mismatches, "bundle_id": payload.get("bundle_id")}


def run_semantic_multibox_campaign(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Record and distill one coordinate demonstration on the leader.
    leader = SimulatedInstanceAdapter(
        SimulatedFleetSpec("leader", "Leader", DisplayVariant(variant_id="leader")),
        seed=17,
    )
    macro, snapshots = record_coordinate_demonstration(leader)
    write_json(out / "macro" / "coordinate_macro.json", macro.to_dict())
    source_lines = [f"size {macro.source_resolution[0]} {macro.source_resolution[1]}"]
    for action in macro.actions:
        if action.kind.value == "tap" and action.point:
            source_lines.append(
                f"touch {int(round(action.point[0]*macro.source_resolution[0]))} {int(round(action.point[1]*macro.source_resolution[1]))}"
            )
        elif action.kind.value == "wait":
            source_lines.append(f"wait {int(action.duration_ms)}")
    (out / "macro").mkdir(parents=True, exist_ok=True)
    (out / "macro" / "coordinate_macro.ldmacro").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    leader.reset_task("settings_dark_mode")
    procedure, distillation = distill_macro(
        macro,
        leader,
        app_family="settings",
        procedure_name="Settings dark mode",
    )
    write_json(out / "procedure" / "semantic_procedure.json", procedure.to_dict())
    write_json(out / "procedure" / "distillation_receipt.json", distillation.to_dict())

    # 2. Compile visual variants once. Runtime replay will never call teacher.
    index = VisualStateIndex(
        out / "visual_index.json",
        policy=VisualIndexPolicy(
            minimum_confidence=0.92,
            minimum_margin=0.03,
            minimum_crop_confidence=0.965,
            maximum_variants_per_family=48,
        ),
    )
    teaching_receipts = []
    teaching_specs = [
        SimulatedFleetSpec("teach:default", "Teach default", DisplayVariant(variant_id="teach-default")),
        SimulatedFleetSpec("teach:account", "Teach account", DisplayVariant(variant_id="teach-account"), account_label="Account B"),
        SimulatedFleetSpec("teach:dark", "Teach dark", DisplayVariant(theme="dark", variant_id="teach-dark")),
        SimulatedFleetSpec("teach:density", "Teach density", DisplayVariant(density=1.25, variant_id="teach-density")),
        SimulatedFleetSpec("teach:landscape", "Teach landscape", DisplayVariant(orientation="landscape", variant_id="teach-landscape")),
        SimulatedFleetSpec("teach:moved", "Teach moved", DisplayVariant(move_controls=True, variant_id="teach-moved")),
        SimulatedFleetSpec("teach:renamed", "Teach renamed", DisplayVariant(rename_control=True, variant_id="teach-renamed")),
    ]
    for spec in teaching_specs:
        adapter = SimulatedInstanceAdapter(spec, seed=37)
        teaching_receipts.extend(
            _teach_variant(adapter, snapshots, index, variant_label=spec.variant.variant_id)
        )
    write_json(out / "teaching" / "variant_receipts.json", teaching_receipts)

    # 3. Coordinate baseline and semantic replay across the same divergent fleet.
    fleet = SimulatedFleet(_fleet_specs(), seed=91)
    baseline_receipts: list[InstanceRunReceipt] = []
    semantic_receipts: list[InstanceRunReceipt] = []
    for instance in fleet.list_instances():
        task = instance.reset_task("settings_dark_mode")
        baseline_receipts.append(
            run_coordinate_baseline(
                macro,
                instance,
                expected_success=instance.spec.expected_baseline_success,
                task_success=lambda i=instance, t=task: i.task_success(t),
            )
        )
    for instance in fleet.list_instances():
        task = instance.reset_task("settings_dark_mode")
        semantic_receipts.append(
            run_semantic_procedure(
                procedure,
                instance,
                index,
                expected_success=instance.spec.expected_semantic_success,
                task_success=lambda i=instance, t=task: i.task_success(t),
                policy=ReplayPolicy(app_family_hint="settings"),
            )
        )

    # 4. Unknown and deceptive look-alike gates.
    unknown = SimulatedInstanceAdapter(
        SimulatedFleetSpec("negative:unknown", "Unknown", DisplayVariant(variant_id="unknown")),
        seed=303,
    )
    unknown.world.reset(app_family="unknown", screen_name="home")
    unknown_match = index.match(unknown.capture_png())
    lookalike = SimulatedInstanceAdapter(
        SimulatedFleetSpec("negative:lookalike", "Lookalike", DisplayVariant(variant_id="lookalike")),
        seed=304,
    )
    lookalike.world.reset(app_family="lookalike", screen_name="home")
    lookalike_match = index.match(lookalike.capture_png())
    negative_controls = {
        "unknown_known": unknown_match.known,
        "unknown_confidence": unknown_match.confidence,
        "lookalike_known": lookalike_match.known,
        "lookalike_confidence": lookalike_match.confidence,
    }
    write_json(out / "negative_controls.json", negative_controls)

    baseline_by_id = {row.instance_id: row for row in baseline_receipts}
    semantic_by_id = {row.instance_id: row for row in semantic_receipts}
    expected_semantic = all(row.success == row.expected_success for row in semantic_receipts)
    safe_overlay = (
        semantic_by_id["fleet:overlay"].success is False
        and semantic_by_id["fleet:overlay"].actions_injected <= 1
    )
    semantic_advantages = sum(
        1
        for instance_id, semantic_row in semantic_by_id.items()
        if semantic_row.success and not baseline_by_id[instance_id].success
    )
    runtime_teacher_reads = sum(row.teacher_reads_runtime for row in semantic_receipts)
    duplicate_actions = sum(row.duplicate_actions for row in semantic_receipts)
    pending_overlap_rejections = sum(row.pending_overlap_rejections for row in semantic_receipts)
    large_model_calls = sum(row.large_model_calls for row in semantic_receipts)
    gates = {
        "distillation_compiled_all_actions": distillation.steps_compiled == 3,
        "exact_and_account_coordinate_baseline_pass": baseline_by_id["fleet:exact"].success and baseline_by_id["fleet:account"].success,
        "coordinate_baseline_exposes_divergence": semantic_advantages >= 3,
        "semantic_outcomes_match_declared_expectations": expected_semantic,
        "obscured_target_aborts_safely": safe_overlay,
        "runtime_teacher_reads_zero": runtime_teacher_reads == 0,
        "runtime_large_model_calls_zero": large_model_calls == 0,
        "duplicate_actions_zero": duplicate_actions == 0,
        "pending_overlap_rejections_zero": pending_overlap_rejections == 0,
        "unknown_surface_not_confidently_matched": not unknown_match.known,
        "deceptive_lookalike_not_confidently_matched": not lookalike_match.known,
    }
    metrics = {
        "fleet_size": len(fleet.instances),
        "visual_families": index.family_count,
        "visual_variants": len(index.variants),
        "semantic_advantages": semantic_advantages,
        "runtime_teacher_reads": runtime_teacher_reads,
        "large_model_calls": large_model_calls,
        "duplicate_actions": duplicate_actions,
        "pending_overlap_rejections": pending_overlap_rejections,
        "baseline_successes": sum(row.success for row in baseline_receipts),
        "semantic_successes": sum(row.success for row in semantic_receipts),
        "safe_aborts": sum((not row.success) and (not row.expected_success) for row in semantic_receipts),
    }
    comparison = FleetComparison.create(
        procedure_id=procedure.procedure_id,
        baseline=baseline_receipts,
        semantic=semantic_receipts,
        gates=gates,
        metrics=metrics,
    )
    write_json(out / "comparison.json", comparison.to_dict())
    write_json(out / "receipts" / "baseline.json", [row.to_dict() for row in baseline_receipts])
    write_json(out / "receipts" / "semantic.json", [row.to_dict() for row in semantic_receipts])
    (out / "REPORT.md").write_text(
        _report_markdown(
            comparison,
            distillation=distillation.to_dict(),
            negative_controls=negative_controls,
        ),
        encoding="utf-8",
    )
    conclusion = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "comparison_id": comparison.comparison_id,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "allowed_conclusion": (
            "A coordinate demonstration can be compiled into a semantic procedure that survives account, theme, density, orientation, layout, and control-copy variation in the deterministic fleet while independently settling and verifying each instance."
            if all(gates.values())
            else "No premise conclusion; one or more engineering gates failed."
        ),
        "forbidden_conclusions": [
            "MEmu, LDPlayer, or BlueStacks passed on this machine",
            "real third-party applications passed",
            "a local GUI model was required or measured",
            "coordinate macros are safe production controllers",
        ],
    }
    write_json(out / "CONCLUSION.json", conclusion)
    write_json(out / "MANIFEST.json", _bundle_manifest(out))
    verification = verify_bundle(out)
    write_json(out / "BUNDLE_VERIFICATION.json", verification)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"semantic multibox campaign failed gates: {failed}")
    if not verification["ok"]:
        raise RuntimeError(f"campaign bundle verification failed: {verification['mismatches']}")
    return out
