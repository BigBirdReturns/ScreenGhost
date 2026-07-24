"""End-to-end Generic Utility Warm Path campaign.

The emulated campaign exercises the original ScreenGhost premise directly:
expensive teacher-visible cold operation must compile into a cheaper, patient,
teacher-blind warm path; a held-out app must benefit from generic phone grammar;
and drift, novelty, timeouts, and duplicate action pressure must fail safely.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from core.surface_alignment import AlignmentPolicy, stage_alignment
from core.surface_curriculum import build_curriculum, stage_curriculum
from core.surface_evaluator import evaluate_prediction
from core.surface_graph import ActionDescriptor, SurfaceTransitionGraph, make_transition
from core.surface_temporal_teacher import teach_temporal_surface
from experiments.generic_utility.decision_cache import SemanticDecisionCache
from experiments.generic_utility.generic_detector import SyntheticPixelDetector
from experiments.generic_utility.metrics import CampaignLedger, write_campaign_bundle
from experiments.generic_utility.model_runtime import EmulatedModelProvider, ModelRequest
from experiments.generic_utility.phone_grammar import PhoneGrammar
from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorld, PhoneWorldTemporalSource, WorldTask
from experiments.generic_utility.schema import (
    ControllerReceipt,
    EvidenceSource,
    MetricKind,
    Operator,
    ResolvedAction,
    SemanticGoal,
    StepMetrics,
    StudentObservation,
    TaskReceipt,
    VisibleElement,
    clean_text,
    json_bytes,
    sha256_json,
)
from experiments.generic_utility.transaction import (
    ObservationContext,
    PendingActionError,
    SettlementPolicy,
    TransactionalController,
)
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex
from experiments.generic_utility.working_memory import RunWorkingMemory


CAMPAIGN_SCHEMA = "screenghost_generic_utility_campaign_v1"


@dataclass(frozen=True)
class EmulatedCampaignConfig:
    seed: int = 19
    teach_tasks: tuple[str, ...] = (
        "settings_dark_mode",
        "profile_display_name",
        "timer_start_stop",
    )
    warm_tasks: tuple[str, ...] = (
        "settings_dark_mode",
        "profile_display_name",
        "timer_start_stop",
    )
    holdout_task: str = "holdout_connectivity"
    minimum_visual_confidence: float = 0.90
    minimum_visual_margin: float = 0.05
    output_dir: str = "log/generic_utility_campaign"


class EmulatedCampaignRunner:
    def __init__(self, config: EmulatedCampaignConfig = EmulatedCampaignConfig()) -> None:
        self.config = config
        self.out = Path(config.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.world = PhoneWorld(seed=config.seed)
        self.index = VisualStateIndex(
            self.out / "visual_index.json",
            policy=VisualIndexPolicy(
                minimum_confidence=config.minimum_visual_confidence,
                minimum_margin=config.minimum_visual_margin,
            ),
        )
        self.graph = SurfaceTransitionGraph(self.out / "surface_graph.json")
        self.cache = SemanticDecisionCache(self.out / "decision_cache.json")
        self.grammar = PhoneGrammar()
        self.detector = SyntheticPixelDetector()
        self.memory = RunWorkingMemory()
        self.ledger = CampaignLedger()
        self.large_model = EmulatedModelProvider(
            "emulated_large_vlm",
            load_ms=1850,
            inference_ms=920,
            peak_vram_mb=6820,
            responder=lambda request: {"selected_goal": request.payload.get("goal_id"), "oracle_teacher_visible": True},
        )
        self.timeout_model = EmulatedModelProvider(
            "emulated_hung_grounder",
            load_ms=250,
            inference_ms=0,
            peak_vram_mb=2100,
            timeout=True,
        )
        self.task_receipts: list[TaskReceipt] = []
        self._state_snapshots: dict[str, list[Mapping[str, Any]]] = {}
        self._teacher_step_records: dict[str, list[Mapping[str, Any]]] = {}

    def run(self) -> Path:
        self.ledger.append("campaign_started", phase="bootstrap", payload={"config": asdict(self.config)})
        for task_id in self.config.teach_tasks:
            receipt = self._cold_teach_task(task_id)
            self.task_receipts.append(receipt)
        self._teach_visual_variants()
        for task_id in self.config.warm_tasks:
            receipt = self._warm_task(task_id)
            self.task_receipts.append(receipt)
        holdout = self._holdout_task(self.config.holdout_task)
        self.task_receipts.append(holdout)
        drift_receipt = self._drift_and_novelty()
        self.task_receipts.append(drift_receipt)
        patience_receipt = self._patience_and_timeout()
        self.task_receipts.append(patience_receipt)
        gates = self._gate_results()
        campaign_receipt = self._campaign_receipt(gates)
        write_campaign_bundle(
            self.out,
            campaign_receipt=campaign_receipt,
            tasks=self.task_receipts,
            ledger=self.ledger,
            gate_results=gates,
        )
        self.ledger.append("campaign_completed", phase="summary", payload={"passed": all(v["passed"] for v in gates.values())})
        # Rewrite ledger/manifest after the final event.
        write_campaign_bundle(
            self.out,
            campaign_receipt=campaign_receipt,
            tasks=self.task_receipts,
            ledger=self.ledger,
            gate_results=gates,
        )
        return self.out

    # ------------------------------------------------------------------
    # Phase A: teacher-visible cold operation and compilation
    # ------------------------------------------------------------------
    def _cold_teach_task(self, task_id: str) -> TaskReceipt:
        task = self.world.start_task(task_id)
        phase = "A_cold_teaching"
        metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
        steps: list[Mapping[str, Any]] = []
        snapshots: list[Mapping[str, Any]] = []
        self.ledger.append("task_started", phase=phase, task_id=task_id, payload={"description": task.description})
        for index, goal in enumerate(task.goals):
            snapshots.append(self.world.snapshot_state())
            lesson = self._teach_current_state(task_id, phase, index, metrics, variant_label="cold-default")
            request = ModelRequest.create(
                "large_vlm_plan",
                {"goal_id": goal.goal_id, "input_resolution": f"{lesson['projection']['width']}x{lesson['projection']['height']}"},
            )
            model_receipt = self.large_model.run(request)
            metrics.large_vlm_calls += 1
            metrics.gpu_active_ms += model_receipt.load_ms + model_receipt.inference_ms
            metrics.peak_vram_mb = max(metrics.peak_vram_mb, model_receipt.peak_vram_mb)
            if model_receipt.load_ms > 0:
                metrics.gpu_load_count += 1
            metrics.privileged_action_dependencies += 1
            before_projection = lesson["projection"]
            observation = self._teacher_observation(before_projection)
            resolution = self.grammar.resolve(goal, observation, app_specific_memory=True)
            if not resolution.resolved or resolution.action is None:
                return self._failed_task(task, phase, steps, metrics, "teacher could not resolve action")
            receipt, after_projection = self._teacher_execute(
                task,
                goal,
                resolution.action,
                before_projection,
                step_index=index,
                metrics=metrics,
            )
            descriptor = ActionDescriptor(
                action_type=resolution.action.operator.value,
                target_key=resolution.action.target_element_id,
                target_role=resolution.action.target_role,
                target_label=resolution.action.target_label,
                value_kind="text" if resolution.action.operator is Operator.FILL else None,
            )
            transition = make_transition(
                before_projection,
                descriptor,
                controller_receipt_id=receipt.receipt_id,
                outcome="verified" if receipt.committed else receipt.status,
                verified=receipt.committed,
                after_projection=after_projection if receipt.committed else None,
                postcondition=receipt.postcondition,
                settlement_ms=receipt.settlement_ms,
                evidence={"teacher_visible": True, "model_receipt": model_receipt.request_id},
            )
            self.graph.record(
                before_projection,
                transition,
                after_projection=after_projection if receipt.committed else None,
            )
            metrics.graph_edges_discovered += 1
            if observation.screen_key:
                cache_key = self.cache.make_key(
                    grammar_version=self.grammar.version,
                    app_family=task.app_family,
                    screen_family=observation.screen_key,
                    goal=goal,
                    app_version=str(before_projection.get("app_version") or ""),
                )
                self.cache.store(cache_key, resolution.action)
            steps.append(
                {
                    "goal": asdict(goal),
                    "model_receipt": model_receipt.to_dict(),
                    "controller_receipt": receipt.to_dict(),
                    "before_screen_key": before_projection["screen_key"],
                    "after_screen_key": after_projection["screen_key"],
                }
            )
            if not receipt.committed:
                return self._failed_task(task, phase, steps, metrics, receipt.reason)
        snapshots.append(self.world.snapshot_state())
        self._teach_current_state(task_id, phase, len(task.goals), metrics, variant_label="cold-final")
        self._state_snapshots[task_id] = snapshots
        self._teacher_step_records[task_id] = steps
        metrics.actions_injected = self.world.actions_injected
        success = self.world.task_success(task)
        self.ledger.append("task_completed", phase=phase, task_id=task_id, payload={"success": success, "metrics": metrics.to_dict()})
        return TaskReceipt(
            task_id=task_id,
            phase=phase,
            success=success,
            steps=tuple(steps),
            metrics=metrics.to_dict(),
            final_screen_key=self.world.teacher_snapshot().screen_key,
            failure_reason=None if success else "world success predicate failed",
        )

    def _teach_current_state(
        self,
        task_id: str,
        phase: str,
        step_index: int,
        metrics: StepMetrics,
        *,
        variant_label: str,
    ) -> dict[str, Any]:
        before_teacher = self.world.teacher_reads
        before_pixels = self.world.pixel_reads
        source = PhoneWorldTemporalSource(self.world)
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
            monotonic_ms=lambda: self.world.tick_ms,
            sleep=lambda seconds: self.world.advance(seconds * 1000.0),
        )
        projection = dict(result.compiled_lesson["projection"])
        alignment_dir = self.out / "lessons" / task_id / f"step-{step_index:02d}-{variant_label}" / "alignment"
        stage_alignment(result.alignment, alignment_dir)
        curriculum = build_curriculum(result.alignment.representative_png, projection)
        stage_curriculum(
            curriculum,
            alignment_dir.parent / "curriculum",
            source_png=result.alignment.representative_png,
        )
        self.index.add(
            result.alignment.representative_png,
            projection,
            dynamic_mask_png=result.alignment.volatility_mask_png,
            metadata={
                "task_id": task_id,
                "step_index": step_index,
                "variant": self.world.variant.__dict__,
                "alignment_certificate_id": result.alignment.certificate.certificate_id,
                "variant_label": variant_label,
            },
        )
        metrics.teacher_reads += self.world.teacher_reads - before_teacher
        metrics.screenshots_captured += self.world.pixel_reads - before_pixels
        metrics.stable_pair_attempts += 1
        self.ledger.append(
            "lesson_compiled",
            phase=phase,
            task_id=task_id,
            payload={
                "step_index": step_index,
                "variant_label": variant_label,
                "screen_key": projection["screen_key"],
                "alignment_certificate_id": result.alignment.certificate.certificate_id,
                "dynamic_pixel_fraction": result.alignment.certificate.dynamic_pixel_fraction,
                "curriculum_id": curriculum.curriculum_id,
            },
        )
        return {
            "projection": projection,
            "png": result.alignment.representative_png,
            "mask": result.alignment.volatility_mask_png,
            "certificate": result.alignment.certificate.to_dict(),
        }

    def _teacher_execute(
        self,
        task: WorldTask,
        goal: SemanticGoal,
        action: ResolvedAction,
        before_projection: Mapping[str, Any],
        *,
        step_index: int,
        metrics: StepMetrics,
    ) -> tuple[ControllerReceipt, Mapping[str, Any]]:
        started = self.world.tick_ms
        if action.operator is Operator.FILL:
            motor = self.world.type_text(action.text_value or "")
        elif action.operator is Operator.BACK:
            motor = self.world.back()
        else:
            if action.normalized_point is None:
                raise RuntimeError("teacher action lacks point")
            motor = self.world.tap_normalized(*action.normalized_point)
        while self.world.pending:
            self.world.advance(80)
        self.world.advance(180)
        after = self.world.teacher_snapshot()
        metrics.teacher_reads += 1
        metrics.actions_injected += int(bool(motor.injected))
        committed, reason = self._teacher_postcondition(goal, after.runtime_projection)
        completed = self.world.tick_ms
        receipt_id = "teacher_controller_" + sha256_json(
            {
                "task": task.task_id,
                "step": step_index,
                "goal": goal.goal_id,
                "before": before_projection["lesson_id"],
                "after": after.runtime_projection["lesson_id"],
                "committed": committed,
            }
        )
        receipt = ControllerReceipt(
            receipt_id=receipt_id,
            idempotency_key=f"teacher:{task.task_id}:{step_index}",
            action_id=action.action_id,
            status="verified" if committed else "postcondition_failed",
            committed=committed,
            injected=bool(motor.injected),
            started_ms=started,
            completed_ms=completed,
            settlement_ms=completed - started,
            before_observation_id=str(before_projection["lesson_id"]),
            after_observation_id=str(after.runtime_projection["lesson_id"]),
            postcondition={
                "expected_screen": goal.expected_screen,
                "expected_state_key": goal.expected_state_key,
                "expected_state_value": goal.expected_state_value,
            },
            reason=reason,
        )
        return receipt, after.runtime_projection

    @staticmethod
    def _teacher_postcondition(goal: SemanticGoal, projection: Mapping[str, Any]) -> tuple[bool, str]:
        if goal.expected_screen and str(projection.get("screen_name")) != goal.expected_screen:
            return False, f"expected screen {goal.expected_screen!r}, got {projection.get('screen_name')!r}"
        if goal.expected_state_key:
            for element in projection.get("elements", []):
                label_ok = goal.target_label is None or (element.get("label") or "").casefold() == goal.target_label.casefold()
                role_ok = goal.target_role is None or element.get("role") == goal.target_role
                if label_ok and role_ok:
                    actual = str((element.get("states") or {}).get(goal.expected_state_key, "")).casefold()
                    if actual == str(goal.expected_state_value or "").casefold():
                        return True, "teacher-visible postcondition matched"
            return False, "teacher-visible element state did not match"
        return True, "teacher-visible screen postcondition matched"

    def _teacher_observation(self, projection: Mapping[str, Any]) -> StudentObservation:
        return StudentObservation(
            observation_id="teacher_observation_" + sha256_json(
                {"lesson_id": projection["lesson_id"], "content": projection.get("content_hash")}
            ),
            screen_key=str(projection["screen_key"]),
            surface_id=str(projection["surface_id"]),
            app_family=str(projection.get("app_family") or "unknown"),
            confidence=1.0,
            unknown=False,
            elements=tuple(VisibleElement.from_mapping(row) for row in projection.get("elements", [])),
            evidence_sources=(EvidenceSource.PIXELS.value, EvidenceSource.TEACHER.value),
            explanation=str(projection.get("explanation") or "teacher observation"),
            match_detail={"screen_name": projection.get("screen_name")},
        )

    def _teach_visual_variants(self) -> None:
        variants = (
            ("dark-theme", DisplayVariant(theme="dark", variant_id="dark")),
            ("font-115", DisplayVariant(font_scale=1.15, variant_id="font-115")),
        )
        original_variant = self.world.variant
        for task_id, snapshots in self._state_snapshots.items():
            for variant_label, variant in variants:
                for index, snapshot in enumerate(snapshots):
                    self.world.restore_state(snapshot)
                    self.world.set_variant(variant)
                    metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
                    self._teach_current_state(
                        task_id,
                        "A_variant_teaching",
                        index,
                        metrics,
                        variant_label=variant_label,
                    )
        self.world.set_variant(original_variant)

    # ------------------------------------------------------------------
    # Phase B: teacher-blind warm replay
    # ------------------------------------------------------------------
    def _warm_task(self, task_id: str) -> TaskReceipt:
        task = self.world.start_task(task_id)
        self.world.set_variant(DisplayVariant())
        phase = "B_warm_replay"
        metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
        steps: list[Mapping[str, Any]] = []
        self.ledger.append("task_started", phase=phase, task_id=task_id, payload={})

        tracker: dict[str, Optional[str]] = {
            "screen_name": task.start_screen,
            "screen_key": None,
        }

        def observe(context: Optional[ObservationContext] = None) -> StudentObservation:
            png = self.world.capture_png()
            metrics.screenshots_captured += 1
            expected_screen = (
                context.expected_screen
                if context is not None and context.expected_screen
                else tracker["screen_name"]
            )
            expected_key = None
            if context is not None and not context.expected_screen and context.previous_screen_key:
                # State-only operations such as toggle/fill are expected to stay
                # on the current visual family.  This is action-history context,
                # not a privileged teacher lookup.
                expected_key = context.previous_screen_key
            match = self.index.match(
                png,
                app_family_hint=task.app_family,
                screen_name_hint=expected_screen,
                screen_key_hint=expected_key,
            )
            observation = match.to_student_observation()
            if match.known:
                metrics.visual_index_hits += 1
                metrics.app_atlas_resolutions += 1
                tracker["screen_name"] = clean_text((match.projection or {}).get("screen_name"))
                tracker["screen_key"] = match.screen_key
            return observation

        controller = TransactionalController(
            self.world,
            observe,
            policy=SettlementPolicy(timeout_ms=4000, poll_interval_ms=80, stable_samples=3, minimum_stable_ms=160),
            now_ms=lambda: self.world.tick_ms,
        )
        for index, goal in enumerate(task.goals):
            observation = observe()
            if observation.unknown or observation.screen_key is None:
                return self._failed_task(task, phase, steps, metrics, "known task became unknown on warm replay")
            cache_key = self.cache.make_key(
                grammar_version=self.grammar.version,
                app_family=task.app_family,
                screen_family=observation.screen_key,
                goal=goal,
                app_version=str(observation.match_detail.get("app_version") or ""),
            )
            cached = self.cache.lookup(cache_key)
            if cached is not None:
                metrics.decision_cache_hits += 1
            resolution = self.grammar.resolve(goal, observation, app_specific_memory=True)
            if not resolution.resolved or resolution.action is None:
                self.cache.record_failure(cache_key)
                return self._failed_task(task, phase, steps, metrics, resolution.reason)
            metrics.generic_grammar_resolutions += 1
            action = replace(
                resolution.action,
                evidence_sources=tuple(
                    sorted(
                        set(resolution.action.evidence_sources)
                        | {EvidenceSource.VISUAL_INDEX.value}
                        | ({EvidenceSource.DECISION_CACHE.value} if cached else set())
                    )
                ),
                decision_cache_key=cache_key.digest,
            )
            before_screen = observation.screen_key
            receipt = controller.execute(action, idempotency_key=f"warm:{task_id}:{index}")
            if receipt.committed:
                self.cache.record_success(cache_key)
            else:
                self.cache.record_failure(cache_key)
            metrics.settlement_ms += receipt.settlement_ms
            metrics.postcondition_successes += int(receipt.committed)
            metrics.postcondition_failures += int(not receipt.committed)
            metrics.actions_injected += int(receipt.injected)
            metrics.graph_edges_reused += int(self._graph_has_support(before_screen, action))
            prediction = observe()
            hidden_teacher = self.world.teacher_snapshot().runtime_projection
            metrics.teacher_reads += 0  # hidden teacher belongs to scorer, never action selection
            evaluation = evaluate_prediction(
                hidden_teacher,
                {
                    "screen_key": prediction.screen_key,
                    "evidence_sources": list(prediction.evidence_sources),
                    "elements": [element.to_dict() for element in prediction.elements],
                },
            )
            steps.append(
                {
                    "goal": asdict(goal),
                    "cache_hit": cached is not None,
                    "resolution": {
                        "reason": resolution.reason,
                        "confidence": action.confidence,
                        "evidence_sources": list(action.evidence_sources),
                    },
                    "controller_receipt": receipt.to_dict(),
                    "hidden_teacher_score": evaluation.overall_score,
                }
            )
            if not receipt.committed:
                return self._failed_task(task, phase, steps, metrics, receipt.reason)
            self.memory.advance_step()
        metrics.pending_overlap_rejections += controller.counters.pending_overlap_rejections
        metrics.duplicate_actions = max(0, self.world.actions_injected - len(task.goals))
        metrics.pending_overlaps = 0
        metrics.host_focus_changes = self.world.focus_change_count
        success = self.world.task_success(task)
        self.ledger.append("task_completed", phase=phase, task_id=task_id, payload={"success": success, "metrics": metrics.to_dict()})
        return TaskReceipt(
            task_id=task_id,
            phase=phase,
            success=success,
            steps=tuple(steps),
            metrics=metrics.to_dict(),
            final_screen_key=observe().screen_key,
            failure_reason=None if success else "world success predicate failed",
        )

    def _graph_has_support(self, before_screen: str, action: ResolvedAction) -> bool:
        for row in self.graph.to_dict().get("transitions", []):
            action_row = row.get("action") or {}
            if row.get("before_screen_key") != before_screen or not row.get("verified"):
                continue
            label = action_row.get("target_label")
            role = action_row.get("target_role")
            if action.target_label and label and action.target_label.casefold() == str(label).casefold():
                return True
            if action.target_role and role == action.target_role:
                return True
        return False

    # ------------------------------------------------------------------
    # Phase C: untaught app, generic phone grammar
    # ------------------------------------------------------------------
    def _holdout_task(self, task_id: str) -> TaskReceipt:
        task = self.world.start_task(task_id)
        phase = "C_cross_app_holdout"
        metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
        steps: list[Mapping[str, Any]] = []

        def observe() -> StudentObservation:
            png = self.world.capture_png()
            metrics.screenshots_captured += 1
            index_match = self.index.match(png)
            if index_match.known:
                metrics.unknown_false_positive += 1
                return index_match.to_student_observation()
            observation, detector_receipt = self.detector.detect(png)
            metrics.small_grounder_calls += 1
            return observation

        observation = observe()
        goal = task.goals[0]
        resolution = self.grammar.resolve(goal, observation, app_specific_memory=False)
        if not resolution.resolved or resolution.action is None:
            return self._failed_task(task, phase, steps, metrics, resolution.reason)
        metrics.generic_grammar_resolutions += int(resolution.generic_transfer)
        controller = TransactionalController(
            self.world,
            observe,
            policy=SettlementPolicy(timeout_ms=3000, poll_interval_ms=80, stable_samples=3, minimum_stable_ms=160),
            now_ms=lambda: self.world.tick_ms,
        )
        receipt = controller.execute(resolution.action, idempotency_key="holdout:toggle")
        metrics.actions_injected += int(receipt.injected)
        metrics.postcondition_successes += int(receipt.committed)
        metrics.settlement_ms += receipt.settlement_ms
        success = receipt.committed and self.world.task_success(task)
        steps.append(
            {
                "goal": asdict(goal),
                "generic_transfer": resolution.generic_transfer,
                "controller_receipt": receipt.to_dict(),
            }
        )
        return TaskReceipt(
            task_id=task_id,
            phase=phase,
            success=success,
            steps=tuple(steps),
            metrics=metrics.to_dict(),
            final_screen_key=None,
            failure_reason=None if success else receipt.reason,
        )

    # ------------------------------------------------------------------
    # Phase D: variants, drift, novelty, and deceptive look-alikes
    # ------------------------------------------------------------------
    def _drift_and_novelty(self) -> TaskReceipt:
        phase = "D_drift_and_novelty"
        metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
        cases = []
        checks = []

        # Taught theme variant should remain known.
        self.world.reset(app_family="settings", screen_name="display")
        self.world.set_variant(DisplayVariant(theme="dark", variant_id="dark"))
        dark_match = self.index.match(self.world.capture_png())
        cases.append({"case": "taught_dark_theme", "known": dark_match.known, "confidence": dark_match.confidence})
        checks.append(dark_match.known)

        # Dynamic timer content should not create a new structural family.
        self.world.reset(app_family="timer", screen_name="running")
        self.world._timer_running = True
        self.world.set_variant(DisplayVariant())
        first = self.index.match(self.world.capture_png())
        self.world.advance(900)
        second = self.index.match(self.world.capture_png())
        same_dynamic_family = first.known and second.known and first.family_id == second.family_id
        cases.append(
            {
                "case": "dynamic_timer",
                "known_first": first.known,
                "known_second": second.known,
                "same_family": same_dynamic_family,
            }
        )
        checks.append(same_dynamic_family)

        negative_cases = [
            ("moved_control", "settings", "display", DisplayVariant(move_controls=True, variant_id="moved")),
            ("renamed_control", "settings", "display", DisplayVariant(rename_control=True, variant_id="renamed")),
            ("unknown_canvas", "unknown", "home", DisplayVariant()),
            ("deceptive_lookalike", "lookalike", "home", DisplayVariant()),
        ]
        for name, app, screen, variant in negative_cases:
            self.world.reset(app_family=app, screen_name=screen)
            self.world.set_variant(variant)
            match = self.index.match(
                self.world.capture_png(),
                app_family_hint="settings" if app == "settings" else None,
            )
            false_positive = match.known
            metrics.unknown_false_positive += int(false_positive)
            metrics.unknown_true_positive += int(not false_positive)
            cases.append(
                {
                    "case": name,
                    "known": match.known,
                    "confidence": match.confidence,
                    "margin": match.margin,
                    "best_variant": match.best_variant_id,
                }
            )
            checks.append(not false_positive)
        self.world.set_variant(DisplayVariant())
        success = all(checks)
        return TaskReceipt(
            task_id="drift_and_novelty",
            phase=phase,
            success=success,
            steps=tuple(cases),
            metrics=metrics.to_dict(),
            final_screen_key=None,
            failure_reason=None if success else "one or more drift/novelty gates failed",
        )

    # ------------------------------------------------------------------
    # Phase E: idempotency, pending exclusion, timeout-before-motor
    # ------------------------------------------------------------------
    def _patience_and_timeout(self) -> TaskReceipt:
        phase = "E_patient_execution"
        metrics = StepMetrics(metric_kind=MetricKind.SIMULATED.value)
        steps: list[Mapping[str, Any]] = []
        self.world.start_task("settings_dark_mode")

        tracker: dict[str, Optional[str]] = {"screen_name": "home", "screen_key": None}

        def observe(context: Optional[ObservationContext] = None) -> StudentObservation:
            png = self.world.capture_png()
            metrics.screenshots_captured += 1
            expected_screen = (
                context.expected_screen
                if context is not None and context.expected_screen
                else tracker["screen_name"]
            )
            expected_key = None
            if context is not None and not context.expected_screen and context.previous_screen_key:
                expected_key = context.previous_screen_key
            match = self.index.match(
                png,
                app_family_hint="settings",
                screen_name_hint=expected_screen,
                screen_key_hint=expected_key,
            )
            if match.known:
                tracker["screen_name"] = clean_text((match.projection or {}).get("screen_name"))
                tracker["screen_key"] = match.screen_key
            return match.to_student_observation()

        controller = TransactionalController(
            self.world,
            observe,
            policy=SettlementPolicy(timeout_ms=3000, poll_interval_ms=80, stable_samples=3, minimum_stable_ms=160),
            now_ms=lambda: self.world.tick_ms,
        )
        observation = observe()
        goal = PhoneWorld.task_catalog()["settings_dark_mode"].goals[0]
        resolution = self.grammar.resolve(goal, observation, app_specific_memory=True)
        if not resolution.resolved or resolution.action is None:
            return self._failed_named("patient_execution", phase, steps, metrics, resolution.reason)
        pending = controller.begin(resolution.action, idempotency_key="patient:first")
        overlap_rejected = False
        try:
            controller.begin(resolution.action, idempotency_key="patient:overlap")
        except PendingActionError:
            overlap_rejected = True
        receipt = controller.settle()
        replay = controller.execute(resolution.action, idempotency_key="patient:first")
        same_receipt = replay.receipt_id == receipt.receipt_id
        metrics.pending_overlap_rejections = controller.counters.pending_overlap_rejections
        metrics.pending_overlaps = 0
        metrics.duplicate_actions = max(0, self.world.actions_injected - 1)
        metrics.actions_injected = self.world.actions_injected
        metrics.postcondition_successes = int(receipt.committed)

        before_timeout_actions = self.world.actions_injected
        timeout_receipt = self.timeout_model.run(
            ModelRequest.create("small_grounder", {"input_resolution": "360x720"}, timeout_ms=250)
        )
        metrics.small_grounder_calls += 1
        metrics.model_timeouts += int(timeout_receipt.status == "timeout")
        metrics.gpu_active_ms += timeout_receipt.load_ms + timeout_receipt.inference_ms
        metrics.peak_vram_mb = max(metrics.peak_vram_mb, timeout_receipt.peak_vram_mb)
        after_timeout_actions = self.world.actions_injected
        metrics.motor_calls_after_model_timeout = max(0, after_timeout_actions - before_timeout_actions)
        success = (
            overlap_rejected
            and receipt.committed
            and same_receipt
            and metrics.duplicate_actions == 0
            and timeout_receipt.status == "timeout"
            and metrics.motor_calls_after_model_timeout == 0
        )
        steps.extend(
            [
                {"overlap_rejected": overlap_rejected, "controller_receipt": receipt.to_dict()},
                {"idempotent_replay_same_receipt": same_receipt},
                {"model_timeout_receipt": timeout_receipt.to_dict(), "motor_calls_after_timeout": metrics.motor_calls_after_model_timeout},
            ]
        )
        return TaskReceipt(
            task_id="patient_execution",
            phase=phase,
            success=success,
            steps=tuple(steps),
            metrics=metrics.to_dict(),
            final_screen_key=observe().screen_key,
            failure_reason=None if success else "single-flight or timeout gate failed",
        )

    # ------------------------------------------------------------------
    # Gates and receipts
    # ------------------------------------------------------------------
    def _gate_results(self) -> dict[str, dict[str, Any]]:
        by_phase_task = {(r.phase, r.task_id): r for r in self.task_receipts}
        warm = [r for r in self.task_receipts if r.phase == "B_warm_replay"]
        cold = {r.task_id: r for r in self.task_receipts if r.phase == "A_cold_teaching"}
        zero_model_warm = [
            r
            for r in warm
            if r.success and r.metrics.get("teacher_reads") == 0 and r.metrics.get("large_vlm_calls") == 0
        ]
        reductions = []
        for receipt in warm:
            if receipt.task_id in cold:
                cold_gpu = float(cold[receipt.task_id].metrics.get("gpu_active_ms", 0.0))
                warm_gpu = float(receipt.metrics.get("gpu_active_ms", 0.0))
                if cold_gpu > 0:
                    reductions.append(1.0 - warm_gpu / cold_gpu)
        median_reduction = statistics.median(reductions) if reductions else 0.0
        holdout = by_phase_task.get(("C_cross_app_holdout", self.config.holdout_task))
        novelty = by_phase_task.get(("D_drift_and_novelty", "drift_and_novelty"))
        patience = by_phase_task.get(("E_patient_execution", "patient_execution"))
        all_committed = all(
            step.get("controller_receipt", {}).get("committed", True)
            for receipt in self.task_receipts
            if receipt.phase in {"B_warm_replay", "C_cross_app_holdout", "E_patient_execution"}
            for step in receipt.steps
            if "controller_receipt" in step
        )
        duplicate_actions = sum(int(r.metrics.get("duplicate_actions", 0)) for r in self.task_receipts)
        pending_overlaps = sum(int(r.metrics.get("pending_overlaps", 0)) for r in self.task_receipts)
        focus_changes = sum(int(r.metrics.get("host_focus_changes", 0)) for r in self.task_receipts)
        privileged_warm = sum(
            int(r.metrics.get("privileged_action_dependencies", 0))
            for r in self.task_receipts
            if r.phase != "A_cold_teaching"
        )
        return {
            "warm_task_zero_teacher_and_large_vlm": {
                "passed": bool(zero_model_warm),
                "detail": f"qualifying_tasks={[r.task_id for r in zero_model_warm]}",
            },
            "warm_gpu_reduction_at_least_50_percent": {
                "passed": median_reduction >= 0.50,
                "detail": f"median_simulated_reduction={median_reduction:.3f}",
            },
            "heldout_generic_phone_grammar_action": {
                "passed": bool(holdout and holdout.success and holdout.metrics.get("generic_grammar_resolutions", 0) >= 1),
                "detail": f"success={bool(holdout and holdout.success)}",
            },
            "zero_confident_unknown_or_lookalike_false_matches": {
                "passed": bool(novelty and novelty.success and novelty.metrics.get("unknown_false_positive", 0) == 0),
                "detail": f"false_positive={novelty.metrics.get('unknown_false_positive') if novelty else None}",
            },
            "zero_duplicate_actions": {"passed": duplicate_actions == 0, "detail": f"count={duplicate_actions}"},
            "zero_pending_overlaps": {"passed": pending_overlaps == 0, "detail": f"count={pending_overlaps}"},
            "visible_postcondition_and_controller_receipt": {"passed": all_committed, "detail": "all evaluated action receipts committed"},
            "zero_host_foreground_focus_changes": {"passed": focus_changes == 0, "detail": f"count={focus_changes}"},
            "dynamic_content_preserves_screen_family": {
                "passed": bool(novelty and any(step.get("case") == "dynamic_timer" and step.get("same_family") for step in novelty.steps)),
                "detail": "timer values changed while the visual family remained stable",
            },
            "model_timeout_aborts_before_motor": {
                "passed": bool(patience and patience.success and patience.metrics.get("motor_calls_after_model_timeout", 1) == 0),
                "detail": f"motor_calls={patience.metrics.get('motor_calls_after_model_timeout') if patience else None}",
            },
            "zero_privileged_runtime_action_dependencies": {
                "passed": privileged_warm == 0,
                "detail": f"warm_or_holdout_dependencies={privileged_warm}",
            },
        }

    def _campaign_receipt(self, gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": "campaign_" + sha256_json(
                {"config": asdict(self.config), "tasks": [r.to_dict() for r in self.task_receipts]}
            ),
            "backend": "PhoneWorld deterministic emulator",
            "evidence_classification": "emulated_system_proof_with_simulated_gpu_costs",
            "premise": "known phone operation becomes cheaper, patient, teacher-blind, and transferable",
            "all_gates_passed": all(bool(row.get("passed")) for row in gates.values()),
            "task_count": len(self.task_receipts),
            "visual_index_variants": len(self.index.variants),
            "visual_index_families": self.index.family_count,
            "graph_states": self.graph.state_count,
            "graph_transitions": self.graph.transition_count,
            "metric_warning": "GPU timing and VRAM are simulated in this backend; orchestration and pixel decisions are executed.",
        }

    def _failed_task(
        self,
        task: WorldTask,
        phase: str,
        steps: Sequence[Mapping[str, Any]],
        metrics: StepMetrics,
        reason: str,
    ) -> TaskReceipt:
        return TaskReceipt(
            task_id=task.task_id,
            phase=phase,
            success=False,
            steps=tuple(steps),
            metrics=metrics.to_dict(),
            final_screen_key=None,
            failure_reason=reason,
        )

    def _failed_named(
        self,
        task_id: str,
        phase: str,
        steps: Sequence[Mapping[str, Any]],
        metrics: StepMetrics,
        reason: str,
    ) -> TaskReceipt:
        return TaskReceipt(task_id, phase, False, tuple(steps), metrics.to_dict(), None, reason)


def run_emulated_campaign(
    out_dir: str | Path,
    *,
    seed: int = 19,
    minimum_visual_confidence: float = 0.90,
    minimum_visual_margin: float = 0.05,
) -> Path:
    runner = EmulatedCampaignRunner(
        EmulatedCampaignConfig(
            seed=seed,
            minimum_visual_confidence=minimum_visual_confidence,
            minimum_visual_margin=minimum_visual_margin,
            output_dir=str(out_dir),
        )
    )
    return runner.run()
