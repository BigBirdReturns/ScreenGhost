"""Run-scoped perception cascade for an 8 GB local GPU.

The 4060 should not load a general VLM for every frame.  This module encodes a
cheap-to-expensive cascade and an in-process accelerator queue:

1. exact atlas recognition;
2. local crop/prototype retrieval;
3. small GUI grounder;
4. larger VLM for genuinely novel or contradictory states;
5. teacher review outside teacher-blind evaluation.

The queue is not a daemon and exposes no listener.  One orchestrator constructs
it, enqueues content-addressed work, drains one model kind at a time, and then
lets it die with the run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


class PerceptionRefused(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PerceptionTier(str, Enum):
    ATLAS = "atlas"
    PROTOTYPE = "prototype"
    SMALL_GROUNDER = "small_grounder"
    LARGE_VLM = "large_vlm"
    TEACHER_REVIEW = "teacher_review"


@dataclass(frozen=True)
class PerceptionPolicy:
    atlas_accept_confidence: float = 0.94
    prototype_accept_confidence: float = 0.90
    small_grounder_accept_confidence: float = 0.78
    large_vlm_accept_confidence: float = 0.72
    novelty_escalation: float = 0.45
    changed_fraction_escalation: float = 0.30
    allow_large_vlm: bool = True
    allow_teacher_review: bool = False

    def require_valid(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, float) and not 0.0 <= value <= 1.0:
                raise PerceptionRefused(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class PerceptionRequest:
    request_id: str
    goal: str
    screen_identity: str
    atlas_confidence: float = 0.0
    prototype_confidence: float = 0.0
    small_grounder_confidence: float = 0.0
    large_vlm_confidence: float = 0.0
    novelty: float = 1.0
    changed_fraction: float = 1.0
    contradictory: bool = False
    teacher_blind: bool = True

    @classmethod
    def create(cls, goal: str, screen_identity: str, **kwargs: Any) -> "PerceptionRequest":
        payload = {"goal": str(goal), "screen_identity": str(screen_identity), **kwargs}
        return cls(
            request_id="perception_" + _sha256(_json_bytes(payload)),
            goal=str(goal),
            screen_identity=str(screen_identity),
            **kwargs,
        )


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    selected_tier: PerceptionTier
    accepted: bool
    reason: str
    attempted_tiers: Tuple[PerceptionTier, ...]


def route_perception(
    request: PerceptionRequest,
    policy: PerceptionPolicy = PerceptionPolicy(),
) -> RoutingDecision:
    policy.require_valid()
    for name in (
        "atlas_confidence",
        "prototype_confidence",
        "small_grounder_confidence",
        "large_vlm_confidence",
        "novelty",
        "changed_fraction",
    ):
        value = float(getattr(request, name))
        if not 0.0 <= value <= 1.0:
            raise PerceptionRefused(f"request {name} must be in [0,1]")

    attempted: list[PerceptionTier] = [PerceptionTier.ATLAS]
    if (
        not request.contradictory
        and request.atlas_confidence >= policy.atlas_accept_confidence
        and request.novelty < policy.novelty_escalation
    ):
        return RoutingDecision(
            request.request_id,
            PerceptionTier.ATLAS,
            True,
            "known screen accepted from teacher-blind atlas memory",
            tuple(attempted),
        )

    attempted.append(PerceptionTier.PROTOTYPE)
    if (
        not request.contradictory
        and request.prototype_confidence >= policy.prototype_accept_confidence
        and request.changed_fraction < policy.changed_fraction_escalation
    ):
        return RoutingDecision(
            request.request_id,
            PerceptionTier.PROTOTYPE,
            True,
            "stable crop prototypes resolved the visible target",
            tuple(attempted),
        )

    attempted.append(PerceptionTier.SMALL_GROUNDER)
    if request.small_grounder_confidence >= policy.small_grounder_accept_confidence:
        return RoutingDecision(
            request.request_id,
            PerceptionTier.SMALL_GROUNDER,
            True,
            "small GUI-specific grounder met the action threshold",
            tuple(attempted),
        )

    if policy.allow_large_vlm:
        attempted.append(PerceptionTier.LARGE_VLM)
        if request.large_vlm_confidence >= policy.large_vlm_accept_confidence:
            return RoutingDecision(
                request.request_id,
                PerceptionTier.LARGE_VLM,
                True,
                "novel or contradictory state required broad visual reasoning",
                tuple(attempted),
            )

    if policy.allow_teacher_review and not request.teacher_blind:
        attempted.append(PerceptionTier.TEACHER_REVIEW)
        return RoutingDecision(
            request.request_id,
            PerceptionTier.TEACHER_REVIEW,
            False,
            "runtime confidence was insufficient; privileged review requested and recorded",
            tuple(attempted),
        )

    final_tier = attempted[-1]
    return RoutingDecision(
        request.request_id,
        final_tier,
        False,
        "no declared perception tier met its confidence threshold",
        tuple(attempted),
    )


@dataclass(frozen=True)
class InferenceKind:
    name: str
    capability_probe: Callable[[], Mapping[str, Any]]
    runner: Optional[Callable[[Mapping[str, Any]], Any]] = None
    result_exists: Optional[Callable[[Mapping[str, Any]], bool]] = None

    def capability(self) -> dict[str, Any]:
        try:
            result = dict(self.capability_probe())
        except Exception as exc:
            result = {"ready": False, "error": str(exc)[:300]}
        result["name"] = self.name
        result["runner_registered"] = self.runner is not None
        if self.runner is None:
            result["ready"] = False
        return result


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    kind: str
    identity: str
    lane: str
    payload: Mapping[str, Any]
    sequence: int


class RunScopedInferenceQueue:
    """Deterministic, one-model-at-a-time queue with no background lifetime."""

    LANES = ("interactive", "warm")

    def __init__(self, kinds: Sequence[InferenceKind]):
        self._kinds = {kind.name: kind for kind in kinds}
        if len(self._kinds) != len(kinds):
            raise PerceptionRefused("inference kind names must be unique")
        self._jobs: list[InferenceJob] = []
        self._ids: set[str] = set()
        self.receipts: list[dict[str, Any]] = []

    @staticmethod
    def job_id(kind: str, identity: str, payload: Mapping[str, Any]) -> str:
        stable_payload = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool, type(None), list, tuple))
        }
        return "infer_" + _sha256(
            _json_bytes({"kind": kind, "identity": identity, "payload": stable_payload})
        )

    def enqueue(
        self,
        kind: str,
        identity: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        lane: str = "interactive",
    ) -> Optional[InferenceJob]:
        if kind not in self._kinds:
            raise PerceptionRefused(f"unknown inference kind: {kind!r}")
        if lane not in self.LANES:
            raise PerceptionRefused(f"unknown lane: {lane!r}")
        payload = dict(payload or {})
        job_id = self.job_id(kind, identity, payload)
        if job_id in self._ids:
            return None
        job = InferenceJob(job_id, kind, str(identity), lane, payload, len(self._jobs))
        self._jobs.append(job)
        self._ids.add(job_id)
        return job

    def ordered_jobs(self) -> Tuple[InferenceJob, ...]:
        ordered: list[InferenceJob] = []
        for lane in self.LANES:
            lane_jobs = [job for job in self._jobs if job.lane == lane]
            kind_order: list[str] = []
            for job in lane_jobs:
                if job.kind not in kind_order:
                    kind_order.append(job.kind)
            for kind in kind_order:
                ordered.extend(job for job in lane_jobs if job.kind == kind)
        return tuple(ordered)

    def capabilities(self) -> dict[str, dict[str, Any]]:
        return {name: kind.capability() for name, kind in sorted(self._kinds.items())}

    def drain(self, *, maximum_jobs: int = 0) -> dict[str, Any]:
        ran = cached = errors = 0
        for job in self.ordered_jobs():
            if maximum_jobs and ran >= maximum_jobs:
                break
            kind = self._kinds[job.kind]
            try:
                if kind.result_exists is not None and kind.result_exists(asdict(job)):
                    cached += 1
                    self.receipts.append({"job_id": job.job_id, "kind": job.kind, "status": "cached"})
                    continue
                if kind.runner is None:
                    errors += 1
                    self.receipts.append(
                        {"job_id": job.job_id, "kind": job.kind, "status": "error", "reason": "no runner"}
                    )
                    continue
                result = kind.runner(asdict(job))
                ran += 1
                self.receipts.append(
                    {"job_id": job.job_id, "kind": job.kind, "status": "done", "result": result}
                )
            except Exception as exc:
                errors += 1
                self.receipts.append(
                    {"job_id": job.job_id, "kind": job.kind, "status": "error", "reason": str(exc)[:300]}
                )
        return {"ran": ran, "cached": cached, "errors": errors, "receipts": list(self.receipts)}
