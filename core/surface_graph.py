"""Teacher-compiled UI transition graph and symbolic path planner.

Surface Teacher explains a frame.  This module learns how explained frames are
connected.  It records actions performed by an external ScreenGhost controller,
observed outcomes, postcondition receipts, and settlement timing.  It never
executes an action itself.

The graph converts repeated UI operation from open-ended per-frame reasoning into
a pathfinding problem.  Uncertainty remains explicit because one action may have
multiple observed outcomes and failed attempts contribute to edge reliability.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

GRAPH_SCHEMA = "surface_teacher_transition_graph_v1"
TRANSITION_SCHEMA = "surface_teacher_transition_observation_v1"


class GraphRefused(ValueError):
    """The graph update or requested plan lacks the required evidence."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


@dataclass(frozen=True)
class GraphState:
    screen_key: str
    surface_id: str
    grammar_hash: str
    control_hash: str
    latest_content_hash: Optional[str] = None
    explanation: Optional[str] = None
    app_version: Optional[str] = None
    observation_count: int = 1

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "GraphState":
        required = ("screen_key", "surface_id", "grammar_hash", "control_hash")
        missing = [name for name in required if not _clean(projection.get(name))]
        if missing:
            raise GraphRefused(f"runtime projection is missing graph identity fields: {missing!r}")
        return cls(
            screen_key=str(projection["screen_key"]),
            surface_id=str(projection["surface_id"]),
            grammar_hash=str(projection["grammar_hash"]),
            control_hash=str(projection["control_hash"]),
            latest_content_hash=(
                str(projection["content_hash"]) if projection.get("content_hash") is not None else None
            ),
            explanation=_clean(projection.get("explanation")),
            app_version=_clean(projection.get("app_version")),
            observation_count=1,
        )


@dataclass(frozen=True)
class ActionDescriptor:
    action_type: str
    target_key: Optional[str] = None
    target_role: Optional[str] = None
    target_label: Optional[str] = None
    value_kind: Optional[str] = None
    risk: float = 0.0

    def __post_init__(self) -> None:
        if not self.action_type.strip():
            raise GraphRefused("action_type is required")
        if not 0.0 <= float(self.risk) <= 1.0:
            raise GraphRefused("action risk must be in [0,1]")

    @property
    def action_key(self) -> str:
        payload = {
            "action_type": self.action_type.strip().lower(),
            "target_key": _clean(self.target_key),
            "target_role": _clean(self.target_role),
            "target_label": _clean(self.target_label),
            "value_kind": _clean(self.value_kind),
        }
        return "action_" + _sha256(_json_bytes(payload))[:24]


@dataclass(frozen=True)
class TransitionObservation:
    schema: str
    transition_id: str
    before_screen_key: str
    after_screen_key: Optional[str]
    action: ActionDescriptor
    controller_receipt_id: str
    outcome: str
    verified: bool
    postcondition: Mapping[str, Any]
    settlement_ms: Optional[float]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"]["action_key"] = self.action.action_key
        return value


@dataclass(frozen=True)
class GraphPlanStep:
    from_screen_key: str
    to_screen_key: str
    action: ActionDescriptor
    reliability: float
    mean_settlement_ms: float
    supporting_successes: int
    total_attempts: int


@dataclass(frozen=True)
class GraphPlan:
    start_screen_key: str
    goal_screen_key: str
    steps: Tuple[GraphPlanStep, ...]
    total_cost: float


@dataclass(frozen=True)
class _EdgeAggregate:
    before: str
    after: str
    action: ActionDescriptor
    successes: int
    total_attempts: int
    reliability: float
    mean_settlement_ms: float


def make_transition(
    before_projection: Mapping[str, Any],
    action: ActionDescriptor,
    *,
    controller_receipt_id: str,
    outcome: str,
    verified: bool,
    after_projection: Optional[Mapping[str, Any]] = None,
    postcondition: Optional[Mapping[str, Any]] = None,
    settlement_ms: Optional[float] = None,
    evidence: Optional[Mapping[str, Any]] = None,
) -> TransitionObservation:
    """Create a content-addressed observation of an externally executed action."""

    receipt = _clean(controller_receipt_id)
    if not receipt:
        raise GraphRefused("every transition requires an external controller_receipt_id")
    before = GraphState.from_projection(before_projection)
    after = GraphState.from_projection(after_projection) if after_projection is not None else None
    outcome = str(outcome or "").strip().lower()
    allowed = {
        "verified",
        "postcondition_failed",
        "execution_failed",
        "settlement_timeout",
        "refused",
    }
    if outcome not in allowed:
        raise GraphRefused(f"unsupported transition outcome: {outcome!r}")
    if verified and outcome != "verified":
        raise GraphRefused("verified=True requires outcome='verified'")
    if outcome == "verified" and (not verified or after is None):
        raise GraphRefused("a verified transition requires verified=True and an after projection")
    if settlement_ms is not None:
        settlement_ms = float(settlement_ms)
        if not math.isfinite(settlement_ms) or settlement_ms < 0:
            raise GraphRefused("settlement_ms must be a finite non-negative value")
    payload = {
        "schema": TRANSITION_SCHEMA,
        "before_screen_key": before.screen_key,
        "after_screen_key": after.screen_key if after else None,
        "action": asdict(action),
        "action_key": action.action_key,
        "controller_receipt_id": receipt,
        "outcome": outcome,
        "verified": bool(verified),
        "postcondition": dict(postcondition or {}),
        "settlement_ms": settlement_ms,
        "evidence": dict(evidence or {}),
    }
    return TransitionObservation(
        schema=TRANSITION_SCHEMA,
        transition_id="transition_" + _sha256(_json_bytes(payload)),
        before_screen_key=before.screen_key,
        after_screen_key=after.screen_key if after else None,
        action=action,
        controller_receipt_id=receipt,
        outcome=outcome,
        verified=bool(verified),
        postcondition=dict(postcondition or {}),
        settlement_ms=settlement_ms,
        evidence=dict(evidence or {}),
    )


class SurfaceTransitionGraph:
    """Persistent graph of teacher-blind states and externally observed actions."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else None
        self._data: dict[str, Any] = {
            "schema": GRAPH_SCHEMA,
            "states": {},
            "transitions": [],
        }
        if self.path is not None and self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("schema") != GRAPH_SCHEMA:
                raise GraphRefused(f"unsupported graph schema: {loaded.get('schema')!r}")
            self._data = loaded

    @property
    def state_count(self) -> int:
        return len(self._data["states"])

    @property
    def transition_count(self) -> int:
        return len(self._data["transitions"])

    def add_state(self, projection: Mapping[str, Any]) -> GraphState:
        incoming = GraphState.from_projection(projection)
        states = self._data["states"]
        current = states.get(incoming.screen_key)
        if current is None:
            states[incoming.screen_key] = asdict(incoming)
        else:
            if (
                current["surface_id"] != incoming.surface_id
                or current["grammar_hash"] != incoming.grammar_hash
                or current["control_hash"] != incoming.control_hash
            ):
                raise GraphRefused(
                    f"screen_key collision or identity drift for {incoming.screen_key!r}"
                )
            current["latest_content_hash"] = incoming.latest_content_hash
            current["explanation"] = incoming.explanation
            current["app_version"] = incoming.app_version
            current["observation_count"] = int(current.get("observation_count", 0)) + 1
        self._persist()
        row = states[incoming.screen_key]
        return GraphState(**row)

    def record(
        self,
        before_projection: Mapping[str, Any],
        transition: TransitionObservation,
        *,
        after_projection: Optional[Mapping[str, Any]] = None,
    ) -> None:
        before = self.add_state(before_projection)
        if transition.before_screen_key != before.screen_key:
            raise GraphRefused("transition before_screen_key does not match supplied projection")
        if after_projection is not None:
            after = self.add_state(after_projection)
            if transition.after_screen_key != after.screen_key:
                raise GraphRefused("transition after_screen_key does not match supplied projection")
        elif transition.after_screen_key is not None:
            raise GraphRefused("after projection is required when transition names an after state")
        rows = self._data["transitions"]
        if any(row["transition_id"] == transition.transition_id for row in rows):
            return
        rows.append(transition.to_dict())
        rows.sort(key=lambda row: row["transition_id"])
        self._persist()

    def _aggregates(self) -> list[_EdgeAggregate]:
        attempts_by_action: dict[tuple[str, str], int] = {}
        successes: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        action_docs: dict[str, ActionDescriptor] = {}
        for row in self._data["transitions"]:
            action_row = dict(row["action"])
            action_row.pop("action_key", None)
            action = ActionDescriptor(**action_row)
            action_docs[action.action_key] = action
            base_key = (row["before_screen_key"], action.action_key)
            attempts_by_action[base_key] = attempts_by_action.get(base_key, 0) + 1
            if row.get("verified") and row.get("after_screen_key"):
                key = (row["before_screen_key"], action.action_key, row["after_screen_key"])
                successes.setdefault(key, []).append(row)
        out: list[_EdgeAggregate] = []
        for (before, action_key, after), rows in sorted(successes.items()):
            total = attempts_by_action[(before, action_key)]
            successful = len(rows)
            settlements = [float(row["settlement_ms"]) for row in rows if row.get("settlement_ms") is not None]
            mean_settlement = sum(settlements) / len(settlements) if settlements else 0.0
            out.append(
                _EdgeAggregate(
                    before=before,
                    after=after,
                    action=action_docs[action_key],
                    successes=successful,
                    total_attempts=total,
                    reliability=successful / total,
                    mean_settlement_ms=mean_settlement,
                )
            )
        return out

    def plan(
        self,
        start_screen_key: str,
        goal_screen_key: str,
        *,
        minimum_reliability: float = 0.5,
        risk_weight: float = 2.0,
        latency_weight: float = 0.05,
    ) -> GraphPlan:
        """Find the lowest-cost verified route through the learned UI graph."""

        if start_screen_key not in self._data["states"]:
            raise GraphRefused(f"unknown start state: {start_screen_key!r}")
        if goal_screen_key not in self._data["states"]:
            raise GraphRefused(f"unknown goal state: {goal_screen_key!r}")
        if not 0 < minimum_reliability <= 1:
            raise GraphRefused("minimum_reliability must be in (0,1]")
        if start_screen_key == goal_screen_key:
            return GraphPlan(start_screen_key, goal_screen_key, (), 0.0)

        adjacency: dict[str, list[_EdgeAggregate]] = {}
        for edge in self._aggregates():
            if edge.reliability >= minimum_reliability:
                adjacency.setdefault(edge.before, []).append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda edge: (edge.after, edge.action.action_key))

        queue: list[tuple[float, str]] = [(0.0, start_screen_key)]
        best: dict[str, float] = {start_screen_key: 0.0}
        previous: dict[str, tuple[str, _EdgeAggregate]] = {}
        while queue:
            cost, state = heapq.heappop(queue)
            if cost != best.get(state):
                continue
            if state == goal_screen_key:
                break
            for edge in adjacency.get(state, []):
                edge_cost = (
                    1.0 / max(edge.reliability, 1e-9)
                    + risk_weight * float(edge.action.risk)
                    + latency_weight * (edge.mean_settlement_ms / 1000.0)
                )
                candidate = cost + edge_cost
                if candidate + 1e-12 < best.get(edge.after, float("inf")):
                    best[edge.after] = candidate
                    previous[edge.after] = (state, edge)
                    heapq.heappush(queue, (candidate, edge.after))
        if goal_screen_key not in previous:
            raise GraphRefused(
                f"no verified route from {start_screen_key!r} to {goal_screen_key!r} "
                f"at reliability >= {minimum_reliability:.2f}"
            )

        reverse: list[GraphPlanStep] = []
        cursor = goal_screen_key
        while cursor != start_screen_key:
            prior, edge = previous[cursor]
            reverse.append(
                GraphPlanStep(
                    from_screen_key=prior,
                    to_screen_key=cursor,
                    action=edge.action,
                    reliability=round(edge.reliability, 6),
                    mean_settlement_ms=round(edge.mean_settlement_ms, 3),
                    supporting_successes=edge.successes,
                    total_attempts=edge.total_attempts,
                )
            )
            cursor = prior
        steps = tuple(reversed(reverse))
        return GraphPlan(
            start_screen_key=start_screen_key,
            goal_screen_key=goal_screen_key,
            steps=steps,
            total_cost=round(best[goal_screen_key], 8),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_json_bytes(self._data))
