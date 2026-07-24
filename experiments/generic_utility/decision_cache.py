"""Semantic decision cache.

The cache stores target semantics and expected outcomes, never raw coordinates.
Coordinates are always resolved against the current frame immediately before an
action.  Repeated misses, postcondition failures, app-version changes, or screen
identity drift demote the cached decision.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from experiments.generic_utility.schema import Operator, ResolvedAction, SemanticGoal, json_bytes, sha256_json


CACHE_SCHEMA = "screenghost_semantic_decision_cache_v1"


@dataclass(frozen=True)
class DecisionKey:
    phone_grammar_version: str
    app_family: str
    screen_family: str
    operator: str
    target_semantics: str
    app_version: Optional[str] = None

    @property
    def digest(self) -> str:
        return "decision_" + sha256_json(asdict(self))


@dataclass
class CachedDecision:
    key: DecisionKey
    target_element_id: Optional[str]
    target_role: Optional[str]
    target_label: Optional[str]
    expected_screen: Optional[str]
    expected_state_key: Optional[str]
    expected_state_value: Optional[str]
    successes: int = 0
    consecutive_failures: int = 0
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["key"]["digest"] = self.key.digest
        return value


class SemanticDecisionCache:
    def __init__(self, path: Optional[str | Path] = None, *, failure_limit: int = 2):
        if failure_limit < 1:
            raise ValueError("failure_limit must be positive")
        self.path = Path(path) if path is not None else None
        self.failure_limit = int(failure_limit)
        self._entries: dict[str, CachedDecision] = {}
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != CACHE_SCHEMA:
                raise ValueError(f"unsupported decision cache schema: {payload.get('schema')!r}")
            for row in payload.get("entries", []):
                key_payload = dict(row["key"])
                key_payload.pop("digest", None)
                row = dict(row)
                row["key"] = DecisionKey(**key_payload)
                entry = CachedDecision(**row)
                self._entries[entry.key.digest] = entry

    def make_key(
        self,
        *,
        grammar_version: str,
        app_family: str,
        screen_family: str,
        goal: SemanticGoal,
        app_version: Optional[str] = None,
    ) -> DecisionKey:
        return DecisionKey(
            phone_grammar_version=grammar_version,
            app_family=app_family,
            screen_family=screen_family,
            operator=goal.operator.value,
            target_semantics=goal.normalized_target,
            app_version=app_version,
        )

    def lookup(self, key: DecisionKey) -> Optional[CachedDecision]:
        entry = self._entries.get(key.digest)
        return entry if entry is not None and entry.active else None

    def store(self, key: DecisionKey, action: ResolvedAction) -> CachedDecision:
        entry = CachedDecision(
            key=key,
            target_element_id=action.target_element_id,
            target_role=action.target_role,
            target_label=action.target_label,
            expected_screen=action.expected_screen,
            expected_state_key=action.expected_state_key,
            expected_state_value=action.expected_state_value,
        )
        existing = self._entries.get(key.digest)
        if existing is not None:
            entry.successes = existing.successes
            entry.consecutive_failures = existing.consecutive_failures
        self._entries[key.digest] = entry
        self._persist()
        return entry

    def record_success(self, key: DecisionKey) -> None:
        entry = self._entries.get(key.digest)
        if entry is None:
            return
        entry.successes += 1
        entry.consecutive_failures = 0
        entry.active = True
        self._persist()

    def record_failure(self, key: DecisionKey) -> None:
        entry = self._entries.get(key.digest)
        if entry is None:
            return
        entry.consecutive_failures += 1
        if entry.consecutive_failures >= self.failure_limit:
            entry.active = False
        self._persist()

    def invalidate_app_version(self, app_family: str, app_version: str) -> int:
        count = 0
        for entry in self._entries.values():
            if entry.key.app_family == app_family and entry.key.app_version != app_version:
                entry.active = False
                count += 1
        self._persist()
        return count

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CACHE_SCHEMA,
            "failure_limit": self.failure_limit,
            "entries": [entry.to_dict() for entry in sorted(self._entries.values(), key=lambda e: e.key.digest)],
        }
        self.path.write_bytes(json_bytes(payload))
