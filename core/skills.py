"""Semantic skills: cache meaning, never pixels.

The Cerulean lesson (Trillian, multi-protocol chat, ~2000): per-service
adapters die by attrition because vendors churn — sometimes adversarially.
A recorded tap-coordinate macro is exactly such an adapter, rebuilt one
screen at a time. So skills here store *semantic waypoints* ("tap the
element labeled 'Display'"), which are resolved against the live screen at
replay time. When an app redesign moves or renames things, the skill goes
stale, gets demoted, and the VLM re-derives the path from scratch — the
same way a human user copes with an update. Skills are an accelerator,
never a dependency: losing every skill costs speed, not capability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.contracts import RunStep

# A skill that keeps failing is stale (UI churned) — drop it and let the
# VLM re-derive. Consecutive, so a flaky screen doesn't kill a good skill.
MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class Waypoint:
    """One semantic step: what to do, anchored by meaning, not position.

    ``label`` names an element on whatever the screen looks like *today*;
    coordinates are resolved live from the current observation. If the
    label can't be found, the waypoint fails and the skill is stale.
    """

    action: str  # tap / type / swipe / back
    label: Optional[str] = None      # semantic anchor for tap
    value: Optional[str] = None      # text for type, direction for swipe
    expect_screen: Optional[str] = None  # postcondition hint (substring match)


@dataclass
class SemanticSkill:
    intent_key: str          # e.g. "toggle:display.Settings.Dark Mode"
    app: str
    waypoints: List[Waypoint] = field(default_factory=list)
    successes: int = 0
    consecutive_failures: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> "SemanticSkill":
        waypoints = [Waypoint(**wp) for wp in data.get("waypoints", [])]
        return cls(
            intent_key=data["intent_key"],
            app=data.get("app", "unknown"),
            waypoints=waypoints,
            successes=data.get("successes", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
        )


def intent_key_for_step(step: RunStep) -> str:
    return f"{step.action.action.lower()}:{step.action.target.key}"


class SkillStore:
    """JSON-backed store of semantic skills, keyed by canonical intent."""

    def __init__(self, path: Path = Path("log/skills.json")):
        self.path = Path(path)
        self._skills: Dict[str, SemanticSkill] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for entry in data.get("skills", []):
            skill = SemanticSkill.from_dict(entry)
            self._skills[skill.intent_key] = skill

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [asdict(s) for s in self._skills.values()]}
        self.path.write_text(json.dumps(payload, indent=2))

    def lookup(self, step: RunStep) -> Optional[SemanticSkill]:
        return self._skills.get(intent_key_for_step(step))

    def save_skill(self, skill: SemanticSkill) -> None:
        self._skills[skill.intent_key] = skill
        self._persist()

    def record_success(self, skill: SemanticSkill) -> None:
        skill.successes += 1
        skill.consecutive_failures = 0
        self._persist()

    def record_failure(self, skill: SemanticSkill) -> None:
        skill.consecutive_failures += 1
        if skill.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            # Stale: the UI evolved past this skill. Forget it entirely and
            # let live derivation rebuild the path from the new screens.
            self._skills.pop(skill.intent_key, None)
        self._persist()

    def __len__(self) -> int:
        return len(self._skills)
