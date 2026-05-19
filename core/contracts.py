from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CanonicalTarget:
    """Stable target identity for dashboard-level intents."""

    domain: str
    location: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.location}.{self.name}"


@dataclass
class CanonicalAction:
    """Platform-agnostic action emitted by planner/intents."""

    action: str  # set/toggle/open/check
    target: CanonicalTarget
    value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunStep:
    action: CanonicalAction
    app_hint: str
    verify_label: Optional[str] = None
    verify_value: Optional[str] = None


@dataclass
class RunPlan:
    intent: str
    steps: List[RunStep]
