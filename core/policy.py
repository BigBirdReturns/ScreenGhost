from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set


@dataclass
class SafetyPolicy:
    """Simple allowlist policy for app-level execution."""

    allowed_apps: Set[str] = field(default_factory=set)
    blocked_actions: Set[str] = field(default_factory=lambda: {"factory_reset", "purchase", "delete_account"})

    def allows_app(self, app: str) -> bool:
        return not self.allowed_apps or app.lower() in {a.lower() for a in self.allowed_apps}

    def allows_action(self, action: str) -> bool:
        return action.lower() not in {a.lower() for a in self.blocked_actions}

    @classmethod
    def from_iterables(cls, allowed_apps: Iterable[str], blocked_actions: Iterable[str] | None = None) -> "SafetyPolicy":
        return cls(set(allowed_apps), set(blocked_actions or []))
