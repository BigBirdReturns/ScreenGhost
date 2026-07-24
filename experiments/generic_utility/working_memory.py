"""Run-scoped task memory kept separate from screen-family identity."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from experiments.generic_utility.schema import json_bytes, sha256_json


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    key: str
    value_kind: str
    value: Any
    source_observation_id: str
    created_step: int
    expires_after_step: Optional[int] = None


class RunWorkingMemory:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def advance_step(self) -> None:
        self._step += 1
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_after_step is not None and self._step > entry.expires_after_step
        ]
        for key in expired:
            self._entries.pop(key, None)

    def put(
        self,
        key: str,
        value: Any,
        *,
        value_kind: str,
        source_observation_id: str,
        ttl_steps: Optional[int] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            entry_id="memory_" + sha256_json(
                {
                    "key": key,
                    "value_kind": value_kind,
                    "value": value,
                    "source": source_observation_id,
                    "step": self._step,
                }
            ),
            key=str(key),
            value_kind=str(value_kind),
            value=value,
            source_observation_id=str(source_observation_id),
            created_step=self._step,
            expires_after_step=(self._step + ttl_steps if ttl_steps is not None else None),
        )
        self._entries[key] = entry
        return entry

    def get(self, key: str) -> Optional[MemoryEntry]:
        return self._entries.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "screenghost_run_working_memory_v1",
            "step": self._step,
            "entries": [asdict(entry) for entry in sorted(self._entries.values(), key=lambda e: e.key)],
        }
