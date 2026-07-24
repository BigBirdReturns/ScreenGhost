"""Generic A -> structure -> B temporal teaching orchestrator.

Source adapters provide only two read methods: capture pixels and inspect the
underlying structure.  This orchestrator captures bounded triples, certifies the
whole burst with :mod:`core.surface_alignment`, selects the representative frame,
and calls the existing Surface Teacher compiler through an injected callback.

The callback seam lets PR #13 retain its current ``compile_lesson`` authority
while Android, DOM, CDP, emulator, or fixture sources share one temporal gate.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

from core.surface_alignment import (
    AlignmentArtifact,
    AlignmentNode,
    AlignmentPolicy,
    FrameObservation,
    certify_alignment,
)


class TemporalTeacherRefused(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class StructureSnapshot:
    source_digest: str
    alignment_nodes: Tuple[AlignmentNode, ...]
    compiler_nodes: Any
    event_idle: bool = True
    source_payload: Optional[Any] = None

    def __post_init__(self) -> None:
        if len(self.source_digest) != 64:
            raise TemporalTeacherRefused("source_digest must be a 64-character SHA-256")
        try:
            int(self.source_digest, 16)
        except ValueError as exc:
            raise TemporalTeacherRefused("source_digest is not hexadecimal") from exc
        if not self.alignment_nodes:
            raise TemporalTeacherRefused("structure snapshot has no alignment nodes")


@runtime_checkable
class TemporalTeacherSource(Protocol):
    def capture_png(self) -> bytes: ...

    def inspect_structure(self) -> StructureSnapshot: ...


@dataclass(frozen=True)
class TemporalTeachingResult:
    alignment: AlignmentArtifact
    compiled_lesson: Any
    representative_structure_digest: str
    burst_source_digest: str


MappingLike = dict[str, Any]
CompilerCallback = Callable[[bytes, Any, str, MappingLike], Any]


def teach_temporal_surface(
    source: TemporalTeacherSource,
    compiler: CompilerCallback,
    *,
    burst_count: int = 3,
    interval_ms: float = 90.0,
    alignment_policy: AlignmentPolicy = AlignmentPolicy(),
    monotonic_ms: Callable[[], float] = lambda: time.monotonic() * 1000.0,
    sleep: Callable[[float], None] = time.sleep,
) -> TemporalTeachingResult:
    """Capture, certify, and compile one temporal surface lesson.

    The compiler receives ``(representative_png, compiler_nodes,
    burst_source_digest, alignment_certificate_dict)``.
    """

    if burst_count < 2:
        raise TemporalTeacherRefused("burst_count must be at least two")
    if interval_ms < 0:
        raise TemporalTeacherRefused("interval_ms cannot be negative")

    observations: list[FrameObservation] = []
    snapshots_for_observation: list[StructureSnapshot] = []
    source_digests: list[str] = []
    for index in range(burst_count):
        before = source.capture_png()
        before_time = monotonic_ms()
        snapshot = source.inspect_structure()
        after = source.capture_png()
        after_time = monotonic_ms()
        source_digests.append(snapshot.source_digest)
        observations.extend(
            [
                FrameObservation(
                    png_bytes=before,
                    nodes=snapshot.alignment_nodes,
                    observed_monotonic_ms=before_time,
                    source_digest=snapshot.source_digest,
                    event_idle=snapshot.event_idle,
                    note=f"burst-{index}-before",
                ),
                FrameObservation(
                    png_bytes=after,
                    nodes=snapshot.alignment_nodes,
                    observed_monotonic_ms=after_time,
                    source_digest=snapshot.source_digest,
                    event_idle=snapshot.event_idle,
                    note=f"burst-{index}-after",
                ),
            ]
        )
        snapshots_for_observation.extend((snapshot, snapshot))
        if index + 1 < burst_count and interval_ms:
            sleep(interval_ms / 1000.0)

    alignment = certify_alignment(observations, alignment_policy)
    representative_index = alignment.certificate.representative_index
    representative_snapshot = snapshots_for_observation[representative_index]
    burst_payload = {
        "alignment_certificate_id": alignment.certificate.certificate_id,
        "source_digests": source_digests,
        "representative_source_digest": representative_snapshot.source_digest,
    }
    burst_digest = _sha256(_json_bytes(burst_payload))
    compiled = compiler(
        alignment.representative_png,
        representative_snapshot.compiler_nodes,
        burst_digest,
        alignment.certificate.to_dict(),
    )
    return TemporalTeachingResult(
        alignment=alignment,
        compiled_lesson=compiled,
        representative_structure_digest=representative_snapshot.source_digest,
        burst_source_digest=burst_digest,
    )
