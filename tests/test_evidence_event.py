"""Conformance: ScreenGhost's EvidenceEvent matches the GhostBox interop contract.

ScreenGhost emits the level-1 boundary object (EvidenceEvent) that GhostBox's
level-2 attention layer ingests. The canonical shape is defined by
ghostbox/interop/contracts.py. ScreenGhost mirrors the ID derivation instead of
importing it (peers across a seam, not a monolith), so this test pins that the
mirror is byte-for-byte identical to the contract.

The golden values below were produced by the GhostBox contract module itself for
the same explicit construction. If ScreenGhost's derivation ever drifts from the
contract, these assertions fail. That is the point: the contract wins, the mirror
is the bug.
"""
import re

from core.evidence_event import (
    EvidenceEvent,
    ProvenanceState,
    _HASH_TAG,
    _digest,
    canonical_bytes,
    evidence_from_observation,
)

# --- golden fixture, produced by ghostbox/interop/contracts.py ---------------

_OBS_GOLDEN = {
    "app": "Settings",
    "screen": "Display",
    "topic": "Settings.Display",
    "confidence": 0.8,
    "elements": [{"type": "toggle", "label": "Dark Mode", "value": "off"}],
}
_GOLDEN_ARGS = dict(
    source="screen_ghost",
    surface="screen",
    observation=_OBS_GOLDEN,
    captured_at="2025-12-10T03:45:00Z",
    raw_ref="log/screenshots/0001.png",
)
# canonical_bytes(identity) from the contract — algorithm-independent.
_GOLDEN_CANONICAL = (
    b'{"captured_at":"2025-12-10T03:45:00Z","observation":{"app":"Settings",'
    b'"confidence":0.8,"elements":[{"label":"Dark Mode","type":"toggle",'
    b'"value":"off"}],"screen":"Display","topic":"Settings.Display"},'
    b'"raw_ref":"log/screenshots/0001.png","source":"screen_ghost",'
    b'"surface":"screen"}'
)
# Full event_id. SHA-256 is pinned as a fixed contract constant, so this is the
# id in every environment (there is no optional-backend variance to branch on).
_GOLDEN_ID_SHA256 = "evt:sha256:c34679d7fefa95b85ce187ef5fa9ec6f"

_ID_RE = re.compile(r"^evt:sha256:[0-9a-f]{32}$")


def _identity(args):
    return {
        "source": args["source"],
        "surface": args["surface"],
        "observation": args["observation"],
        "captured_at": args["captured_at"],
        "raw_ref": args["raw_ref"],
    }


def test_canonicalization_is_byte_identical_to_contract():
    # The load-bearing cross-repo guarantee: identical canonical bytes. If this
    # holds and the digest rule holds, identical IDs follow by construction in
    # any environment, regardless of which hash backend is active.
    assert canonical_bytes(_identity(_GOLDEN_ARGS)) == _GOLDEN_CANONICAL


def test_event_id_is_tagged_digest_of_canonical_bytes():
    ev = EvidenceEvent(**_GOLDEN_ARGS)
    expected = f"evt:{_HASH_TAG}:{_digest(_GOLDEN_CANONICAL)[:32]}"
    assert ev.event_id == expected
    assert _ID_RE.match(ev.event_id)


def test_algorithm_is_pinned_to_sha256_no_optional_variance():
    # The pin is the whole point of P1: no environment-dependent backend. If this
    # regresses to an optional blake3 import, the same observation would hash
    # differently across machines and cross-repo correlation would silently break.
    assert _HASH_TAG == "sha256"


def test_matches_contract_golden():
    # Unconditional now: the same value the GhostBox contract emits, in every
    # environment, because both sides pin SHA-256.
    ev = EvidenceEvent(**_GOLDEN_ARGS)
    assert ev.event_id == _GOLDEN_ID_SHA256


def test_content_addressing_is_deterministic():
    a = EvidenceEvent(**_GOLDEN_ARGS)
    b = EvidenceEvent(**_GOLDEN_ARGS)
    assert a.event_id == b.event_id


def test_changing_any_identity_field_changes_the_id():
    base = EvidenceEvent(**_GOLDEN_ARGS).event_id
    moved = EvidenceEvent(**{**_GOLDEN_ARGS, "surface": "view_tree"}).event_id
    assert base != moved


def test_default_provenance_is_proven_capture_not_truth():
    # PROVEN asserts faithful capture, never that the content is accurate.
    ev = EvidenceEvent(**_GOLDEN_ARGS)
    assert ev.provenance is ProvenanceState.PROVEN


def test_mapper_lifts_capture_path_to_surface():
    # The trust of the extraction path must survive to the boundary, not be
    # flattened: api > view_tree > screen (vision) are distinct surfaces.
    for source_type, expected_surface in (
        ("api", "api"),
        ("view_tree", "view_tree"),
        ("photonic", "screen"),
    ):
        obs = {"source_type": source_type, "app": "Settings", "screen": "Display"}
        ev = evidence_from_observation(obs)
        assert ev.surface == expected_surface


def test_mapper_carries_observation_verbatim_and_uses_its_timestamp():
    obs = {
        "source_type": "photonic",
        "app": "Settings",
        "screen": "Display",
        "timestamp": "2025-12-10T03:45:00Z",
    }
    ev = evidence_from_observation(obs, raw_ref="log/screenshots/0009.png")
    assert ev.observation == obs
    assert ev.captured_at == "2025-12-10T03:45:00Z"
    assert ev.raw_ref == "log/screenshots/0009.png"
    assert ev.provenance is ProvenanceState.PROVEN
    assert _ID_RE.match(ev.event_id)
