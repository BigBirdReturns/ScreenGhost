"""The objection stack is complete, disciplined, and claim-bounded."""
from core.objections import (
    ARCHITECTURAL, FROZEN, OBJECTIONS, PARTIAL, PASS, gather_evidence,
)

_EXPECTED = {
    "THAI_TEXT_RELIABILITY", "VISION_LATENCY", "NON_TEXT_PAYLOADS",
    "LIVE_COMMERCE_BURST", "CENTRAL_IP_BLOCKING", "USER_INFRASTRUCTURE",
    "WINDOW_MANAGEMENT", "PARSER_GENERALIZATION", "APP_SURFACE_DRIFT",
    "BUSINESS_OUTCOME",
}


def test_every_original_objection_is_mapped():
    assert {o.id for o in OBJECTIONS} == _EXPECTED


def test_every_objection_states_its_claim_boundaries():
    # No objection may exist without an explicit allowed AND forbidden claim.
    for o in OBJECTIONS:
        assert o.allowed.strip()
        assert o.forbidden.strip()
        assert o.answer.strip() and o.next_proof.strip()
        assert o.base_status in {PASS, PARTIAL, ARCHITECTURAL, FROZEN}


def test_business_outcome_stays_frozen():
    biz = next(o for o in OBJECTIONS if o.id == "BUSINESS_OUTCOME")
    assert biz.base_status == FROZEN
    assert "revenue" in biz.forbidden.lower() or "seller-hour" in biz.forbidden.lower()


def test_device_dependent_objections_are_not_marked_pass():
    # Anything needing hardware must not claim PASS from in-process evidence.
    for oid in ("CENTRAL_IP_BLOCKING", "USER_INFRASTRUCTURE"):
        o = next(x for x in OBJECTIONS if x.id == oid)
        assert o.base_status == ARCHITECTURAL


def test_live_refinement_reflects_reality():
    # Cheap live checks: the seam is clean, so refinable objections pass.
    e = gather_evidence(n=30)
    assert e["pop_corrupt"] == 0
    assert e["seam_corrupt"] == 0
    assert e["seam_delta"] == 0.0
    assert e["busy_keeps_up"] and e["firehose_fails"]
    thai = next(o for o in OBJECTIONS if o.id == "THAI_TEXT_RELIABILITY")
    assert thai.status(e) == PASS
    assert thai.status(None) == PARTIAL  # no borrowed trust without evidence
