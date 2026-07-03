"""The Pop-proof demo runs, stays honest, and refuses to overclaim."""
import pytest

from examples.pop_proof_demo import (
    _require_boundary_failure, run_demo, seed_to_int,
)


def test_runs_fast_with_small_population():
    out = run_demo("ci-seed", sellers=10)
    assert "ACT I" in out and "ACT VII" in out


def test_accepts_arbitrary_seed_deterministically():
    assert seed_to_int("pop-picks-this-live") == seed_to_int("pop-picks-this-live")
    assert seed_to_int("a") != seed_to_int("b")
    a = run_demo("seed-A", sellers=10)
    assert "seed-A" in a  # operator's seed is echoed, not a hardcoded one


def test_output_carries_allowed_and_forbidden_claims():
    out = run_demo("claims", sellers=10)
    assert "ALLOWED" in out and "FORBIDDEN" in out


def test_exposes_a_boundary_failure():
    out = run_demo("boundary", sellers=10)
    assert "FAILS" in out  # at least one boundary condition failed openly


def test_hard_error_if_no_boundary_failure_exercised():
    with pytest.raises(RuntimeError, match="no boundary failure"):
        _require_boundary_failure([("all", True), ("green", True)])
    # a single real failure satisfies the requirement
    _require_boundary_failure([("ok", True), ("firehose", False)])


def test_does_not_claim_hardware_proof():
    out = run_demo("hw", sellers=10)
    assert "NOT live hardware proof" in out
    # device-dependent objections must not read PASS
    assert "[CENTRAL_IP_BLOCKING] ARCHITECTURAL" in out
    assert "[USER_INFRASTRUCTURE] ARCHITECTURAL" in out


def test_does_not_claim_business_proof():
    out = run_demo("biz", sellers=10)
    assert "[BUSINESS_OUTCOME] FROZEN" in out
    assert "no seller-hour claim" in out


def test_cha_mode_is_terse_and_factual_without_names():
    out = run_demo("cha", sellers=10, cha=True)
    assert "ten claims, five receipts, three frozen seams" in out
    for banned in ("Pop", "Prathan", "Page365", "Softbaked"):
        assert banned not in out  # systems argument, no person named
