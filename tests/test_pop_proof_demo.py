"""The Pop-proof demo runs, stays honest, and refuses to pass by not testing."""
import pytest

from core.objections import ARCHITECTURAL, FROZEN, OBJECTIONS, PASS
from examples.pop_proof_demo import (
    DemoInvalid, _assert_no_forbidden_upgrade, _check_single_cause,
    _require_attribution_exercised, _require_boundary_failure, run_demo,
    seed_to_int,
)


def test_runs_fast_with_small_population():
    out = run_demo("ci-seed", sellers=10)
    assert "ACT I" in out and "ACT VII" in out


def test_act_one_ends_on_the_dare():
    out = run_demo("dare", sellers=10)
    act1 = out.split("ACT II")[0]
    # last real line of Act I (ignoring blank/separator lines) is the dare
    lines = [ln for ln in act1.splitlines() if ln.strip() and set(ln.strip()) != {"="}]
    assert lines[-1] == "Pick a seed. Pick a mode. Try to make it lie."


def test_act_four_echoes_operator_seed_as_your_results():
    seed = "operator-chose-this"
    out = run_demo(seed, sellers=10)
    assert f"FAILURE ATTRIBUTION  (seed: {seed}  — these are YOUR results)" in out


def test_accepts_arbitrary_seed_deterministically():
    assert seed_to_int("pop-picks-this-live") == seed_to_int("pop-picks-this-live")
    assert seed_to_int("a") != seed_to_int("b")
    assert "seed-A" in run_demo("seed-A", sellers=10)


def test_output_carries_allowed_and_forbidden_claims():
    out = run_demo("claims", sellers=10)
    assert "ALLOWED" in out and "FORBIDDEN" in out


def test_exposes_a_boundary_failure():
    assert "FAILS" in run_demo("boundary", sellers=10)


def test_boundary_guard_fails_if_all_green():
    with pytest.raises(DemoInvalid, match="no boundary failure"):
        _require_boundary_failure([("all", True), ("green", True)])
    _require_boundary_failure([("ok", True), ("firehose", False)])


def test_attribution_guard_fails_if_labels_present_but_no_bleed():
    with pytest.raises(DemoInvalid, match="no bleed"):
        _require_attribution_exercised(True, 0)
    _require_attribution_exercised(False, 0)   # nothing-to-test is fine
    _require_attribution_exercised(True, 5)    # bleed present is fine


def test_single_cause_guard_rejects_double_counting():
    with pytest.raises(DemoInvalid, match="multi-cause"):
        _check_single_cause(5, {"resolver": 3, "row_grouping": 3})  # sums to 6
    _check_single_cause(6, {"resolver": 3, "row_grouping": 3})       # ok


def test_no_forbidden_status_upgrade():
    # Even with maximally-favorable evidence, ARCH/FROZEN never read PASS.
    good = {"pop_corrupt": 0, "seam_corrupt": 0, "seam_delta": 0.0,
            "busy_keeps_up": True, "firehose_fails": True}
    _assert_no_forbidden_upgrade(good)  # must not raise
    for o in OBJECTIONS:
        if o.base_status in (ARCHITECTURAL, FROZEN):
            assert o.status(good) != PASS


def test_does_not_claim_hardware_proof():
    out = run_demo("hw", sellers=10)
    assert "NOT live hardware proof" in out
    assert "[CENTRAL_IP_BLOCKING] ARCHITECTURAL" in out
    assert "[USER_INFRASTRUCTURE] ARCHITECTURAL" in out


def test_does_not_claim_business_proof():
    out = run_demo("biz", sellers=10)
    assert "[BUSINESS_OUTCOME] FROZEN" in out
    assert "no seller-hour claim" in out


def test_cha_mode_ends_on_separation_verdict_no_scoreboard():
    out = run_demo("cha", sellers=10, cha=True)
    assert out.rstrip().endswith(
        "The objection named ten problems and asserted one verdict.\n"
        "Here are the ten, separated.\n"
        "The verdict does not survive the separation.")
    assert "five receipts" not in out      # scoreboard line deleted
    for banned in ("Pop", "Prathan", "Page365", "Softbaked"):
        assert banned not in out
