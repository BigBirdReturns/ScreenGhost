from __future__ import annotations

import pytest

from core.surface_graph import (
    ActionDescriptor,
    GraphRefused,
    SurfaceTransitionGraph,
    make_transition,
)
from tests.surface_teacher_v1.support import runtime_projection


def record(graph, before, action, after=None, *, receipt, verified=True, outcome="verified", settle=300):
    transition = make_transition(
        before,
        action,
        controller_receipt_id=receipt,
        outcome=outcome,
        verified=verified,
        after_projection=after,
        settlement_ms=settle,
        postcondition={"kind": "visible_state"},
    )
    graph.record(before, transition, after_projection=after)


def test_graph_requires_external_controller_receipt():
    with pytest.raises(GraphRefused, match="controller_receipt_id"):
        make_transition(
            runtime_projection("a"),
            ActionDescriptor("tap"),
            controller_receipt_id="",
            outcome="execution_failed",
            verified=False,
        )


def test_symbolic_planner_uses_reliability_latency_and_risk():
    graph = SurfaceTransitionGraph()
    a, b, c = (runtime_projection(name) for name in ("a", "b", "c"))
    to_b = ActionDescriptor("tap", target_label="Next", risk=0.0)
    to_c = ActionDescriptor("tap", target_label="Finish", risk=0.0)
    direct = ActionDescriptor("tap", target_label="Dangerous shortcut", risk=0.9)

    record(graph, a, to_b, b, receipt="r1", settle=300)
    record(graph, a, to_b, b, receipt="r2", settle=500)
    record(
        graph,
        a,
        to_b,
        None,
        receipt="r3",
        verified=False,
        outcome="settlement_timeout",
        settle=1000,
    )
    record(graph, b, to_c, c, receipt="r4", settle=200)
    record(graph, a, direct, c, receipt="r5", settle=100)

    plan = graph.plan("a", "c", minimum_reliability=0.5)
    assert [step.to_screen_key for step in plan.steps] == ["b", "c"]
    assert plan.steps[0].supporting_successes == 2
    assert plan.steps[0].total_attempts == 3
    assert plan.steps[0].reliability == pytest.approx(2 / 3, rel=1e-5)


def test_graph_refuses_route_below_reliability_threshold():
    graph = SurfaceTransitionGraph()
    a, b = runtime_projection("a"), runtime_projection("b")
    action = ActionDescriptor("tap", target_label="Next")
    record(graph, a, action, b, receipt="ok")
    for index in range(3):
        record(
            graph,
            a,
            action,
            None,
            receipt=f"fail-{index}",
            verified=False,
            outcome="postcondition_failed",
        )
    with pytest.raises(GraphRefused, match="no verified route"):
        graph.plan("a", "b", minimum_reliability=0.5)


def test_graph_persists_teacher_blind_state(tmp_path):
    path = tmp_path / "graph.json"
    graph = SurfaceTransitionGraph(path)
    a, b = runtime_projection("a"), runtime_projection("b")
    action = ActionDescriptor("tap", target_label="Next")
    record(graph, a, action, b, receipt="r1")
    loaded = SurfaceTransitionGraph(path)
    assert loaded.state_count == 2
    assert loaded.transition_count == 1
    raw = path.read_text()
    assert "source_ref" not in raw
    assert "selector" not in raw
