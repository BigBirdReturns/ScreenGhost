import json
from pathlib import Path

import pytest

from experiments.emulator_fleet.machine import MachinePlan, resolve_instance
from experiments.emulator_fleet.schema import EmulatorVendor, InstanceRef


def test_machine_plan_requires_disjoint_execution_sets_by_runner(tmp_path: Path):
    payload = {
        "schema": "semantic_multibox_machine_plan_v1",
        "vendor": "memu",
        "executable": "memuc.exe",
        "app_family": "demo",
        "macro_path": "demo.txt",
        "leader": {"name": "Leader"},
        "visual_teacher_instances": [{"name": "Density Teacher"}],
        "baseline_instances": [{"name": "Baseline"}],
        "semantic_instances": [{"name": "Semantic"}],
    }
    path = tmp_path / "plan.json"; path.write_text(json.dumps(payload))
    plan = MachinePlan.load(path)
    assert plan.vendor is EmulatorVendor.MEMU
    assert plan.leader.name == "Leader"
    assert plan.visual_teacher_instances[0].name == "Density Teacher"


def test_instance_resolution_is_exact():
    rows = (
        InstanceRef(EmulatorVendor.MEMU, "memu:0", "Leader", index=0),
        InstanceRef(EmulatorVendor.MEMU, "memu:1", "Clone", index=1),
    )
    from experiments.emulator_fleet.machine import InstanceSelector
    assert resolve_instance(rows, InstanceSelector(index=1)).name == "Clone"
    with pytest.raises(Exception):
        resolve_instance(rows, InstanceSelector(name="Missing"))
