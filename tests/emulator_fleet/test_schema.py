from experiments.emulator_fleet.macro import parse_ldplayer_macro
from experiments.emulator_fleet.schema import CoordinateMacro, SemanticProcedure
from experiments.emulator_fleet.campaign import record_coordinate_demonstration
from experiments.emulator_fleet.distill import distill_macro
from experiments.emulator_fleet.simulated import SimulatedFleetSpec, SimulatedInstanceAdapter
from experiments.generic_utility.phone_world import DisplayVariant


def test_coordinate_macro_roundtrip():
    macro = parse_ldplayer_macro("size 360 720\ntouch 20 40\nwait 500\nkey back\n")
    restored = CoordinateMacro.from_dict(macro.to_dict())
    assert restored == macro


def test_semantic_procedure_roundtrip():
    leader = SimulatedInstanceAdapter(
        SimulatedFleetSpec("leader", "Leader", DisplayVariant()), seed=4
    )
    macro, _snapshots = record_coordinate_demonstration(leader)
    leader.reset_task("settings_dark_mode")
    procedure, _receipt = distill_macro(macro, leader, app_family="settings")
    restored = SemanticProcedure.from_dict(procedure.to_dict())
    assert restored == procedure
