from pathlib import Path

import pytest

from experiments.emulator_fleet.command import RecordingCommandRunner
from experiments.emulator_fleet.providers.base import MutationRefused
from experiments.emulator_fleet.providers.ldplayer import (
    LDPlayerFleetProvider,
    parse_ldplayer_list2,
)
from experiments.emulator_fleet.providers.memu import (
    MEmuFleetProvider,
    parse_memuc_listvms,
)
from experiments.emulator_fleet.schema import (
    EmulatorVendor,
    InstanceProfile,
    InstanceRef,
    InstanceStatus,
)


def test_memuc_inventory_parser_accepts_csv_and_status():
    rows = parse_memuc_listvms('0,"Leader",1,1234,5678\n1,"Clone B",0,0,0\n')
    assert [row.name for row in rows] == ["Leader", "Clone B"]
    assert rows[0].status is InstanceStatus.RUNNING
    assert rows[1].status is InstanceStatus.STOPPED


def test_ldplayer_inventory_parser():
    rows = parse_ldplayer_list2("0,Leader,111,222,1,333,444\n1,Clone,0,0,0,0,0\n")
    assert rows[0].vendor is EmulatorVendor.LDPLAYER
    assert rows[0].status is InstanceStatus.RUNNING
    assert rows[1].status is InstanceStatus.STOPPED


def test_memu_mutations_are_planned_by_default():
    runner = RecordingCommandRunner()
    provider = MEmuFleetProvider("memuc.exe", runner=runner, apply=False)
    instance = InstanceRef(EmulatorVendor.MEMU, "memu:0", "Leader", index=0)
    result = provider.start(instance)
    assert result.planned_only
    assert runner.calls == []
    assert result.argv == ("memuc.exe", "start", "-i", "0")


def test_memu_profile_uses_resolution_keys_with_real_readback_support():
    runner = RecordingCommandRunner()
    provider = MEmuFleetProvider("memuc.exe", runner=runner, apply=False)
    instance = InstanceRef(EmulatorVendor.MEMU, "memu:3", "Clone", index=3)
    results = provider.configure(
        instance,
        InstanceProfile("portrait", 720, 1280, 240, cpu_cores=2, memory_mb=2048),
    )
    commands = [result.argv for result in results]
    assert ("memuc.exe", "setconfigex", "-i", "3", "resolution_width", "720") in commands
    assert ("memuc.exe", "setconfigex", "-i", "3", "resolution_height", "1280") in commands
    assert ("memuc.exe", "setconfigex", "-i", "3", "vbox_dpi", "240") in commands
    assert not any("custom_resolution" in command for command in commands)


def test_ldplayer_profile_command_is_instance_scoped():
    runner = RecordingCommandRunner()
    provider = LDPlayerFleetProvider("ldconsole.exe", runner=runner, apply=False)
    instance = InstanceRef(EmulatorVendor.LDPLAYER, "ldplayer:3", "Clone", index=3)
    result, = provider.configure(
        instance,
        InstanceProfile("p", 540, 960, 240, cpu_cores=2, memory_mb=1536),
    )
    assert result.planned_only
    assert "--index" in result.argv and "3" in result.argv
    assert "--resolution" in result.argv and "540,960,240" in result.argv


def test_destructive_remove_requires_token():
    runner = RecordingCommandRunner()
    provider = MEmuFleetProvider("memuc.exe", runner=runner, apply=True)
    instance = InstanceRef(EmulatorVendor.MEMU, "memu:1", "Clone", index=1)
    with pytest.raises(MutationRefused):
        provider.remove(instance)


def test_read_only_inventory_executes_without_apply():
    exe = "memuc.exe"
    runner = RecordingCommandRunner({(exe, "listvms"): (0, b"0,Leader,1,2,3\n", b"")})
    provider = MEmuFleetProvider(exe, runner=runner, apply=False)
    rows = provider.list_instances()
    assert len(rows) == 1 and rows[0].name == "Leader"
    assert runner.calls == [(exe, "listvms")]
