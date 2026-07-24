import json
from pathlib import Path

import pytest

from experiments.emulator_fleet.macro import (
    MacroParseError,
    catalog_memu_macros,
    parse_bluestacks_macro_json,
    parse_ldplayer_macro,
)
from experiments.emulator_fleet.schema import MacroActionKind


def test_ldplayer_macro_parses_and_redacts_text():
    macro = parse_ldplayer_macro(
        "size 360 720\ntouch 100 200\nwait 450\ntext secret value\nkey back\n"
    )
    assert macro.source_resolution == (360, 720)
    assert [row.kind for row in macro.actions] == [
        MacroActionKind.TAP,
        MacroActionKind.WAIT,
        MacroActionKind.TEXT,
        MacroActionKind.KEY,
    ]
    text = macro.actions[2]
    assert text.raw == "text <redacted>"
    assert text.text_length == len("secret value")
    assert "secret value" not in json.dumps(macro.to_dict())


def test_ldplayer_unknown_command_is_explicitly_unsupported():
    macro = parse_ldplayer_macro("size 100 100\nwobble 20 30\n")
    assert len(macro.unsupported_actions) == 1
    assert macro.unsupported_actions[0].kind is MacroActionKind.UNKNOWN


def test_ldplayer_rejects_out_of_bounds_point():
    with pytest.raises(MacroParseError):
        parse_ldplayer_macro("size 100 100\ntouch 101 20\n")


def test_bluestacks_common_json_shape_parses():
    macro = parse_bluestacks_macro_json(
        {
            "width": 360,
            "height": 720,
            "events": [
                {"type": "click", "x": 100, "y": 200},
                {"type": "wait", "duration": 300},
                {"type": "text", "text": "private"},
            ],
        }
    )
    assert [row.kind for row in macro.actions] == [
        MacroActionKind.TAP,
        MacroActionKind.WAIT,
        MacroActionKind.TEXT,
    ]
    assert "private" not in json.dumps(macro.to_dict())


def test_memu_macro_catalog_retains_opaque_artifact(tmp_path: Path):
    (tmp_path / "Demo.mir").write_bytes(b"opaque-memu-macro")
    (tmp_path / "info.ini").write_text(
        "[Demo]\nname=Demo flow\nreplayCycles=3\n", encoding="utf-8"
    )
    rows = catalog_memu_macros(tmp_path)
    assert len(rows) == 1
    assert rows[0]["name"] == "Demo flow"
    assert rows[0]["actions_available"] is False
    assert rows[0]["size_bytes"] == len(b"opaque-memu-macro")


def test_ldplayer_leading_comments_are_ignored():
    macro = parse_ldplayer_macro("# harmless demo\n// second comment\nsize 100 200\ntouch 50 100\n")
    assert len(macro.actions) == 1
    assert macro.actions[0].point == (0.5, 0.5)
