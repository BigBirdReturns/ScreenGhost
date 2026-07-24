import io

from PIL import Image

from experiments.emulator_fleet.command import CommandResult
from experiments.emulator_fleet.live import LiveFleetInstanceAdapter
from experiments.emulator_fleet.schema import EmulatorVendor, InstanceRef


def _png(value=(20, 30, 40)):
    image = Image.new("RGB", (100, 200), value)
    out = io.BytesIO(); image.save(out, format="PNG")
    return out.getvalue()


class Provider:
    def __init__(self):
        self.png = _png()
        self.calls = []

    def capture_png(self, instance):
        self.calls.append(("capture", instance.instance_id)); return self.png

    def dump_ui_xml(self, instance):
        self.calls.append(("xml", instance.instance_id)); return "<hierarchy/>"

    @staticmethod
    def _ok(name):
        return CommandResult(name, (name,), 0, b"", b"", 0, 1)

    def tap(self, instance, x, y): self.calls.append(("tap", x, y)); return self._ok("tap")
    def swipe(self, instance, x1, y1, x2, y2, duration_ms=300): self.calls.append(("swipe", x1,y1,x2,y2,duration_ms)); return self._ok("swipe")
    def input_text(self, instance, text): self.calls.append(("text", text)); return self._ok("text")
    def keyevent(self, instance, key): self.calls.append(("key", key)); return self._ok("key")


def compiler(png, xml, surface_id, app_family):
    assert png == _png() and xml == "<hierarchy/>"
    return {
        "lesson_id": "lesson",
        "screen_key": "screen",
        "surface_id": surface_id,
        "app_family": app_family,
        "screen_name": "home",
        "elements": [],
    }


def test_live_adapter_is_instance_scoped_and_aligned():
    provider = Provider()
    ref = InstanceRef(EmulatorVendor.MEMU, "memu:2", "Clone", index=2)
    backend = LiveFleetInstanceAdapter(
        provider, ref, app_family="demo", teacher_compiler=compiler, sleep=lambda _: None
    )
    projection = backend.capture_teacher()
    assert projection["screen_name"] == "home"
    assert provider.calls[:3] == [("capture", "memu:2"), ("xml", "memu:2"), ("capture", "memu:2")]
    result = backend.tap_normalized(0.5, 0.25)
    assert result.injected
    assert ("tap", 50, 50) in provider.calls
    assert backend.actions_injected == 1


def test_live_adapter_refuses_misaligned_teacher_pair():
    provider = Provider()
    frames = iter([_png((1,2,3)), _png((4,5,6))] * 3)
    provider.capture_png = lambda instance: next(frames)
    ref = InstanceRef(EmulatorVendor.LDPLAYER, "ldplayer:0", "Leader", index=0)
    backend = LiveFleetInstanceAdapter(
        provider, ref, app_family="demo", teacher_compiler=compiler, sleep=lambda _: None
    )
    import pytest
    with pytest.raises(Exception, match="surface changed"):
        backend.capture_teacher()
