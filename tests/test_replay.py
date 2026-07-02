import pytest

from drivers import ReplayDriver


def test_replay_records_actions_and_advances_frames():
    driver = ReplayDriver(frames=["f0.png", "f1.png", "f2.png"])
    assert driver.available()
    assert driver.list_devices() == ["replay-0"]
    assert driver.index == 0

    driver.tap(100, 200)
    driver.type_text("hello")
    assert driver.index == 2, "each action advances the world"
    assert driver.actions == [
        {"action": "tap", "x": 100, "y": 200},
        {"action": "type", "text": "hello"},
    ]


def test_replay_clamps_at_last_frame():
    driver = ReplayDriver(frames=["only.png"])
    driver.tap(1, 1)
    driver.swipe(0, 0, 10, 10)
    driver.keyevent(4)
    assert driver.index == 0, "single-frame trace never advances out of range"
    assert len(driver.actions) == 3


def test_replay_empty_is_unavailable():
    driver = ReplayDriver(frames=[])
    assert not driver.available()
    with pytest.raises(RuntimeError):
        driver.screencap()
