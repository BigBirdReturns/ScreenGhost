import pytest

from drivers import AndroidAdbDriver, RemoteHandsError, is_network_target


def test_network_target_detection():
    assert is_network_target("192.168.1.5:5555")
    assert is_network_target("phone.local:5555")
    assert not is_network_target("emulator-5554")
    assert not is_network_target("R58Mxxxxxxx")  # typical USB serial
    assert not is_network_target(None)
    assert not is_network_target("")


def test_local_only_by_default_blocks_network_hands():
    driver = AndroidAdbDriver()
    with pytest.raises(RemoteHandsError):
        driver.tap(10, 10, device="192.168.1.5:5555")
    with pytest.raises(RemoteHandsError):
        driver.type_text("hi", device="192.168.1.5:5555")
    with pytest.raises(RemoteHandsError):
        driver.keyevent(4, device="192.168.1.5:5555")


def test_allow_network_opt_in_skips_guard():
    driver = AndroidAdbDriver(allow_network=True)
    # Guard must not fire; failure now would be a real adb call, not our guard.
    driver._guard_local("192.168.1.5:5555")
