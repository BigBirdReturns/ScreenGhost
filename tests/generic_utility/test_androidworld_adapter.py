from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.generic_utility.androidworld_adapter import AndroidWorldBackend


@dataclass
class Box:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


@dataclass
class UI:
    text: str = ""
    content_description: str = ""
    hint_text: str = ""
    class_name: str = "android.widget.Button"
    bbox_pixels: Box | None = None
    is_checked: bool = False
    is_checkable: bool = False
    is_clickable: bool = True
    is_editable: bool = False
    is_enabled: bool = True
    is_focused: bool = False
    is_scrollable: bool = False
    is_selected: bool = False
    resource_name: str = "id/button"
    resource_id: str = ""


@dataclass
class State:
    pixels: np.ndarray
    ui_elements: list[UI]


class FakeEnv:
    logical_screen_size = (100, 200)
    foreground_activity_name = "com.example/.Main"

    def __init__(self):
        self.state = State(
            np.zeros((200, 100, 3), dtype=np.uint8),
            [UI(text="Continue", bbox_pixels=Box(10, 90, 100, 150))],
        )
        self.actions = []
        self.closed = False

    def reset(self, go_home=True):
        return self.state

    def get_state(self, wait_to_stabilize=False):
        return self.state

    def execute_action(self, action):
        self.actions.append(action)

    def close(self):
        self.closed = True


def test_projection_normalizes_androidworld_geometry():
    backend = AndroidWorldBackend(FakeEnv())
    png, projection, nodes = backend.teacher_snapshot()
    assert projection["width"] == 100 and projection["height"] == 200
    assert projection["elements"][0]["normalized_bounds"] == [0.1, 0.5, 0.9, 0.75]
    assert len(nodes) == 1 and png.startswith(b"\x89PNG")


def test_teacher_snapshot_counts_privileged_read():
    backend = AndroidWorldBackend(FakeEnv())
    backend.teacher_snapshot()
    assert backend.teacher_reads == 1


def test_pixel_capture_does_not_count_teacher_read():
    backend = AndroidWorldBackend(FakeEnv())
    backend.capture_png()
    assert backend.pixel_reads == 1 and backend.teacher_reads == 0


def test_close_delegates_to_environment():
    env = FakeEnv()
    backend = AndroidWorldBackend(env)
    backend.close()
    assert env.closed


def test_role_mapping_uses_editable_and_checkable_flags():
    env = FakeEnv()
    env.state.ui_elements = [
        UI(text="Name", class_name="android.widget.EditText", bbox_pixels=Box(0, 50, 0, 40), is_editable=True),
        UI(text="Wi-Fi", class_name="android.widget.Switch", bbox_pixels=Box(0, 50, 50, 90), is_checkable=True),
    ]
    projection = AndroidWorldBackend(env).teacher_snapshot()[1]
    assert [row["role"] for row in projection["elements"]] == ["text_field", "switch"]
