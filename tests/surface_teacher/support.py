from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image, ImageDraw

from core.surface_teacher import (
    LessonPolicy,
    Rect,
    SourceKind,
    TeacherNode,
    compile_lesson,
)


def png_bytes(*, button_fill=(30, 80, 180), value_fill=(230, 230, 230)) -> bytes:
    image = Image.new("RGB", (400, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 360, 140), outline="black", fill=value_fill)
    draw.rectangle((40, 180, 360, 250), outline="black", fill=button_fill)
    draw.rectangle((40, 300, 360, 370), outline="black", fill=value_fill)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def nodes(*, button_label="Continue", field_value="Jonathan", button_role="button", button_y=180):
    return (
        TeacherNode(
            source_ref="root",
            role="group",
            bounds=Rect(0, 0, 400, 800),
            raw_type="android.widget.FrameLayout",
        ),
        TeacherNode(
            source_ref="title",
            role="heading",
            bounds=Rect(40, 80, 360, 140),
            label="Welcome back",
            parent_ref="root",
            label_source="text",
        ),
        TeacherNode(
            source_ref="continue",
            role=button_role,
            bounds=Rect(40, button_y, 360, button_y + 70),
            label=button_label,
            interactive=True,
            parent_ref="root",
            states=(("focused", "false"),),
            label_source="text",
        ),
        TeacherNode(
            source_ref="name",
            role="text_field",
            bounds=Rect(40, 300, 360, 370),
            label="Name",
            value=field_value,
            interactive=True,
            parent_ref="root",
            label_source="content-desc",
        ),
    )


def compile_fixture(**kwargs):
    payload = kwargs.pop("payload", "teacher-source")
    return compile_lesson(
        kwargs.pop("png", png_bytes()),
        surface_id=kwargs.pop("surface_id", "com.example/.Login"),
        source_kind=kwargs.pop("source_kind", SourceKind.ANDROID_UIAUTOMATOR),
        source_payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        nodes=kwargs.pop("nodes", nodes()),
        policy=kwargs.pop("policy", LessonPolicy()),
        **kwargs,
    )


ANDROID_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="root" class="android.widget.FrameLayout"
        package="com.example" content-desc="" clickable="false" enabled="true"
        focusable="false" focused="false" scrollable="false" password="false"
        selected="false" checked="false" bounds="[0,0][400,800]">
    <node index="0" text="Welcome back" resource-id="title" class="android.widget.TextView"
          content-desc="" clickable="false" enabled="true" focusable="false"
          focused="false" scrollable="false" password="false" selected="false"
          checked="false" bounds="[40,80][360,140]" />
    <node index="1" text="Continue" resource-id="continue" class="android.widget.Button"
          content-desc="" clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" password="false" selected="false"
          checked="false" bounds="[40,180][360,250]" />
    <node index="2" text="Jonathan" resource-id="name" class="android.widget.EditText"
          content-desc="Name" clickable="true" enabled="true" focusable="true"
          focused="true" scrollable="false" password="false" selected="false"
          checked="false" bounds="[40,300][360,370]" />
    <node index="3" text="secret" resource-id="password" class="android.widget.EditText"
          content-desc="Password" clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" password="true" selected="false"
          checked="false" bounds="[40,400][360,470]" />
    <node index="4" text="Hidden" resource-id="hidden" class="android.widget.Button"
          content-desc="" clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" password="false" selected="false"
          checked="false" visible-to-user="false" bounds="[40,500][360,570]" />
  </node>
</hierarchy>
"""


DOM_RECORDS = [
    {
        "source_ref": "body",
        "tag": "body",
        "visible": True,
        "interactive": False,
        "bounds": {"x": 0, "y": 0, "width": 400, "height": 800},
    },
    {
        "source_ref": "#title",
        "parent_ref": "body",
        "tag": "h1",
        "text": "Welcome back",
        "visible": True,
        "interactive": False,
        "bounds": [40, 80, 320, 60],
    },
    {
        "source_ref": "#continue",
        "parent_ref": "body",
        "tag": "button",
        "accessible_name": "Continue",
        "text": "Continue",
        "visible": True,
        "interactive": True,
        "enabled": True,
        "bounds": [40, 180, 320, 70],
        "states": {"expanded": "false"},
    },
    {
        "source_ref": "#name",
        "parent_ref": "body",
        "tag": "input",
        "input_type": "text",
        "accessible_name": "Name",
        "value": "Jonathan",
        "visible": True,
        "interactive": True,
        "enabled": True,
        "bounds": [40, 300, 320, 70],
    },
    {
        "source_ref": "#display-none",
        "tag": "button",
        "accessible_name": "Do not learn me",
        "visible": False,
        "interactive": True,
        "bounds": [40, 500, 320, 70],
    },
]


def android_node(parsed_nodes, resource_id):
    return next(node for node in parsed_nodes if node.source_ref.startswith(f"{resource_id}@"))
