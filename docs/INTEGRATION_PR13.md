# Integration on top of ScreenGhost PR #13

## 1. Add the research graft

Apply `SCREEN_GHOST_SURFACE_TEACHER_V1_RESEARCH.patch` on branch
`agent/surface-teacher-v0` and run the verifier.

## 2. Convert PR teacher nodes to alignment nodes

Both current source adapters already return `TeacherNode` objects. Add one pure
conversion helper:

```python
from core.surface_alignment import AlignmentNode


def alignment_node(node):
    return AlignmentNode(
        semantic_key=node.source_ref,
        role=node.role,
        bounds=(node.bounds.x1, node.bounds.y1, node.bounds.x2, node.bounds.y2),
        label=node.label,
        interactive=node.interactive,
        enabled=node.enabled,
        parent_key=node.parent_ref,
        states=node.states,
        dynamic=False,
    )
```

Noninteractive nodes whose label or state changes across the burst are promoted to
dynamic automatically. Source adapters may explicitly mark known clock, progress,
video, or animation nodes as dynamic.

## 3. Wrap Android and browser sources

Implement the `TemporalTeacherSource` protocol for each source:

```python
class AndroidTemporalSource:
    def capture_png(self) -> bytes:
        image = driver.screencap(device)
        return image_png(image)[0]

    def inspect_structure(self) -> StructureSnapshot:
        xml = driver.dump_ui_xml(device)
        nodes = parse_uiautomator_xml(xml, viewport=current_size)
        return StructureSnapshot(
            source_digest=sha256_text(xml),
            alignment_nodes=tuple(alignment_node(node) for node in nodes),
            compiler_nodes=nodes,
            event_idle=optional_idle_probe(),
        )
```

The browser wrapper follows the same shape with `page.screenshot()` and
`parse_dom_snapshot()`. A later Chromium-only wrapper can replace the JavaScript
walker with `DOMSnapshot.captureSnapshot` while returning the same source-neutral
nodes.

## 4. Keep the existing compiler authoritative

Pass a callback that delegates to PR #13's `compile_lesson`:

```python
def compile_callback(png, nodes, burst_digest, certificate):
    artifact = compile_lesson(
        png,
        surface_id=surface_id,
        source_kind=source_kind,
        source_payload_sha256=burst_digest,
        nodes=nodes,
        policy=lesson_policy,
        app_version=app_version,
        locale=locale,
    )
    stage_alignment_certificate(certificate, lesson_directory)
    return artifact
```

The alignment certificate should be staged beside the lesson and hashed into the
lesson bundle manifest. The existing runtime projection does not need the
volatility mask or source receipt.

## 5. Compile curriculum after the lesson

```python
curriculum = build_curriculum(
    artifact.png_bytes,
    artifact.lesson.runtime_projection(),
)
stage_curriculum(curriculum, lesson_directory / "curriculum", source_png=artifact.png_bytes)
```

Keep curriculum generation optional for ordinary capture, but deterministic when
enabled.

## 6. Record graph transitions only after external action receipts

The teacher never executes the action. After the current procedure or future
transactional navigator returns its receipt:

```python
transition = make_transition(
    before.lesson.runtime_projection(),
    ActionDescriptor(
        action_type="tap",
        target_key=target.element_id,
        target_role=target.role,
        target_label=target.label,
    ),
    controller_receipt_id=procedure_receipt.receipt_id,
    outcome="verified",
    verified=True,
    after_projection=after.lesson.runtime_projection(),
    settlement_ms=procedure_receipt.settlement_ms,
    postcondition=procedure_receipt.postcondition,
)
graph.record(
    before.lesson.runtime_projection(),
    transition,
    after_projection=after.lesson.runtime_projection(),
)
```

## 7. Evaluate before navigator integration

Run the local visual student over held-out screenshots. Give its prediction and
declared evidence sources to `evaluate_prediction`. Do not let the navigator use
the atlas until held-out screen-family and grounding metrics are known.

## 8. Live receipt matrix

The next merge evidence should contain:

| Receipt | Required result |
|---|---|
| Android static screen | certificate accepted; correct normalized controls |
| Android clock or spinner | dynamic mask localizes motion; controls remain stable |
| Android moved control | alignment refused |
| Browser DPR 2 | **locally passed:** full-screen control normalized to `[0,0,1,1]` |
| Browser caret or animation | **locally passed:** changing clock localized to 0.399402% volatility |
| Browser overlay | occluded target classified as not receiving events |
| Pixel-only held-out screen | screen-family and grounding scores reported |
| Privileged leakage negative control | score forced to zero |
