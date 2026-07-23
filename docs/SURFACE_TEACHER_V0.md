# ScreenGhost Surface Teacher v0

## Classification

Surface Teacher is a read-only observation compiler for ScreenGhost. It pairs the
rendered surface the user can see with a privileged structural description of the
same surface, then compiles that pair into a deterministic lesson. On Android, the
privileged description is a UI Automator hierarchy. On the web, it is a DOM and
accessibility snapshot produced by a caller-owned page. Neither source has action
authority in this module.

The module exists to teach ScreenGhost what ordinary phone and browser surfaces
mean. It does not preserve a pixel-only fiction during teaching. It preserves the
more important boundary: privileged evidence may explain, label, score, and diagnose
a visible surface, but it may not silently select or authorize an input in a mode
that claims otherwise.

## The three artifacts

One aligned capture compiles into three distinct objects:

1. `teacher_lesson.json` is the inspectable teacher record. It carries the pixel
   evidence identity, the declared source kind and payload hash, source locators,
   structural roles, exposed labels, geometry, state, and crop fingerprints.
2. `runtime_projection.json` is the teacher-blind memory object. It retains learned
   roles, visible labels, normalized geometry, visual fingerprints, and structural
   relationships. It omits source locators, source node IDs, raw widget types,
   label provenance, raw field values, and the privileged source payload.
3. `training.jsonl` contains image-to-explanation and image-to-grounding examples
   suitable for a local VLM training, distillation, retrieval, or evaluation
   pipeline. The records contain the runtime projection, not the teacher locators.

The original `surface.png` is copied verbatim. Per-element crops are derived
teaching artifacts. `lesson_manifest.json` hashes every written file and states the
evidence tier and its limits.

## Stable identities

Surface Teacher separates four forms of change instead of treating every frame as
a new screen:

- `grammar_hash` covers roles, normalized geometry, interactivity, and containment.
- `control_hash` adds stable control labels to the grammar.
- `content_hash` covers visible labels, non-sensitive value fingerprints, states,
  and enabled status.
- `observation_hash` binds the current pixels, privileged-source digest, and content.

A `screen_key` is derived from the surface identity, grammar, and stable controls.
A changed account balance or typed value therefore produces a new observation and
content hash without invalidating the known screen. A renamed control produces
control drift. A role, geometry, interactivity, or containment change produces
structural drift.

`SurfaceAtlas` stores only runtime projections. Repeated observations of the same
screen update the content and observation count while preserving the stable screen
identity.

## Alignment gate

A lesson is valid only when its two planes refer to the same rendered state. The
live Android and browser adapters therefore use this bounded read sequence:

```text
pixel frame A
    -> privileged structure snapshot
    -> pixel frame B
    -> accept only when A and B have equal canonical pixels
```

A changing animation, navigation, or transient overlay causes an explicit refusal
after the configured number of attempts. The module does not guess which frame the
structure described and does not act to stabilize the surface.

## Android use

The existing `AndroidAdbDriver` already supplies the two read methods Surface
Teacher needs: `screencap()` and `dump_ui_xml()`. A one-shot live lesson can be
captured without exposing tap, swipe, text, or key methods to the compiler:

```python
from drivers import AndroidAdbDriver
from core.surface_teacher import SurfaceAtlas, stage_lesson
from core.teacher_sources import capture_android_lesson

artifact = capture_android_lesson(
    AndroidAdbDriver(),
    device="emulator-5554",
    surface_id="com.example/.Login",
)
stage_lesson(artifact, "log/teacher/login")
SurfaceAtlas("log/teacher/atlas.json").add(artifact.lesson)
```

The file-based demonstration does not require a connected device:

```text
python examples/surface_teacher_demo.py android-files \
  --png screen.png \
  --xml window_dump.xml \
  --surface-id com.example/.Login \
  --out log/teacher/login \
  --atlas log/teacher/atlas.json
```

A live one-shot capture uses:

```text
python examples/surface_teacher_demo.py android-live \
  --device emulator-5554 \
  --surface-id com.example/.Login \
  --out log/teacher/login \
  --atlas log/teacher/atlas.json
```

## Web use

A caller that already owns a browser page can call `capture_browser_lesson(page,
...)`. Surface Teacher neither launches a browser nor navigates it. The adapter
reads a PNG and evaluates the bundled read-only snapshot expression. An offline DOM
recording can be compiled with:

```text
python examples/surface_teacher_demo.py dom-files \
  --png page.png \
  --dom-json dom_snapshot.json \
  --surface-id https://example.test/login \
  --out log/teacher/web-login
```

The DOM JSON is a list of records accepted by `parse_dom_snapshot`. Each record may
carry a source reference, parent reference, tag or role, accessible name, visible
text, value, state object, interactivity, enabled status, and pixel bounds.

## Retention and privacy

Raw field values are redacted by default. Non-sensitive values receive a local
SHA-256 fingerprint and length so content changes can be detected without retaining
the string. Sensitive values, including password fields, receive no retained value,
hash, or length even when value retention is explicitly enabled. The runtime
projection never contains any value or value fingerprint.

The screenshot itself can still contain private information. A lesson bundle is a
local training artifact and should inherit the same custody, deletion, and access
rules as ScreenGhost pixel evidence. The privileged-source digest binds the source
used for teaching; it does not turn UI Automator or DOM data into vendor backend
truth.

## EarCrate lineage

The implementation follows the trial-and-error discipline established in EarCrate:

- expensive privileged inspection compiles once into a smaller sealed projection;
- the runtime consumes the compiled object rather than rescanning the raw source;
- source provenance and execution authority remain separate;
- failure is named and retained instead of repaired with a hidden fallback;
- negative controls prove the absence of input calls and privileged runtime fields;
- deterministic tests are the binding source of truth;
- the compiler is isolated before any navigator integration is attempted.

The direct analogy is EarCrate's compiled crate. A live set does not search the
private library on every phrase. ScreenGhost should likewise avoid rediscovering a
known phone screen on every frame. Surface Teacher performs the expensive under-hood
correlation once and emits the smaller object that later visual operation can use.

## Deliberate boundary

Surface Teacher v0 does not:

- click, tap, swipe, type, launch applications, navigate URLs, or authorize actions;
- run a listener, daemon, HTTP service, or persistent model server;
- call a VLM or claim that training has occurred merely because records were written;
- prove application backend state, platform truth, or legal-grade provenance;
- replace ScreenGhost's pending, settlement, semantic verification, and commit/abort
  transaction boundary;
- expose teacher locators to the runtime atlas.

## Verification

Run the executable gate directly:

```text
python -m pytest tests/test_surface_teacher.py -q
```

The gate covers parser honesty, Unicode, repeated Android resource IDs, deterministic
compilation, teacher/runtime separation, sensitive-value erasure, change
classification, atlas persistence, bundle hashes, read-only driver call logs, and
pixel/structure race refusal.
