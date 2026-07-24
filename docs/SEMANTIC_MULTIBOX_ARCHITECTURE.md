# Semantic Multibox architecture

## Classification

Semantic Multibox is a local emulator-fleet experiment and optional execution
provider. It does not claim that vendor coordinate synchronization is safe. It uses
that behavior as the baseline against which ScreenGhost's semantic controller is
measured.

## Actors

The emulator vendor owns Android virtualization, instance creation, cloning,
configuration, graphics, and its own macro recorder. Surface Teacher owns the
read-only relationship between rendered pixels and the UI hierarchy during
teaching. ScreenGhost owns semantic target resolution and the transactional motor
boundary. The hidden evaluator may inspect the final hierarchy after a run, but
that score cannot authorize a runtime action.

## Authority planes

```text
Fleet lifecycle plane
    create, clone, start, stop, configure, inventory

Teacher plane
    screenshot, UI hierarchy, semantic labels, geometry, visible state

Runtime perception plane
    pixels, visual index, phone grammar, semantic procedure

Motor plane
    instance-scoped ADB tap, swipe, text, Back

Scoring plane
    final hidden-teacher validation and baseline comparison
```

The implementation keeps these authorities separate. A `SemanticProcedure` stores
roles, labels, expected states, and visible transitions. It never stores a durable
runtime coordinate. A `ResolvedAction` contains a short-lived normalized point
bound to the current visual observation.

## Distillation

The leader starts in a declared initial state. For each coordinate macro action:

1. Capture the teacher projection.
2. Find the visible enabled element under the demonstration point.
3. Inject the action once into the leader.
4. Wait until the emulator no longer reports a pending synthetic transition, or
   until the live timeout expires.
5. Capture the teacher projection again.
6. Compile the operator, target, and visible postcondition.

Text content is referenced by a logical key. The raw value is never written into
the macro record. Unsupported vendor commands stop distillation rather than being
silently skipped.

## Deterministic fleet

The emulated fleet intentionally separates content, geometry, and state variation.
Coordinate replay uses the leader's source pixels without normalization. Semantic
replay matches the current screen from pixels, resolves the target from the current
runtime projection, and verifies each postcondition before continuing.

A deceptive look-alike is a required negative control. Similar overall appearance
is insufficient. The visual index requires confidence, cross-family margin, and
interactive crop agreement; an exact pixel identity may bypass the margin only
when it maps unambiguously to one family.

## Vendor providers

MEmu and LDPlayer providers build argv arrays and always invoke subprocesses with
`shell=False`. Discovery is read-only. Lifecycle and Android input calls are
planned rather than executed unless `apply=True`.

MEmu's documented MEMUC interface supplies inventory, lifecycle, cloning,
configuration, application control, and a wrapped ADB command. MEmu Operation
Recorder `.mir` bytes are retained as opaque evidence because no documented
headless playback or stable action schema is claimed.

LDPlayer's documented console interface supplies lifecycle, cloning,
configuration, application control, and ADB. Its documented keyboard-macro syntax
is parsed into the common coordinate-macro contract.

BlueStacks supplies mature multi-instance, synchronization, macros, and resource
management through its supported user interface. This package parses exported
macro JSON conservatively but deliberately does not call undocumented lifecycle
executables.

## Measured plan

A measured plan requires one leader, at least one baseline instance, and at least
one semantic instance. It may also declare visual-teacher instances for geometry
variants. A visual teacher enrolls pixels and compiled projections before runtime;
it is never a measured baseline or semantic instance. All selectors must resolve
exactly and all sets must be disjoint. This design makes reset truth explicit: the
harness consumes independent clones instead of pretending it restored an
application after a destructive test.

## Control question

Does one coordinate demonstration compile into a semantic procedure that survives
content and layout divergence where coordinate replay fails, while each emulator
instance independently settles, verifies, and records exactly one action at a
time?
