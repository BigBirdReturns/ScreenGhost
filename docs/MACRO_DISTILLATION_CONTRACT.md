# Macro distillation contract

A vendor macro is demonstration evidence. It is not a ScreenGhost skill until the
actions have been correlated with the visible surface and assigned explicit
postconditions.

## Accepted evidence

An imported macro must declare its source resolution and an ordered action stream.
Supported deterministic primitives are tap, long press, swipe, text reference,
Back or Home key, and wait. Loop constructs, key-state programs, random jitter,
undocumented commands, and ambiguous events are recorded as unsupported and stop
the compiler.

Text values are represented by a logical reference, byte length, and SHA-256 digest.
The macro record never persists the typed value.

## Target rule

For a pointed action, the teacher selects the smallest enabled interactive element
containing the point. If no such element exists, distillation refuses. It never
chooses the nearest element or moves the demonstration point.

A durable target contains semantic role, visible label when available, optional
containment hints, and a declared rule for unique-role fallback. Demonstration
coordinates remain in the receipt as evidence, but are not copied into runtime
target memory.

## Postcondition rule

A compiled step must carry one of:

- visible screen transition;
- visible element-state transition;
- bounded visible-content transition;
- explicit action-acceptance-only classification.

The runtime controller may commit only after its declared postcondition is stable.
Verification exceptions and unknown screens abort. They never imply progress.

## Baseline rule

Coordinate replay uses the leader's source pixels in every instance. It does not
normalize for density, orientation, or moved controls. This intentionally models the
fragility that vendor synchronizers warn about when instances do not share the same
geometry and state.

## Runtime rule

Semantic replay receives pixels and the compiled runtime memory. Teacher reads and
large-model calls are counted. The deterministic pass requires both to remain zero.
Each instance has its own pending transaction, idempotency key, settlement loop,
and visible verification result.
