# Surface Teacher v1: compile the phone, then operate the compiled utility

## Thesis

The first version of Surface Teacher correctly binds rendered pixels to
privileged structure and removes privileged locators from its runtime projection.
The next problem is temporal and behavioral. A mobile application is not a set of
independent screenshots. It is a recurring state machine whose screens contain a
small stable control grammar, variable content, overlays, asynchronous settling,
and learned transitions.

The implementation in this graft treats the teaching product as a compiler with
four outputs:

```text
observation burst
  -> temporal alignment certificate
  -> representative surface + volatility mask
  -> grounding curriculum
  -> screen family + transition graph
  -> hidden-teacher evaluation contract
```

## 1. Temporal alignment certificate

A single A -> structure -> B equality check refuses harmless clocks and animation.
A global similarity threshold can accept a serious local mismatch. The
certificate combines stricter local invariants:

- stable semantic node membership and role;
- unchanged labels and state for stable controls;
- bounded interactive-control movement;
- a localized volatility mask for declared or inferred dynamic regions;
- a maximum volatile-area fraction;
- a strict stable-region difference budget;
- a medoid frame selected from the burst;
- optional source event-idle evidence.

This borrows Playwright's actionability principle that an element is stable when
its bounding box remains unchanged across consecutive animation frames, while
extending it to a whole teacher snapshot. AndroidX UI Automator 2.4 now exposes
`waitForStableInActiveWindow`, and platform `UiAutomation.waitForIdle` exposes
accessibility-event quiet. Both can become source-specific evidence in the live
adapter.

A video or live clock is not treated as a stable image. Its changing pixels are
recorded as changing. Acceptance means that the dynamic area is bounded and the
rest of the taught interface is stable.

## 2. Grounding curriculum

The PR currently emits surface explanation and visible-element grounding. The
curriculum compiler expands each lesson into deterministic tasks that small GUI
models can actually learn:

- classify the screen family;
- explain the surface;
- extract visible controls;
- instruction to point;
- instruction to bounding box;
- numbered set-of-marks selection;
- element role and label from a crop;
- visible control state;
- same-screen, dynamic-content, control-drift, and structural-drift contrastive
  classification.

Coordinates use an integer 0..1000 space, so one record is independent of the
source resolution. Sensitive elements are omitted from the task set and crop
output.

The set-of-marks representation comes from the same practical insight used by
AppAgent and many GUI-agent systems: discrete visible candidate IDs reduce the
coordinate-generation burden while retaining a direct relation to the screen.
The raw point and box tasks remain available because a visual grounder must also
work when accessibility structure is absent.

## 3. Transition graph

Static lessons are compiled into screen families. External action receipts then
create directed transitions:

```text
before screen family
  + externally authorized semantic action
  + settlement receipt
  + visible postcondition
  -> after screen family
```

One action may have several outcomes. Failed attempts remain in the denominator,
so the planner sees reliability rather than a cleaned success-only history. The
path planner uses verified transitions, reliability, observed settlement time,
and declared action risk.

This follows AutoDroid's UI Transition Graph and app-memory pattern, then adopts
the stronger result from Agent+P: long-horizon UI work should be guided by a
symbolic route through the known graph rather than repeated local guessing. The
LLM selects or explains goals and resolves novelty. The graph handles already
learned topology.

## 4. Hidden-teacher evaluation

The teacher plane can score a runtime without becoming runtime evidence. The
evaluator compares a pixel-only prediction against the hidden runtime projection
and reports:

- screen-family match;
- element precision and recall;
- role and label accuracy;
- point-hit or box-IoU grounding accuracy;
- visible-state accuracy;
- privileged-source leakage.

A prediction that declares UI Automator, accessibility tree, DOM, CDP snapshot,
or teacher lesson as runtime evidence receives a zero score in teacher-blind mode.
The evaluator is a scoring boundary, not an action boundary.

## 5. 4060 perception policy

The local GPU should not run a broad VLM on each frame. The encoded cascade is:

```text
known atlas state
  -> stable crop/prototype retrieval
  -> small GUI-specific grounder
  -> larger VLM only for novelty or contradiction
  -> privileged teacher review only outside teacher-blind evaluation
```

The in-process queue is content-addressed, prioritizes interactive work over warm
work, and batches by model kind so one model can remain resident at a time. It is
constructed and drained by the current run. It has no daemon or listener.

ShowUI demonstrates that a 2B GUI-specialized model can be a viable grounding
component. OmniParser demonstrates the value of a dedicated interactable-region
and icon layer. FOCUSUI demonstrates that instruction-conditioned UI token
selection can retain a fraction of visual tokens while reducing memory and
latency. Those projects support a specialized grounding student rather than a
general descriptive VLM as the ordinary motor-perception loop.

## 6. Browser source upgrade

The PR's DOM walker is a valid first source and correctly applies
`devicePixelRatio`. A Chromium-specific advanced source can use CDP
`DOMSnapshot.captureSnapshot`, which provides flattened DOM, iframe and shadow-DOM
content, layout rectangles, text boxes, clickability, computed styles, and paint
order. Paint order and hit testing make it possible to label an element as
geometrically present but visually occluded.

CDP is a teacher source, not a runtime action source. The runtime projection still
removes backend node IDs and source locators.

## 7. Android source upgrade

The existing ADB `uiautomator dump` route remains useful and broadly deployable.
For controlled AVD teaching, an instrumentation companion using AndroidX UI
Automator 2.4 can return the stable hierarchy and screenshot as one source
receipt. It can also record active window, package, display, layer, and timeout
status. If the stable-screenshot requirement times out because of video, the
source can repeat the capture with hierarchy-only stability and let the temporal
certificate localize the volatile region.

## 8. Deliberate non-goals

This graft does not:

- execute or authorize input;
- claim that generated JSONL has trained a model;
- silently accept a surface after an alignment refusal;
- use privileged locators in the transition planner;
- start a local inference service;
- claim live AVD, physical-handset, or real vendor-application proof.

The real DPR-2 browser receipt now passes against local Chromium 144 and Playwright
1.57, including a changing clock localized to a 0.399402% volatility mask with zero
stable-region difference. The next decisive receipts are a real Android capture and
a teacher-blind local-model grounding evaluation.

## 9. Local model shortlist for the RTX 4060

The first benchmark should compare two 2B grounding students against the same
held-out Surface Teacher curriculum:

1. **UGround-V1-2B**, because it is purpose-built for point-based GUI grounding,
   has official Android-control evaluation paths, and is based on Qwen2-VL-2B.
2. **ShowUI-2B**, because it supports local inference, int8 quantization, navigation
   tasks, and iterative grounding refinement.

A 3B model such as GroundNext-3B or FOCUSUI-3B is a second benchmark if the 2B
models miss icons or spatial relations. A 7B model should remain the slow novelty
resolver until real VRAM, latency, and accuracy receipts justify promoting it.
OmniParser can supply an interactable-region and icon proposal layer, but its V2
icon detector inherits AGPL licensing, so distribution and product-use implications
must be reviewed before it becomes a shipped dependency.

The evaluation order is fixed:

```text
held-out teacher lessons
  -> quantized local model capability probe
  -> point-hit / box-IoU / state metrics
  -> latency and peak VRAM receipt
  -> only then navigator trial
```

No model wins because its repository reports a benchmark. It wins only if it
improves ScreenGhost's held-out phone corpus on the actual 4060.
