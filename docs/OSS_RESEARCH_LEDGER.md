# OSS and practitioner research ledger

This ledger records the mechanisms used in the research graft. Project claims are
attributed to their own documentation or papers. Reddit observations are treated
as anecdotal implementation reports, not benchmark evidence.

## AutoDroid

Sources:

- https://autodroid-sys.github.io/
- https://github.com/MobileLLM/AutoDroid

Mechanism adopted: offline app exploration, UI Transition Graph, app-specific
memory, simulated task generation, and online memory-augmented execution. The
important transfer is that app knowledge should be compiled before the live
stepwise loop.

## AppAgent

Source:

- https://github.com/TencentQQGYLab/AppAgent

Mechanism adopted: autonomous exploration or human demonstration, inspectable UI
element documentation, and numeric marks over visible controls. Surface Teacher's
set-of-marks curriculum is a source-neutral version of that teaching interface.

## Agent+P and GraphPilot

Sources:

- https://arxiv.org/abs/2510.06042
- https://arxiv.org/abs/2601.17418

Mechanism adopted: treat known UI topology as a planning graph. The LLM should not
spend a query rediscovering a path already represented in the graph. Failed action
attempts remain in reliability denominators.

## Playwright

Source:

- https://playwright.dev/docs/actionability

Mechanism adopted: actionability is factored into visibility, stability, enabled
state, and whether the element receives events. Stability is local to an element's
bounding box, not global screenshot equality.

## Android UI Automator

Sources:

- https://developer.android.com/reference/androidx/test/uiautomator/UiDeviceExt
- https://developer.android.com/reference/android/app/UiAutomation

Mechanism adopted: source-level hierarchy stability, optional screenshot stability,
and accessibility-event idle evidence. The APIs also explicitly warn that
constantly changing content such as video prevents stable-screenshot completion,
which motivates the volatility mask.

## Chrome DevTools DOMSnapshot

Source:

- https://chromedevtools.github.io/devtools-protocol/tot/DOMSnapshot/

Mechanism adopted: flattened DOM and layout snapshots, iframe and shadow-DOM
coverage, rectangles, text boxes, clickability, and paint order. This is a stronger
browser teacher than a selector-oriented JavaScript walk, but remains optional and
Chromium-specific.

## OmniParser

Source:

- https://github.com/microsoft/OmniParser

Mechanism adopted: separate interactable-region detection and icon description,
plus local trajectory logging for training data. The visual fallback should be a
specialized parser rather than a general scene narrator.

## ShowUI and FOCUSUI

Sources:

- https://github.com/showlab/ShowUI
- https://showlab.github.io/FocusUI/

Mechanism adopted: a small GUI-specific grounding model as the ordinary visual
escalation, and instruction-conditioned token selection to avoid paying for every
high-resolution patch. Model selection remains a local benchmark question.

## Airtest and Poco

Sources:

- https://github.com/AirtestProject/Airtest
- https://github.com/AirtestProject/Poco

Mechanism adopted: image recognition and runtime hierarchy are complementary
planes, especially for game-like or custom-rendered surfaces. Normalized
coordinates are the stable interchange format.

## MobileUse and curiosity-driven AppCards

Sources:

- https://arxiv.org/abs/2507.16853
- https://arxiv.org/abs/2601.19306

Mechanism adopted: exploration is proactive, reflection is triggered by uncertainty,
and retrieved app knowledge should be packaged into small task-relevant cards
instead of flooding every prompt with the entire history.

## Practitioner sweep

Searches across Reddit and local-model communities were noisy, but several
consistent operational reports were useful:

- one accelerator model resident at a time is more reliable on bounded VRAM;
- local OpenAI-compatible endpoints can stall even when health checks remain green,
  so inference requires hard run-scoped timeouts and retry receipts;
- accessibility-first agents still need screenshot fallback for Flutter, games,
  canvas, WebView, or empty trees;
- unchanged-frame repetition needs an explicit stuck detector rather than another
  blind click.

These observations influenced the run-scoped queue and escalation policy. They do
not establish model accuracy or product reliability.
