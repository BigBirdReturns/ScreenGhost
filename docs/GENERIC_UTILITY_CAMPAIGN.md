# ScreenGhost Generic Utility Campaign

## Purpose

This package turns Surface Teacher into an executable experiment. It tests whether expensive first-use teaching compiles into a cheaper, patient, teacher-blind warm path and whether the generic phone grammar transfers to an untaught application.

The default campaign is fully deterministic and requires no Android device, network service, model download, or host input. `PhoneWorld` renders multiple phone-shaped applications into PNG pixels, maintains a hidden structural teacher, accepts ordinary touch and text actions, models asynchronous transitions, and injects theme, font, density, orientation, content, control, layout, unknown-screen, and deceptive-look-alike variation.

## Experiment phases

### A. Cold teaching

Three task families are executed with privileged labels and explicitly simulated large-model costs:

1. Open Display, toggle Dark mode, save.
2. Open profile editing, enter a display name, save.
3. Start and stop a timer.

Each state produces a temporal alignment certificate, volatility mask, Surface Teacher projection, GUI curriculum, visual prototypes, semantic cache entry, graph transition, model receipt, and external controller receipt.

### B. Warm replay

The tasks restart from their reset states. The student receives pixels, visual memory, task intent, app graph, semantic decision cache, and generic phone grammar. UI structure and large-model planning are absent from action selection.

The transactional controller permits one pending action. It narrows recognition using only prior pixel state and the declared expected transition, waits for visible change and temporal stability, verifies the visible postcondition, and then commits or aborts.

### C. Cross-app holdout

A previously untaught Connectivity application presents a standard switch. The model-free emulator detector proposes a switch from pixels, and PhoneGrammar resolves the generic `toggle` operator without app-specific memory.

### D. Drift and novelty

The campaign exercises taught dark theme, dynamic timer content, moved controls, renamed controls, an unknown canvas, and a deceptive look-alike. Context-free novelty checks must not confidently map structural or semantic drift onto a taught screen.

### E. Patient execution

The campaign verifies pending-action exclusion, idempotent replay, no reinjection, visible postconditions, and a model timeout that terminates before the motor boundary.

## Run

On Windows, from the ScreenGhost repository root:

```bat
BOOTSTRAP_GENERIC_UTILITY.cmd -InstallChromium
RUN_GENERIC_UTILITY_CAMPAIGN.cmd
```

The bootstrap script resolves `python` first and honors `SG_PYTHON`; it does not assume that the Windows `py` launcher exists. The `.cmd` entrypoints invoke PowerShell with an explicit execution-policy bypass so a restrictive user policy does not make the documented first run fail.

Without a virtual environment:

```powershell
$env:PYTHONPATH = $PWD
python -m experiments.generic_utility doctor
python -m pytest -q tests/surface_teacher_v1 tests/generic_utility
python -m experiments.generic_utility emulate --out log/generic_utility/campaign
python -m experiments.generic_utility verify log/generic_utility/campaign
python -m experiments.generic_utility grounding-emulated --out log/generic_utility/grounding-emulated
python -m experiments.generic_utility browser-smoke --out log/generic_utility/browser
python -m experiments.generic_utility conclude `
  --campaign log/generic_utility/campaign `
  --browser log/generic_utility/browser/receipt.json `
  --grounding log/generic_utility/grounding-emulated/benchmark/benchmark_receipt.json `
  --out log/generic_utility/conclusion.json
```

## Evidence classification

The campaign uses three explicit metric classes:

- `simulated`: deterministic costs used to test routing, amortization, and accounting.
- `measured`: wall-clock or GPU measurements from an actual process.
- `derived`: values computed from recorded events.

The emulated campaign establishes orchestration and behavioral conclusions. It does not establish RTX 4060 latency, VRAM use, model accuracy, Android app compatibility, or physical-device transport. Those require the optional measured receipts described in the machine runbook.
