# ScreenGhost DTI Cartridge v0

## Classification

This package is a domain cartridge over ScreenGhost's visible-surface execution floor. It is intended for family co-play and Freeplay practice in Roblox Dress to Impress through the official, unmodified Windows client. It is not an exploit, injected script, modified client, account farmer, voting bot, or unattended public-server agent.

The cartridge currently establishes the software floor required before a physical round can be claimed. It does not claim that a live DTI round has already been completed on the target Windows machine.

## Deployment decision

The live path is the official Windows Roblox client. MEmu remains useful only as a deterministic ScreenGhost fixture because Roblox restricted general player access from third-party emulators on August 7, 2025. The existing Semantic Multibox evidence therefore remains valid for ScreenGhost's motor and settlement architecture, while DTI receives a Windows-native driver.

Freeplay is the first venue. It removes the ordinary round timer, supports an on-demand runway, and now exposes Quick Teleport between named locations. Those mechanics convert most of the initial 3D navigation problem into bounded semantic menu transitions. Competitive public-server play remains a later, human-directed co-pilot mode.

## Actors and authority

```text
Human director
    chooses the session, can approve or override the theme interpretation,
    remains present, and owns F12 emergency stop

DTI cartridge
    classifies visible round state, resolves the theme, selects a feasible
    outfit from the local wardrobe atlas, and emits bounded action requests

ScreenGhost controller
    evaluates policy, admits one action, injects it once, settles, verifies
    the visible postcondition, and creates an idempotent receipt

Windows game driver
    captures the visible Roblox client with DXcam and emits ordinary Win32
    SendInput events only while the verified client window is foreground

Roblox / Dress to Impress
    owns the game, map, UI, wardrobe content, scoring, updates, and account
```

The cartridge cannot authorize its own policy exception. The observer cannot inject an action. The motor cannot infer intent. Community databases cannot become action authority. A model score cannot substitute for a visible postcondition.

## Operational modes

`family_copilot` is the product mode. The child or parent directs the look while ScreenGhost absorbs navigation, searching, repetitive selection, and evidence capture. `assisted` permits a human-authored plan to run through bounded actions. `autonomous` is admitted only for attended Freeplay or a private practice context. Autonomous public-server play and automated voting are structurally refused.

The policy also refuses emulators, modified clients, process injection, process-memory access, multi-account operation, reward farming, anti-idle behavior, chat automation, rejoin loops, captcha handling, and unattended execution.

## Durable floor

### Visible round contract

`RoundSignals` is the replaceable perception boundary. OCR, a visual index, a small GUI grounder, or a human annotation can supply visible text, labels, timer state, and frame hashes. `PhaseClassifier` converts those signals into `DTIObservation`. `DTIRoundTracker` debounces transitions, rejects illegal phase jumps into recovery, and halts when the unknown-screen budget is exhausted.

### Theme contract

`ThemeCatalog` normalizes OCR text and resolves it against compact `ThemeCard` records with a confidence and cross-candidate margin. Weak or ambiguous themes remain unresolved. The included seed is a generic style kernel, not a claim that it is the current official DTI theme list.

Community theme databases can be imported into the same schema. Their churn does not alter the planner or receipt format.

### Wardrobe contract

`WardrobeAtlas` stores locally taught items by semantic item ID, slot, style tags, palettes, access tier, named station, and route cost. It stores no durable action coordinate. The runtime must resolve each named target from current pixels.

`OutfitPlanner` selects standard, code, or event items by default, excludes VIP and removed content unless explicitly admitted, charges a station once rather than once per item, and emits a deterministic content-addressed `OutfitPlan`.

### Motor contract

`WindowsGameDriver` uses DXcam 0.3 for visible client capture and ordinary Win32 `SendInput` events for bounded keyboard and mouse actions. Before every action it verifies:

1. Windows is the host.
2. F12 is not held.
3. one visible window matches the declared Roblox title pattern;
4. that window is foreground;
5. its process name is in the official-client allowlist;
6. its client dimensions match the enrolled profile within tolerance; and
7. the full client rectangle lies on DXcam's admitted primary display output.

The driver does not activate a background window or silently redirect input. Focus drift is a refusal. The v0 DXcam adapter also requires the entire Roblox client area to remain on the Windows primary display, which prevents virtual-screen coordinates from being misapplied to another capture output.

### Transaction contract

`BoundedGameController` admits one action at a time. Every request requires an idempotency key. A duplicate request returns the original receipt rather than reinjecting the action. A second action is refused while the first is pending. Commit requires a visible postcondition, any required visible change, and stable observations over the settlement window.

## Current validation

The v0 package carries deterministic tests for:

- exact and OCR-degraded theme resolution;
- weak-theme refusal;
- access-tier and route-cost outfit planning;
- content-addressed plan determinism;
- Freeplay, public co-pilot, emulator, modified-client, voting, and unsafe-key policy decisions;
- phase classification, unknown-screen halt, and illegal-transition recovery;
- single-flight action settlement, duplicate idempotency, pending-overlap refusal, and policy-block receipts;
- profile validation and Windows import safety;
- end-to-end visible theme interpretation into a local-atlas outfit plan;
- policy distinction between automated `give_stars` and legitimate star-themed clothing; and
- primary-output custody configuration for the Windows capture path.

The permanent workflow runs the cartridge on Ubuntu and Windows across Python 3.11 and 3.13. The current local floor is 28 deterministic tests, successful bytecode compilation, a content-addressed example-profile validation, and an OCR-degraded theme resolution fixture.

## First physical campaign

The streamlined Windows path is:

```powershell
# Fresh checkout, after launching the official Roblox client and entering DTI Freeplay:
.\PREFLIGHT_DTI.cmd -Bootstrap

# Later runs:
.\PREFLIGHT_DRI.cmd
```

`PREFLIGHT_DTI.cmd` writes a timestamped `doctor.json`, `capture.json`, `freeplay-entry.png`, and content hash under `artifacts\dti\evidence`. It proves the live eyes and target-window custody only.

The equivalent manual setup, from the branch that contains PR #13 and this cartridge, is:

```powershell
py -3.13 -m venv artifacts\dti\venv
.\artifacts\dti\venv\Scripts\Activate.ps1
python -m pip install -r requirements-generic-utility.txt
python -m pip install -r requirements-dti.txt

python -m experiments.dti validate-profile configs/dti/profile.example.json
```

Launch Dress to Impress through the official Windows Roblox client, enter Freeplay, set the Roblox client area to `1600x900`, keep it foreground, and run:

```powershell
python -m experiments.dti doctor configs/dti/profile.example.json
python -m experiments.dti capture `
  configs/dti/profile.example.json `
  artifacts/dti/evidence/freeplay-entry.png
```

A passing doctor and captured client image prove the live eyes and target-window custody. They do not yet prove a round.

The next evidence pack must then enroll current Freeplay observations and one child- or parent-demonstrated procedure for:

```text
Freeplay entry
    -> Quick Teleport
    -> Dressing Room
    -> equip one taught base item
    -> equip one taught hair item
    -> Quick Teleport
    -> Freeplay Runway
    -> Walk the Runway
    -> return to Freeplay
```

Each transition must carry a before image, after image, resolved target evidence, controller receipt, settlement time, and refusal result for a moved or absent target. The first accepted product claim is one complete attended Freeplay procedure with no hidden teacher reads at runtime and no reinjected action.

## File map

```text
experiments/dti/schema.py          domain and receipt contracts
experiments/dti/policy.py          executable co-play boundary
experiments/dti/theme_kernel.py    theme and wardrobe commodity interface
experiments/dti/rounds.py          phase tracker and bounded controller
experiments/dti/profile.py         versioned surface and wardrobe profile
experiments/dti/windows_driver.py  DXcam and SendInput live adapter
experiments/dti/cartridge.py       thin domain composition
experiments/dti/cli.py             validate, resolve, doctor, capture
configs/dti/profile.example.json   unqualified enrollment template
tests/dti/                         deterministic software gates
```

## Control question

Can one attended Freeplay demonstration compile into a coordinate-free DTI procedure that replays through the official Windows client, survives current-screen variation, completes the on-demand runway, and either verifies every transition once or stops without guessing?
