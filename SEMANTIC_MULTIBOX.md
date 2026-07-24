# ScreenGhost Semantic Multibox

Semantic Multibox turns emulator macro systems into a teaching and benchmark
surface for ScreenGhost.

A vendor macro establishes the coordinate-replay baseline. Surface Teacher watches
one demonstration with privileged UI structure, explains each action against the
visible screen, and compiles a semantic procedure. ScreenGhost then resolves that
procedure independently in every emulator instance, injects one action, waits for
that instance to settle, verifies the visible postcondition, and only then advances.

```text
vendor macro on leader
    -> pixels + hidden UI structure
    -> semantic procedure
    -> visual state index
    -> independent per-instance replay
    -> baseline versus semantic comparison
```

The deterministic campaign proves the architecture without requiring an emulator
installation. The optional machine plan is a compatibility and performance smoke
test over prepared MEmu or LDPlayer clones.

## Run the deterministic conclusion suite

```powershell
BOOTSTRAP_SEMANTIC_MULTIBOX.cmd
RUN_SEMANTIC_MULTIBOX_CAMPAIGN.cmd
```

The authoritative command is:

```powershell
python VERIFY_SEMANTIC_MULTIBOX.py `
  --out log/semantic_multibox/verification
```

The campaign records one three-action Settings macro, distills it, and compares
coordinate replay with semantic replay across exact clones, account-diverse clones,
dark mode, altered density, landscape, moved controls, renamed controls, an
occluding overlay, an unknown canvas, and a deceptive look-alike.

## Measured vendor run

Copy one of the example plans in `configs/emulator_fleet/`, replace the paths and
instance selectors, and prepare **disjoint clones** at the same initial application
state:

```powershell
RUN_SEMANTIC_MULTIBOX_MACHINE.cmd `
  -Plan configs/emulator_fleet/memu.local.json `
  -Apply
```

The leader is consumed by teaching. Baseline clones are consumed by raw coordinate
replay. Semantic clones are consumed by independent ScreenGhost replay. The harness
refuses a plan that reuses an instance across those sets because it cannot honestly
claim a reset it did not perform.

## Safety boundary

- No host mouse or keyboard injection.
- No listener, daemon, or local command API.
- Vendor lifecycle mutation is dry-run by default.
- Instance removal and other destructive operations require the literal token
  `SCREEN_GHOST_FLEET_MUTATION` in code.
- Runtime semantic replay cannot consult Surface Teacher.
- Every semantic action is single-flight, idempotent, settled, and visibly verified.
- BlueStacks exports may be distilled, but lifecycle automation remains
  operator-controlled rather than relying on undocumented executables.

See `docs/SEMANTIC_MULTIBOX_ARCHITECTURE.md`,
`docs/MACRO_DISTILLATION_CONTRACT.md`, and the vendor runbooks for the complete
contract.
