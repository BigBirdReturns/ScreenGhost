# Physical-machine and Android emulator runbook

## 1. Run the conclusion campaign first

```bat
BOOTSTRAP_GENERIC_UTILITY.cmd -InstallChromium
RUN_GENERIC_UTILITY_CAMPAIGN.cmd
```

Inspect:

```text
log/generic_utility/full/doctor.json
log/generic_utility/full/campaign/campaign_receipt.json
log/generic_utility/full/campaign/gate_results.json
log/generic_utility/full/browser/receipt.json
log/generic_utility/full/grounding-emulated/benchmark/benchmark_receipt.json
log/generic_utility/full/conclusion.json
```

The conclusion receipt should say `premise_conclusion_ready: true` before any physical test is attempted.

## 2. Measure a local grounder

```bat
BOOTSTRAP_GENERIC_UTILITY.cmd -Models
powershell -ExecutionPolicy Bypass -File scripts/generic_utility/run-local-grounder.ps1
```

Do not compare simulated campaign milliseconds with measured model milliseconds without preserving their metric labels.

## 3. AndroidWorld final emulator smoke

AndroidWorld currently documents a Pixel 6 AVD using Android 13 / API 33, named `AndroidWorldAvd`, launched from the command line with `-grpc 8554`. Install AndroidWorld through the supplied script or its upstream instructions, launch the AVD without taking desktop foreground, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/generic_utility/run-androidworld-smoke.ps1 `
  -AdbPath "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
```

This smoke is read-only. It captures pixels and accessibility state through a temporal certificate, compiles a curriculum, records zero input actions, and closes the environment.

## 4. Physical or ordinary AVD smoke

With one local ADB device connected:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/generic_utility/run-physical-smoke.ps1 `
  -Device emulator-5554 `
  -SurfaceId "settings.display"
```

The command uses ScreenGhost's existing local-only `AndroidAdbDriver`, PR #13's Android parser and lesson compiler, and the temporal alignment gate. It performs no input action and requests no host foreground.

## 5. Assemble the final measured conclusion

```powershell
python -m experiments.generic_utility conclude `
  --campaign log/generic_utility/full/campaign `
  --browser log/generic_utility/full/browser/receipt.json `
  --grounding log/generic_utility/grounding-local/benchmark/benchmark_receipt.json `
  --androidworld log/generic_utility/androidworld/receipt.json `
  --physical log/generic_utility/physical/receipt.json `
  --out log/generic_utility/final-conclusion.json
```

`production_claim_ready` remains false unless a measured grounder and at least one live Android transport receipt are present.
