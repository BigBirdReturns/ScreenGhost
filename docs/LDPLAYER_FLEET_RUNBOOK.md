# LDPlayer fleet runbook

LDPlayer exposes instance management through `dnconsole.exe` or `ldconsole.exe` and
publishes a keyboard-macro language containing `size`, `touch`, `wait`, `key`, and
`text` primitives. Semantic Multibox parses that language directly.

## Prepare clones

Create one golden instance and clone it into a leader, coordinate-baseline cohort,
and semantic cohort. The cohorts must start at the same application state and must
not share instances.

## Parse a macro

```powershell
python -m experiments.emulator_fleet parse-macro `
  .\my-flow.txt `
  --format ldplayer `
  --output log/semantic_multibox/my-flow.json
```

Unsupported commands are preserved and cause a non-zero parse result. Remove or
replace them with a bounded primitive rather than allowing the compiler to guess.

## Run the measured plan

Copy `configs/emulator_fleet/ldplayer.example.json`, update paths and selectors,
then execute the same verifier command used for MEmu with `--apply-machine`.
