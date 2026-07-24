# MEmu fleet runbook

## Prepare the fleet

1. Create one golden MEmu instance and install the test application.
2. Navigate to the exact initial state and stop the instance cleanly.
3. Clone it into one leader, one or more coordinate-baseline instances, and one or
   more semantic instances. If the campaign changes density, orientation, or layout,
   also create a disjoint visual-teacher clone for each geometry being tested.
4. Keep the application account data intentionally varied only after cloning when
   the experiment calls for account diversity.
5. Use a common resolution for the coordinate baseline. Geometry variants belong
   in semantic clones and their pre-runtime visual-teacher counterparts.

MEMUC paths vary by installation. Set `SG_MEMUC` or place the path in the machine
plan. Inventory is read-only:

```powershell
python -m experiments.emulator_fleet inventory `
  --vendor memu `
  --executable "$env:SG_MEMUC"
```

## Operation Recorder

MEmu `.mir` files and `info.ini` metadata can be cataloged:

```powershell
python -m experiments.emulator_fleet memu-catalog `
  "C:\path\to\MEmu\scripts" `
  --output log/semantic_multibox/memu_macro_catalog.json
```

The `.mir` bytes remain opaque. To distill a MEmu demonstration, export or author a
supported action manifest in the LDPlayer-style `size/touch/wait/key/text` grammar,
or run the operation manually while producing an observed action manifest. The
package does not claim a stable undocumented `.mir` decoder.

## Machine run

Copy `configs/emulator_fleet/memu.example.json`, fill in the leader and disjoint
visual-teacher/baseline/semantic clone names, then run:

```powershell
python VERIFY_SEMANTIC_MULTIBOX.py `
  --machine-plan configs/emulator_fleet/memu.local.json `
  --apply-machine `
  --require-machine
```

`--apply-machine` permits Android input through MEMUC's instance-scoped ADB wrapper.
It does not authorize instance deletion.
