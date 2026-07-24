# BlueStacks baseline lane

BlueStacks provides multi-instance creation and cloning, operation synchronization,
macro recording and import/export, and Eco Mode through supported product surfaces.
Semantic Multibox uses exported macro JSON as demonstration evidence and treats the
built-in synchronizer as a coordinate baseline.

The package intentionally does not invoke undocumented BlueStacks lifecycle
executables. The operator creates and resets instances through the supported
Multi-instance Manager, exports a macro, and may parse it with:

```powershell
python -m experiments.emulator_fleet parse-macro macro.json `
  --format bluestacks `
  --default-resolution 540 960
```

A future CLI provider requires an official stable lifecycle interface or a separately
reviewed adapter. The absence of such a provider does not weaken the MEmu and
LDPlayer experiment.
