# Local GUI-grounder benchmark

## Boundary

The benchmark scores a model against hidden teacher boxes on eleven phone screenshots spanning settings, profile editing, timer controls, an untaught connectivity switch, dark theme, and increased font scale. The provider receives the image and instruction only. Hidden bounds remain in the scorer and are represented in the public suite manifest only by a digest.

No model process receives motor authority. The attached worker communicates through stdin and stdout, binds no port, loads one model once, and is terminated at campaign end or immediately on deadline.

## Emulator plumbing proof

```powershell
python -m experiments.generic_utility grounding-emulated `
  --out log/generic_utility/grounding-emulated
```

This uses a declared oracle fixture and must report `teacher_answers_visible_to_provider=true`, `metric_kind=measured` for process timing, and a 100 percent hit rate. It proves benchmark plumbing, not model quality.

## RTX 4060 measurement

Install optional dependencies:

```bat
BOOTSTRAP_GENERIC_UTILITY.cmd -Models
```

Then run:

```powershell
scripts/generic_utility/run-local-grounder.ps1 `
  -Model osunlp/UGround-V1-2B `
  -DType float16
```

The default worker uses the Hugging Face `image-text-to-text` pipeline and records raw model output, parsed point, per-request time, process lifetime, and peak `nvidia-smi` memory when available. The worker is generic; if a checkpoint requires its upstream custom inference path, configure that path as an attached JSONL worker while preserving the same receipt contract.

A useful model receipt reports completion and hit rate. It must not be merged into the emulator claim as if simulated and measured values were interchangeable.
