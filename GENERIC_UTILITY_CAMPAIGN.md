# Generic Utility Warm Path

This additive experiment package sits on top of Surface Teacher PR #13. It supplies the complete emulator-first path needed to decide the original premise: after an expensive first teaching pass, a phone should become a generic utility whose familiar operation is cheaper, patient, teacher-blind, and transferable across applications.

The package contains a deterministic rendered phone world, hidden teacher, visual state index, generic phone grammar, semantic decision cache, transition graph, single-flight transactional controller, attached local-model worker, hidden-answer grounding benchmark, AndroidWorld adapter, real Chromium receipt, optional read-only ADB smoke, and an explicit conclusion assembler.

## Windows one-command campaign

```bat
BOOTSTRAP_GENERIC_UTILITY.cmd -InstallChromium
RUN_GENERIC_UTILITY_CAMPAIGN.cmd
```

The launcher resolves `python` first, honors `SG_PYTHON`, bypasses only the launcher process's PowerShell policy, starts no listener, and writes all receipts below `log/generic_utility/full`.

The authoritative verifier is:

```powershell
python VERIFY_GENERIC_UTILITY_CAMPAIGN.py --out log/generic_utility/full
```

For the RTX 4060 plus Android emulator or attached-device receipts, use:

```bat
RUN_GENERIC_UTILITY_MEASURED.cmd -AndroidWorldAdbPath "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" -PhysicalDevice emulator-5554
```

It runs compilation, the focused acceptance suite, the complete deterministic campaign, bundle-integrity verification, the oracle grounding protocol, a real local Chromium smoke when available, the full ScreenGhost test suite when executed inside a complete checkout, and the conclusion gate. Optional local-model, AndroidWorld, and physical-device receipts are described in `docs/PHYSICAL_MACHINE_RUNBOOK.md`.

See `docs/GENERIC_UTILITY_CAMPAIGN.md` for the mechanism, `docs/ORIGINAL_PREMISE_LEDGER.md` for the claim being tested, and `docs/CONCLUSION_CONTRACT.md` for the evidence boundary.
