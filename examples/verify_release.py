"""Release verifier — run every evidence stage and print one release receipt.

    python examples/verify_release.py

Runs: proof-demo smoke, operator-demo smoke, receipt verification, adapter
conformance --all, replay check, and a fast test subset. Prints a single
release receipt (also written to examples/receipts/release_v0.1.txt). Exit 0
only if every stage passes. No hardware / live-seller / business claim.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECEIPT = os.path.join(ROOT, "examples", "receipts", "release_v0.1.txt")
FORBIDDEN = "hardware proof, live-seller proof, business-outcome proof, universal real-app compatibility"


def _run(argv, expect=None):
    r = subprocess.run([sys.executable] + argv, cwd=ROOT,
                       capture_output=True, text=True)
    ok = r.returncode == 0 and (expect is None or expect in (r.stdout + r.stderr))
    return ok, r


def main() -> None:
    stages = []
    with tempfile.TemporaryDirectory() as tmp:
        op_out = os.path.join(tmp, "op")
        stages.append(("proof_demo_smoke", _run(
            ["examples/pop_proof_demo.py", "--seed", "release", "--sellers", "15"],
            "universal impossibility claim")[0]))
        stages.append(("operator_demo_smoke", _run(
            ["examples/operator_demo.py", "--seed", "op", "--sellers", "5",
             "--out", op_out], "replay_matched")[0]))
        stages.append(("receipt_verification", _run(
            ["examples/verify_demo_receipt.py", "--receipt",
             "examples/receipts/operator_demo_seed_op.txt"], "MATCH")[0]))
        stages.append(("adapter_conformance_all", _run(
            ["examples/adapter_conformance.py", "--all"], "0 undeclared")[0]))
        stages.append(("replay_check", _run(
            ["examples/replay_ledger.py", "--store",
             os.path.join(op_out, "ledger.db")], "replay matched (all sellers): True")[0]))
        stages.append(("fast_tests", _run(
            ["-m", "pytest", "tests/test_ledger_store.py",
             "tests/test_adapter_conformance.py", "-q"], "passed")[0]))

    all_ok = all(ok for _n, ok in stages)
    lines = ["ScreenGhost v0.1 — Evidence Kernel release receipt",
             "NOT hardware proof. NOT live-seller proof. NOT business proof.",
             ""]
    for name, ok in stages:
        lines.append(f"  {name:26}: {'PASS' if ok else 'FAIL'}")
    lines += ["", f"  release_verdict           : {'PASS' if all_ok else 'FAIL'}",
              f"  forbidden_claims          : {FORBIDDEN}",
              "",
              "reproduce: python examples/verify_release.py",
              "claim boundaries: docs/CLAIM_BOUNDARIES.md"]
    text = "\n".join(lines)
    print(text)
    with open(RECEIPT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
