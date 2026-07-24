# Conclusion contract

The campaign may conclude that the original generic-utility premise is supported only when all of these emulator gates pass:

1. At least one taught multi-step task completes after reset with zero action-time teacher reads and zero large-VLM calls.
2. Median simulated warm GPU-active cost falls by at least 50 percent relative to cold teaching.
3. An untaught application yields at least one successful action through generic PhoneGrammar.
4. Unknown and deceptive look-alike screens produce no confident false match.
5. Duplicate actions, overlapping pending transactions, and host foreground changes remain zero.
6. Every committed action has an external controller receipt and a visible postcondition.
7. Dynamic content remains within a known screen family.
8. Model timeout terminates before any motor call.
9. Warm and holdout actions carry no privileged teacher dependency.

`experiments.generic_utility.conclusion` combines the campaign bundle with optional browser, local-grounder, AndroidWorld, and physical-ADB receipts.

- `premise_conclusion_ready=true` means the deterministic system experiment supports the cold-to-warm premise.
- `production_claim_ready=true` additionally requires a measured local grounder and a live AndroidWorld or physical-device transport receipt.

A physical smoke failure may identify a transport or compatibility defect. It does not retroactively make a passed emulator receipt disappear, and a passed emulator receipt does not excuse a live defect.
