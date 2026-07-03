---
name: Receipt reproducibility issue
about: A receipt does not reproduce from a cold run / same seed differs
title: "[repro] <one-line>"
labels: reproducibility
---

**Receipt**: `examples/receipts/<name>.txt`

**Command**:
```
python examples/verify_demo_receipt.py --receipt examples/receipts/<name>.txt
```

**Mismatch output** (saved vs rerun): ...

**Reminder**: all seeded reproducibility MUST use stable sha256 hashing — never
Python's salted builtin `hash()`. A mismatch is usually a hidden source of
process-local nondeterminism (dict/set order, `hash()`, wall-clock, filesystem
order). Wall-clock time is not compared.
