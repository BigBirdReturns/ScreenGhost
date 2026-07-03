---
name: Product / UI issue
about: The review loop, store, export, or UI behaves unexpectedly
title: "[product] <one-line>"
labels: product
---

**Area**: review UI / ledger store / export / setup / replay / other

**Steps**:
1. `python examples/operator_demo.py --seed <x> --sellers <n> [--serve]`
2. ...

**Expected**: ...
**Got**: ...

**Store / receipt** (if relevant): attach `log/.../ledger.db` path or the
receipt. Remember: raw captures are immutable; corrections and transitions are
append-only — a fix must preserve those invariants.
