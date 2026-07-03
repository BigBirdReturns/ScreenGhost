---
name: Resolver failure
about: A comment resolves to the wrong SKU/quantity/variant, or fails to resolve
title: "[resolver] <one-line>"
labels: resolver
---

**Raw text** (exact): `...`

**Catalog** (sku, name, aliases, variants): ...

**Expected**: sku=`...` qty=`...` variant=`...`
**Got**: sku=`...` qty=`...` variant=`...`

**Class**: spelling / alias ambiguity / variant missing / code-switch / other

**Note**: the product resolver (`core/resolver.py`) is separate from the frozen
proof-stack baseline (`core/population.resolve_sku`). Resolver fixes must not
move the proof baseline. If a required variant is missing, the correct behavior
is `proposed_incomplete` / `needs_info`, never a hallucinated completed order.
