---
name: Adapter failure
about: A surface/adapter does not satisfy the candidate conformance contract
title: "[adapter] <surface> — <one-line>"
labels: adapter
---

**Surface / app**: (e.g. LINE, Facebook Live, Shopee, WebView)

**Fixture**: attach or name the `examples/adapter_fixtures/*.xml` reproducing it.
If none exists, that's fine — this issue is the fixture request.

**Verifier output**:
```
python examples/adapter_conformance.py --fixture <fixture>.xml
```

**Named cause** (exactly one): `no_text_exposed` / `unicode_corruption` /
`row_grouping_failure` / `dedupe_failure` / `payload_classification_failure` /
`visible_window_loss` / `unsupported_surface` / `malformed_fixture`

**Is this `no_text_exposed`?** If the surface genuinely withholds exact text,
say so — that is an honest architectural boundary, not an adapter bug.
