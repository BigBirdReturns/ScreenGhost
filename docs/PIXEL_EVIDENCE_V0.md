# ScreenGhost Pixel Evidence v0

Seal the **rendered surface the user actually saw** — a screenshot — through
genesis, and verify that visual record later without trusting the browser,
clipboard, ShareX, ScreenGhost, or GhostBox.

## Why pixels, not clipboard

A clipboard copy is polluted by DOM, hidden spans, rich text, embeds, quote
cards, and platform formatting. A screenshot captures the **actual visual
surface** the user saw. That pixel surface is the evidence object — and it is a
*different* object from the text-candidate ingestion path (view-tree / API). This
edge does not touch that path.

## Evidence tier — explicit and bounded

The tier is always `pixel_capture`:

- rendered surface only
- **not** DOM truth
- **not** API truth
- **not** an authenticated platform record
- **not** legal-grade provenance by itself

The sealed record proves *these pixels existed and have not changed since
sealing* — nothing about the page state beyond the rendered pixels.

## Path

```
intake folder (ShareX / manual / browser / OS screenshot + optional sidecar json)
   │  FilesystemIntakeProvider  (READ-ONLY; no ShareX dependency, no write-back)
   ▼
capture.png (opaque bytes)  ──►  pixel_capture_manifest.json  +  EvidenceEvent
   │   image_sha256, manifest_sha256, source label, capture method, dimensions
   │   url / page_title / captured_at ONLY if a sidecar supplied them (never inferred)
   ▼  pixel_seal.py  (genesis compiler, out-of-band key)
axm-hybrid1 SealedShard   (genesis-derived sh1_ custody id)
   │
   ▼  verify with an out-of-band key · pixel_exit_test.py
detached verify: PASS with only shard bytes + oob pub
   (no ShareX, no ScreenGhost, no GhostBox, no browser)
```

The PNG is treated as **opaque bytes**: hashed and sealed **verbatim**, never
decoded, re-encoded, OCR'd, or rewritten. (Width/height are read from the PNG
IHDR header — a read, never a rewrite.)

## Provider model

`FilesystemIntakeProvider` watches/reads a local folder where an external tool
(e.g. ShareX scrolling capture) saves PNGs. ScreenGhost consumes the PNG and an
optional `<stem>.json` sidecar:

| Sidecar key | Meaning |
|---|---|
| `url` | page URL, user-entered |
| `page_title` | window/page title |
| `app_name` | source app or browser |
| `capture_tool` | e.g. ShareX |
| `user_note` | free note |
| `captured_at` | capture timestamp |

**ShareX stays outside the codebase.** There is no ShareX dependency and no
browser automation in v0.

## Run it

```bash
# genesis kernel (axm-build / axm-verify) on PATH for the seal/verify steps
python examples/pixel_evidence_demo.py --make-sample          # synthesizes a sample PNG
python examples/pixel_evidence_demo.py --intake /path/to/sharex_folder --capture shot.png
python -m pytest tests/test_pixel_evidence.py -q
```

## Boundaries (enforced + tested)

- **No OCR by default.** Pixels are bytes, not text.
- **Pixels are not platform truth.** Tier is `pixel_capture`, and the limits ship
  inside the manifest.
- **Nothing inferred.** No author identity, timestamp, or URL unless a sidecar
  supplies it; absent → `null` and `source_label` = `unknown`.
- **Not mixed with clipboard/DOM ingestion.** The pixel path imports no
  `core.adapter` / `core.texttree` / view-tree parser (asserted at import + source
  level).
- **GhostBox does not own the image.** Nothing here imports `ghostbox`; GhostBox
  is downstream observation only and is not in this path.
- **The PNG is never rewritten after hashing.** Sealed bytes are byte-identical to
  the intake bytes; the intake folder is read-only.
- **Custody stays genesis's.** `shard_id` is the genesis-derived `sh1_`;
  verification uses an out-of-band key and the frozen
  `PASS / FAIL / MALFORMED / NO_TRUSTED_KEY` taxonomy.

## Live receipts (this environment)

| Check | Result |
|---|---|
| PNG import → `pixel_capture_manifest.json` | **yes** |
| image hash / manifest hash stable | **yes** (byte-deterministic) |
| sealed `axm-hybrid1` shard verifies (out-of-band key) | **PASS** (`sh1_…`, genesis-derived) |
| wrong key | **FAIL** |
| missing key | **NO_TRUSTED_KEY** (before the CLI) |
| PNG bytes unchanged after import + seal | **yes** (verbatim in shard; intake untouched) |
| evidence tier | **`pixel_capture`** |
| no DOM/clipboard parser invoked; no GhostBox imported | **yes** (subprocess-isolated + source-checked) |
| exit test: verify with only shard bytes + oob pub | **PASS** (no ShareX/ScreenGhost/GhostBox/browser) |
| Test suite | **16/16** (184/184 repo-wide) |

**Evidence tier of this slice:** rendered-surface capture proven against a
synthesized sample PNG and a real genesis seal/verify. The crypto backend is the
pure-Python `dilithium-py` fallback — functional, not load-proven. Live browser
automation and window-title URL detection are **not** built in v0.

## Control question

Can a user capture the rendered surface they actually saw, seal the PNG plus
manifest through genesis, and later verify that visual record without trusting the
browser, clipboard, ShareX, ScreenGhost, or GhostBox?

**v0 answer: yes** — sealed as opaque verbatim pixels at the explicit
`pixel_capture` tier, custody left to the genesis `sh1_`, and the detached verify
passes with none of them in the loop.
