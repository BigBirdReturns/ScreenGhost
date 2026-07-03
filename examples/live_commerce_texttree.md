# Live Commerce order intake — exact text, no OCR, no VLM

This is the concrete answer to "vision models can't read Thai reliably at
volume." It doesn't make the model better. It takes the model off the text path.

Android already exposes the live chat screen as a hierarchy of **text nodes**
via UiAutomator. Every node's `text` is the real Unicode string the OS is
rendering, so a Thai order comment (`CF 2 ตัว ค่ะ`) comes back byte-for-byte —
there is nothing to misrecognize because nothing was ever an image. Stickers and
shared locations arrive as typed payloads from their `content-desc`, the way a
protocol API would hand them to you, instead of as pixels to re-classify.

This is deliberately boring, pre-2023 technology. That is the point.

## The loop

```python
from core.texttree import read_tree, to_elements

# Reads the live view tree over the same local-only ADB path as the hands.
# No screenshot, no model load, no GPU.
nodes = read_tree(device="R58Mxxxxxxx")

# Structured, ScreenState-shaped elements — feed straight to order matching.
for el in to_elements(nodes):
    print(el["type"], "→", el["label"])
# text     → CF 2 ตัว ค่ะ
# sticker  → Sticker
# location → Shared location: ตลาดนัด
# input    → ส่งข้อความ
# toggle   → Notifications
```

Because each node carries exact `bounds`, reading feeds the hands directly —
`node.center()` is the tap target, resolved on today's screen, with no cached
coordinate to go stale on a redesign:

```python
order = next(n for n in nodes if n.text.startswith("CF"))
assert order.text == "CF 2 ตัว ค่ะ"     # exact quantity — a mis-shipment averted
x, y = order.center()                     # tap "reply" against the live layout
```

## Where each layer sits

1. **View tree (`read_tree`) — default.** Exact Unicode; Thai and mixed scripts
   solved by construction; near-zero compute, so it keeps up with a live comment
   stream.
2. **Clipboard / copy — for exact selectable text the tree renders thin.**
   Serial gesture; great on settled screens, not a per-message firehose.
3. **VLM — last resort, genuine pixels only.** A sticker's *artwork*, a map
   thumbnail. Small, on-demand slice — never the hot path.

## Honest edges (carried in the open)

- **Non-text payloads are typed, not fully decoded.** The tree tells you *a
  sticker is here* and often *which* via `content-desc`; the artwork itself
  still needs a tap-in or a Layer-3 pass. Better than pixels, not free.
- **`read_tree` needs a device;** the parser (`parse_ui_dump`) is pure and is
  what the offline tests exercise (`tests/test_texttree.py`) — including the
  Thai order line above, asserted verbatim.
- **Topology, not covered here:** running this on each seller's own always-on
  phone with a cloud tier that only aggregates already-structured events is what
  makes it scale without a device farm or a shared origin IP. See
  [`../docs/scaling-architecture.md`](../docs/scaling-architecture.md).
