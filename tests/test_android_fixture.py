"""Category [2a] view-tree seam: rendered seller-world parity, run offline.

Proves ledger recovery through device-format XML + geometric row
reconstruction: exact Thai preserved, sender/ts/text regrouped from bounds,
scroll duplication deduped, realistic display drift absorbed, pathological drift
failing in the open, and seam parity with the in-process result on clean
cohorts. No device, no OCR.
"""
from core.android_fixture import (
    RenderParams, capture_world, group_rows, render_uiautomator, run_parity,
    score_capture,
)
from core.orders import ChatMessage
from core.population import build_population
from core.texttree import parse_ui_dump


def _round_trip(msgs, params=None):
    return group_rows(parse_ui_dump(render_uiautomator(msgs, params or RenderParams())))


def test_render_is_valid_uiautomator_xml_with_thai():
    xml = render_uiautomator([ChatMessage("Nok3", "t00001", "CF A01 x2 ตัว ค่ะ")],
                             RenderParams())
    nodes = parse_ui_dump(xml)
    assert any(n.text == "CF A01 x2 ตัว ค่ะ" for n in nodes)


def test_thai_exact_through_the_seam():
    rows = _round_trip([ChatMessage("Ploy7", "t00042", "CF บี12 x3 ชิ้น ค่ะ")])
    assert len(rows) == 1
    assert rows[0].text == "CF บี12 x3 ชิ้น ค่ะ"  # byte-for-byte


def test_row_reconstruction_recovers_identity():
    src = [ChatMessage("Bee2", "t00001", "CF A01 x1 ค่ะ"),
           ChatMessage("Mai5", "t00002", "ยกเลิก A01 ค่ะ")]
    rows = _round_trip(src)
    got = {(r.sender, r.ts, r.text) for r in rows}
    assert got == {(m.sender, m.ts, m.text) for m in src}
    # recovered ids equal the originals -> the seam added nothing
    assert {r.id for r in rows} == {m.id for m in src}


def test_scroll_duplication_dedupes_no_double_orders():
    world = next(w for w in build_population(n=40, seed=5)
                 if w.profile.cohort == "secondhand")  # has dup_rows
    actionable = capture_world(world)
    ids = [ev.msg_id for ev in actionable]
    assert len(ids) == len(set(ids))  # each order emitted at most once


def test_realistic_display_drift_holds():
    world = next(w for w in build_population(n=40, seed=5)
                 if w.profile.cohort == "bakery")
    base = score_capture(world, capture_world(world))
    for p in (RenderParams(font_scale=1.5), RenderParams(font_scale=0.85),
              RenderParams(bubble_width=280), RenderParams(dark=True)):
        drifted = score_capture(world, capture_world(world, p))
        assert drifted["tp"] == base["tp"]          # recall unchanged
        assert drifted["corruptions"] == 0          # Thai intact through drift


def test_pathological_overlap_fails_in_the_open():
    world = next(w for w in build_population(n=40, seed=5)
                 if w.profile.cohort == "bakery")
    broken = score_capture(world, capture_world(world, RenderParams(row_h_base=15)))
    assert broken["corruptions"] > 0                # reported, not silent
    assert broken["tp"] < broken["should_emit"]     # real recall loss
    assert broken["ledger_match"] is False


def test_payload_node_does_not_become_an_order():
    xml = render_uiautomator([ChatMessage("Nok3", "t1", "CF A01 x1 ค่ะ")],
                             RenderParams(), payloads={0: ("sticker", "Sticker")})
    rows = group_rows(parse_ui_dump(xml))
    assert [r.text for r in rows] == ["CF A01 x1 ค่ะ"]  # sticker row dropped


def test_seam_parity_matches_inprocess_on_clean_cohorts():
    worlds = [w for w in build_population(n=60, seed=1337)
              if w.profile.cohort in {"bakery", "homegoods", "kids"}]
    for r in run_parity(worlds):
        assert r.seam_recall == r.inproc_recall == 1.0
        assert r.seam_precision == r.inproc_precision == 1.0
        assert r.corruptions == 0


def test_seam_boundary_still_fails_when_window_exceeded():
    world = next(w for w in build_population(n=60, seed=1)
                 if w.profile.traffic in {"hot", "burst"})
    tight = score_capture(world, capture_world(world, window=2))
    assert tight["tp"] < world_should_emit(world)


def world_should_emit(world):
    return sum(1 for lm in world.messages if lm.should_emit)
