"""Order extraction + exactly-once emission — the correctness the volume needs.

The critic's real requirement is "capture every comment exactly once." These
tests prove the two hard parts offline: parsing Thai confirm lines, and keeping
two buyers who type the identical thing apart while collapsing the same comment
re-read across polls.
"""
from core.orders import ChatMessage, OrderBook, parse_confirm


def test_parses_thai_confirm_line():
    is_confirm, item, qty = parse_confirm("CF A12 x2 ค่ะ")
    assert is_confirm and item == "A12" and qty == 2


def test_parses_thai_counter_quantity():
    # "2 ตัว" = 2 pieces — quantity in the local counter, not "x2".
    is_confirm, item, qty = parse_confirm("cf B7 2 ตัว")
    assert is_confirm and item == "B7" and qty == 2


def test_non_confirm_is_not_an_order():
    is_confirm, item, qty = parse_confirm("สวัสดีค่ะ ราคาเท่าไหร่")
    assert not is_confirm and item is None and qty is None


def test_confirm_keyword_is_word_bounded():
    # "scuffle"/"cfo" must not trip the "cf" keyword.
    assert parse_confirm("that was a scuffle")[0] is False
    assert parse_confirm("our cfo said hi")[0] is False


def test_same_comment_across_polls_emits_once():
    book = OrderBook()
    m = ChatMessage(sender="buyer001", ts="14:32", text="CF A01 x1")
    first = book.ingest([m])
    second = book.ingest([m])  # same screen re-read on the next poll
    assert len(first) == 1 and second == []
    assert book.duplicates_suppressed == 1


def test_identical_text_from_two_buyers_stays_two_orders():
    # The collision that a text-only dedup would silently merge.
    book = OrderBook()
    a = ChatMessage(sender="buyer001", ts="14:32", text="CF A03 x2")
    b = ChatMessage(sender="buyer002", ts="14:32", text="CF A03 x2")
    events = book.ingest([a, b])
    assert len(events) == 2
    assert {e.sender for e in events} == {"buyer001", "buyer002"}


def test_same_buyer_distinct_timestamps_are_distinct_orders():
    book = OrderBook()
    a = ChatMessage(sender="buyer001", ts="14:32", text="CF A03 x1")
    b = ChatMessage(sender="buyer001", ts="14:35", text="CF A03 x1")
    assert len(book.ingest([a, b])) == 2


def test_documented_identity_merge_limit():
    # Honest bound: same buyer, same coarse timestamp, identical text -> merged.
    # This is the one miss, and it is documented, not silent.
    book = OrderBook()
    a = ChatMessage(sender="buyer001", ts="14:32", text="CF A03 x1")
    b = ChatMessage(sender="buyer001", ts="14:32", text="CF A03 x1")
    assert len(book.ingest([a, b])) == 1
    assert book.duplicates_suppressed == 1


def test_order_event_is_serializable():
    book = OrderBook()
    ev = book.ingest([ChatMessage("buyer9", "10:00", "CF L07 x3 ค่ะ")])[0]
    d = ev.to_dict()
    assert d["item_code"] == "L07" and d["qty"] == 3 and d["is_confirm"] is True
    assert set(d) == {"msg_id", "sender", "ts", "text", "is_confirm", "item_code", "qty"}
