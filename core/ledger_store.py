"""Local, append-only order-ledger store — the product spine, no phone.

Turns ScreenGhost outputs into a reviewable seller workflow backed by SQLite.
Invariants:
  * Raw captures are immutable — never updated, never overwritten.
  * Corrections are append-only facts with full lineage; the event row holds the
    current value, the corrections table holds the history.
  * The final ledger is reproducible by replaying captures + corrections.

Nothing here needs a model, a network, or a device.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional, Tuple

from core.orders import reduce_ledger

# ---- order review states + allowed transitions --------------------------- #
PROPOSED = "proposed"
PROPOSED_INCOMPLETE = "proposed_incomplete"   # order recognized but missing a required field
ACCEPTED = "accepted"
CORRECTED = "corrected"
REJECTED = "rejected"
NEEDS_INFO = "needs_info"
CANCELLED = "cancelled"
FULFILLED = "fulfilled"
STATES = {PROPOSED, PROPOSED_INCOMPLETE, ACCEPTED, CORRECTED, REJECTED,
          NEEDS_INFO, CANCELLED, FULFILLED}

_TRANSITIONS: Dict[str, set] = {
    PROPOSED: {ACCEPTED, CORRECTED, REJECTED, NEEDS_INFO, CANCELLED},
    PROPOSED_INCOMPLETE: {CORRECTED, NEEDS_INFO, REJECTED, CANCELLED, ACCEPTED},
    NEEDS_INFO: {ACCEPTED, CORRECTED, REJECTED, CANCELLED},
    CORRECTED: {ACCEPTED, REJECTED, NEEDS_INFO, CANCELLED},
    ACCEPTED: {CORRECTED, CANCELLED, FULFILLED},
    REJECTED: set(),
    CANCELLED: set(),
    FULFILLED: set(),
}
# statuses that count toward the live order ledger
LEDGER_STATES = {ACCEPTED, CORRECTED, FULFILLED}


SCHEMA_VERSION = 1
APP_VERSION = "demo-rc0"


class TransitionError(ValueError):
    pass


class SchemaError(RuntimeError):
    """A store's schema version is newer than this build supports (fail closed)."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sellers (
    seller_id TEXT PRIMARY KEY, cohort TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS buyers (
    buyer_id TEXT PRIMARY KEY, seller_id TEXT, display_name TEXT);
CREATE TABLE IF NOT EXISTS catalog_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, seller_id TEXT, sku TEXT, name TEXT,
    aliases TEXT, variants TEXT, price INTEGER, stock INTEGER);
CREATE TABLE IF NOT EXISTS captures (
    capture_id TEXT PRIMARY KEY, seller_id TEXT, buyer_display TEXT, ts TEXT,
    raw_text TEXT, unicode_ok INTEGER, dedupe_key TEXT, parser_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS order_events (
    event_id TEXT PRIMARY KEY, seller_id TEXT, capture_id TEXT, buyer TEXT,
    sku TEXT, qty INTEGER, variant TEXT, event_type TEXT, status TEXT,
    confidence REAL, dedupe_key TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS order_event_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, capture_id TEXT);
CREATE TABLE IF NOT EXISTS corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, field TEXT,
    old_value TEXT, new_value TEXT, reason TEXT, source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, from_status TEXT,
    to_status TEXT, source TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS ledger_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, seller_id TEXT, label TEXT, body TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, kind TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_CORRECTABLE = {"buyer", "sku", "qty", "variant"}


class LedgerStore:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False so the local review server (one thread per
        # request) can share the connection. A single human clicking a UI does
        # not need write concurrency beyond SQLite's own serialization.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self._init_meta()
        self.db.commit()

    def _init_meta(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES "
                "('schema_version',?),('app_version',?),"
                "('created_at_utc', strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
                (str(SCHEMA_VERSION), APP_VERSION))
            return
        found = int(row["value"])
        if found > SCHEMA_VERSION:
            raise SchemaError(
                f"store schema v{found} is newer than this build supports "
                f"(v{SCHEMA_VERSION}). Upgrade ScreenGhost to open it.")

    def schema_version(self) -> int:
        return int(self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"])

    def meta(self) -> Dict[str, str]:
        return {r["key"]: r["value"] for r in self.db.execute("SELECT key,value FROM meta")}

    # ---- reference data -------------------------------------------------- #
    def add_seller(self, seller_id: str, cohort: str = "") -> None:
        self.db.execute("INSERT OR REPLACE INTO sellers(seller_id, cohort) VALUES (?,?)",
                        (seller_id, cohort))
        self.db.commit()

    def add_buyer(self, buyer_id: str, seller_id: str, display_name: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO buyers(buyer_id, seller_id, display_name) VALUES (?,?,?)",
            (buyer_id, seller_id, display_name))
        self.db.commit()

    def import_catalog_rows(self, seller_id: str, rows: List[Dict]) -> int:
        n = 0
        for r in rows:
            self.db.execute(
                "INSERT INTO catalog_items(seller_id,sku,name,aliases,variants,price,stock)"
                " VALUES (?,?,?,?,?,?,?)",
                (seller_id, r["sku"], r.get("name", ""),
                 json.dumps(r.get("aliases", []), ensure_ascii=False),
                 json.dumps(r.get("variants", []), ensure_ascii=False),
                 int(r.get("price", 0)), int(r.get("stock", 1))))
            n += 1
        self.db.commit()
        return n

    def catalog(self, seller_id: str) -> List[Dict]:
        cur = self.db.execute("SELECT * FROM catalog_items WHERE seller_id=?", (seller_id,))
        out = []
        for row in cur:
            out.append({"sku": row["sku"], "name": row["name"],
                        "aliases": json.loads(row["aliases"] or "[]"),
                        "variants": json.loads(row["variants"] or "[]"),
                        "price": row["price"], "stock": row["stock"]})
        return out

    # ---- captures (immutable) ------------------------------------------- #
    def add_capture(self, capture_id: str, seller_id: str, buyer_display: str,
                    ts: str, raw_text: str, dedupe_key: str,
                    parser_path: str, unicode_ok: bool = True) -> str:
        # INSERT OR IGNORE: a re-observed capture (same id) never overwrites the
        # original raw text. Raw captures are write-once.
        self.db.execute(
            "INSERT OR IGNORE INTO captures(capture_id,seller_id,buyer_display,ts,"
            "raw_text,unicode_ok,dedupe_key,parser_path) VALUES (?,?,?,?,?,?,?,?)",
            (capture_id, seller_id, buyer_display, ts, raw_text,
             1 if unicode_ok else 0, dedupe_key, parser_path))
        self.db.commit()
        return capture_id

    def get_capture(self, capture_id: str) -> Optional[Dict]:
        row = self.db.execute("SELECT * FROM captures WHERE capture_id=?",
                              (capture_id,)).fetchone()
        return dict(row) if row else None

    # ---- order events ---------------------------------------------------- #
    def propose_event(self, event_id: str, seller_id: str, capture_id: str,
                      buyer: str, sku: Optional[str], qty: Optional[int],
                      variant: Optional[str], event_type: str,
                      confidence: float, dedupe_key: str,
                      status: str = PROPOSED) -> str:
        self.db.execute(
            "INSERT OR IGNORE INTO order_events(event_id,seller_id,capture_id,buyer,"
            "sku,qty,variant,event_type,status,confidence,dedupe_key)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, seller_id, capture_id, buyer, sku, qty, variant,
             event_type, status, confidence, dedupe_key))
        self.db.execute(
            "INSERT INTO order_event_sources(event_id,capture_id) VALUES (?,?)",
            (event_id, capture_id))
        self.db.commit()
        return event_id

    def get_event(self, event_id: str) -> Optional[Dict]:
        row = self.db.execute("SELECT * FROM order_events WHERE event_id=?",
                              (event_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["sources"] = [r["capture_id"] for r in self.db.execute(
            "SELECT capture_id FROM order_event_sources WHERE event_id=?", (event_id,))]
        d["corrections"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM corrections WHERE event_id=? ORDER BY correction_id", (event_id,))]
        return d

    def events(self, seller_id: str, status: Optional[str] = None) -> List[Dict]:
        q = "SELECT * FROM order_events WHERE seller_id=?"
        args: Tuple = (seller_id,)
        if status:
            q += " AND status=?"
            args += (status,)
        q += " ORDER BY created_at, event_id"
        return [dict(r) for r in self.db.execute(q, args)]

    def transition(self, event_id: str, new_status: str, source: str = "human") -> None:
        if new_status not in STATES:
            raise TransitionError(f"unknown status {new_status!r}")
        cur = self.db.execute("SELECT status FROM order_events WHERE event_id=?",
                             (event_id,)).fetchone()
        if not cur:
            raise TransitionError(f"no event {event_id!r}")
        if new_status not in _TRANSITIONS[cur["status"]]:
            raise TransitionError(f"{cur['status']} -> {new_status} not allowed")
        # append-only transition record — the current status lives on the event
        # row; the log preserves how it got there.
        self.db.execute(
            "INSERT INTO transitions(event_id,from_status,to_status,source) VALUES (?,?,?,?)",
            (event_id, cur["status"], new_status, source))
        self.db.execute("UPDATE order_events SET status=? WHERE event_id=?",
                        (new_status, event_id))
        self.db.commit()

    def transitions(self, event_id: str) -> List[Dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM transitions WHERE event_id=? ORDER BY id", (event_id,))]

    def correct(self, event_id: str, field: str, new_value, reason: str,
                source: str = "human") -> int:
        if field not in _CORRECTABLE:
            raise TransitionError(f"field {field!r} is not correctable")
        row = self.db.execute("SELECT * FROM order_events WHERE event_id=?",
                             (event_id,)).fetchone()
        if not row:
            raise TransitionError(f"no event {event_id!r}")
        old = row[field]
        cur = self.db.execute(
            "INSERT INTO corrections(event_id,field,old_value,new_value,reason,source)"
            " VALUES (?,?,?,?,?,?)",
            (event_id, field, str(old), str(new_value), reason, source))
        self.db.execute(f"UPDATE order_events SET {field}=?, status=? WHERE event_id=?",
                        (new_value, CORRECTED, event_id))
        self.db.commit()
        return cur.lastrowid

    # ---- ledger ---------------------------------------------------------- #
    def current_ledger(self, seller_id: str) -> Dict[Tuple[str, str], int]:
        # Reduce in arrival order (capture ts), so MODIFY/repeat semantics match
        # the order the comments actually happened, not event-id hash order.
        rows = self.db.execute(
            "SELECT e.event_type,e.buyer,e.sku,e.qty FROM order_events e "
            "JOIN captures c ON e.capture_id=c.capture_id WHERE e.seller_id=? "
            "AND e.status IN (%s) ORDER BY c.ts, e.event_id"
            % ",".join("?" * len(LEDGER_STATES)),
            (seller_id, *LEDGER_STATES)).fetchall()
        return reduce_ledger((r["event_type"], r["buyer"], r["sku"], r["qty"])
                             for r in rows)

    def corrected_event_count(self, seller_id: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(DISTINCT c.event_id) n FROM corrections c "
            "JOIN order_events e ON c.event_id=e.event_id WHERE e.seller_id=?",
            (seller_id,)).fetchone()
        return row["n"]

    def snapshot_ledger(self, seller_id: str, label: str) -> None:
        led = self.current_ledger(seller_id)
        body = json.dumps({f"{b}|{s}": q for (b, s), q in led.items()}, ensure_ascii=False)
        self.db.execute("INSERT INTO ledger_snapshots(seller_id,label,body) VALUES (?,?,?)",
                        (seller_id, label, body))
        self.db.commit()

    def get_snapshot(self, seller_id: str, label: str):
        row = self.db.execute(
            "SELECT body FROM ledger_snapshots WHERE seller_id=? AND label=? "
            "ORDER BY id DESC LIMIT 1", (seller_id, label)).fetchone()
        if not row:
            return None
        out = {}
        for k, q in json.loads(row["body"]).items():
            b, s = k.split("|", 1)
            out[(b, s)] = q
        return out

    def record_export(self, path: str, kind: str) -> None:
        self.db.execute("INSERT INTO exports(path,kind) VALUES (?,?)", (path, kind))
        self.db.commit()

    def all_captures(self, seller_id: str) -> List[Dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM captures WHERE seller_id=? ORDER BY ts", (seller_id,))]

    def all_corrections(self, seller_id: str) -> List[Dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT c.* FROM corrections c JOIN order_events e ON c.event_id=e.event_id "
            "WHERE e.seller_id=? ORDER BY c.correction_id", (seller_id,))]

    def close(self) -> None:
        self.db.close()
