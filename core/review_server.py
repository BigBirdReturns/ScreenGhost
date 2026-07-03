"""Minimal local review UI — stdlib only, no framework, no network egress.

A single-page web UI over the LedgerStore: inbox of proposed/needs_info events,
raw-capture + parsed-event + catalog panels, correction form, seller ledger,
and export. Bind to localhost. Nothing here calls out.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from core.eval_population import _ledger_shape
from core.ledger_store import (
    ACCEPTED, CANCELLED, FULFILLED, LedgerStore, NEEDS_INFO, REJECTED,
)

_PAGE = """<!doctype html><meta charset=utf-8><title>ScreenGhost — Review Ledger</title>
<style>body{font:14px system-ui;margin:0;display:flex;height:100vh}
#l{width:34%;overflow:auto;border-right:1px solid #ccc}#r{flex:1;overflow:auto;padding:12px}
.ev{padding:8px 12px;border-bottom:1px solid #eee;cursor:pointer}.ev:hover{background:#f4f4f4}
.st{font-size:11px;padding:1px 6px;border-radius:8px;background:#eee}
h3{margin:14px 0 6px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px 8px}
button{margin:2px;padding:4px 8px}input{width:90px}</style>
<div id=l><div style=padding:8px><select id=seller onchange=load()></select>
<button onclick=doExport()>Export</button></div><div id=list></div></div>
<div id=r><div id=detail>pick an event</div><h3>Ledger</h3><div id=ledger></div></div>
<script>
let cur=null;
async function j(u,o){return (await fetch(u,o)).json()}
async function boot(){let s=await j('/api/sellers');document.getElementById('seller').innerHTML=
 s.map(x=>`<option>${x.seller_id}</option>`).join('');load()}
async function load(){let sel=document.getElementById('seller').value;
 let ev=await j('/api/events?seller='+encodeURIComponent(sel));
 document.getElementById('list').innerHTML=ev.map(e=>`<div class=ev onclick="show('${e.event_id}')">
  <span class=st>${e.status}</span> ${e.buyer} — <b>${e.sku||'?'}</b> ×${e.qty||'?'}
  <div style=color:#888>${e.event_type} · conf ${e.confidence}</div></div>`).join('');
 let led=await j('/api/ledger?seller='+encodeURIComponent(sel));
 document.getElementById('ledger').innerHTML='<table><tr><th>buyer<th>sku<th>qty</tr>'+
  led.map(r=>`<tr><td>${r.buyer}<td>${r.sku}<td>${r.qty}</tr>`).join('')+'</table>'}
async function show(id){cur=id;let e=await j('/api/event/'+id);
 document.getElementById('detail').innerHTML=`<h3>raw capture</h3><code>${e.raw_text}</code>
 <h3>parsed</h3>buyer ${e.buyer} · sku ${e.sku} · qty ${e.qty} · ${e.event_type} · ${e.status}
 <h3>correct</h3>field
 <select id=f><option>sku<option>qty<option>variant<option>buyer</select>
 value <input id=v> <input id=rz value=reason placeholder=reason>
 <button onclick=corr()>apply</button>
 <h3>decide</h3><button onclick="act('accept')">accept</button>
 <button onclick="act('reject')">reject</button><button onclick="act('needs_info')">needs_info</button>
 <button onclick="act('cancel')">cancel</button><button onclick="act('fulfilled')">fulfilled</button>
 <h3>corrections</h3>`+e.corrections.map(c=>`${c.field}: ${c.old_value}→${c.new_value} (${c.source})`).join('<br>')}
async function act(a){await j('/api/action',{method:'POST',body:JSON.stringify({event_id:cur,action:a})});load()}
async function corr(){await j('/api/action',{method:'POST',body:JSON.stringify({event_id:cur,action:'correct',
 field:f.value,value:v.value,reason:rz.value})});show(cur);load()}
async function doExport(){let r=await j('/api/export',{method:'POST'});alert('exported to '+r.dir)}
boot()</script>"""


class _Handler(BaseHTTPRequestHandler):
    store: LedgerStore = None            # injected
    worlds: List = []
    export_dir: str = "log/review_export"

    def log_message(self, *a):           # keep the demo quiet
        pass

    def _send(self, obj, ctype="application/json"):
        body = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(_PAGE, "text/html")
        if u.path == "/api/sellers":
            rows = self.store.db.execute("SELECT seller_id,cohort FROM sellers ORDER BY seller_id")
            return self._send([dict(r) for r in rows])
        if u.path == "/api/events":
            return self._send(self.store.events(q["seller"][0]))
        if u.path.startswith("/api/event/"):
            return self._send(self.store.get_event(u.path.rsplit("/", 1)[1]) or {})
        if u.path == "/api/ledger":
            led = self.store.current_ledger(q["seller"][0])
            return self._send([{"buyer": b, "sku": s, "qty": qn}
                               for (b, s), qn in sorted(led.items())])
        self._send({"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}") if n else {}
        if u.path == "/api/action":
            return self._send(self._action(body))
        if u.path == "/api/export":
            from core.review import export_session, receipt_from_store
            rcpt = receipt_from_store(self.store, self.worlds, seed="ui")
            export_session(self.store, self.worlds, rcpt, self.export_dir)
            return self._send({"dir": self.export_dir, "receipt": rcpt})
        self._send({"error": "not found"})

    def _action(self, b: Dict):
        eid, action = b.get("event_id"), b.get("action")
        try:
            if action == "correct":
                self.store.correct(eid, b["field"], b["value"], b.get("reason", ""), "human")
            else:
                self.store.transition(eid, {"accept": ACCEPTED, "reject": REJECTED,
                                            "needs_info": NEEDS_INFO,
                                            "cancel": CANCELLED,
                                            "fulfilled": FULFILLED}[action])
            return {"ok": True}
        except Exception as e:                       # surface, don't crash the UI
            return {"ok": False, "error": str(e)}


def make_server(store: LedgerStore, worlds: List, host: str = "127.0.0.1",
                port: int = 0, export_dir: str = "log/review_export") -> ThreadingHTTPServer:
    handler = type("H", (_Handler,), {"store": store, "worlds": worlds,
                                      "export_dir": export_dir})
    return ThreadingHTTPServer((host, port), handler)
