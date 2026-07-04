"""Live LINE webhook receiver — the last mile to a real [2b-2] receipt.

    export LINE_CHANNEL_SECRET=...        # your OA channel secret, NEVER in code
    python examples/line_live_receiver.py # listens on :8080/webhook

Then point your LINE Official Account's webhook URL (LINE Developers Console)
at a public tunnel to this port and send the OA a message. Each verified inbound
message is turned into candidates + order events by the SAME pipeline every
other surface uses, and appended to examples/receipts/line_live.txt.

This sandbox cannot receive inbound webhooks, so run it on your own machine.
Read-only: only the channel SECRET is used (signature verification); the send
token is never touched. See docs/LINE_LIVE_INTEGRATION.md.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.line_webhook import process

RECEIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "receipts", "line_live.txt")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        sig = self.headers.get("X-Line-Signature", "")
        secret = os.environ["LINE_CHANNEL_SECRET"]
        result = process(body, sig, secret)
        # Always 200 so LINE does not retry; the verdict is in our receipt.
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
        if not result["verified"]:
            print(f"[reject] {result['reason']}")
            return
        with open(RECEIPT, "a", encoding="utf-8") as f:
            for c in result["candidates"]:
                f.write(json.dumps({"surface": c.source_surface, "sender": c.sender,
                                    "text": c.raw_text, "payload": c.payload_type,
                                    "exact": c.unicode_ok}, ensure_ascii=False) + "\n")
            for o in result["orders"]:
                f.write(json.dumps({"ORDER": o}, ensure_ascii=False) + "\n")
        print(f"[ok] {len(result['candidates'])} candidates, "
              f"{len(result['orders'])} order(s) -> {RECEIPT}")

    def log_message(self, *a):   # quiet default access log
        pass


def main():
    if not os.environ.get("LINE_CHANNEL_SECRET"):
        sys.exit("refusing to start: set LINE_CHANNEL_SECRET (never hard-code it)")
    port = int(os.environ.get("PORT", "8080"))
    print(f"listening on :{port}/webhook — read-only, signature-verified")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
