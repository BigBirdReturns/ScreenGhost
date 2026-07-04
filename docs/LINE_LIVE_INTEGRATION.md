# Going live on LINE — the honest path to a [2b-2] receipt

This is the runbook that converts `line_oa` from *event-schema* (a decided
strategy against representative payloads) into a **live receipt**: a real message
sent to a real LINE Official Account, received over the real webhook, turned into
candidates and order events by the same pipeline every other surface uses.

Everything up to the live send is built and tested (`core/line_webhook.py`,
`tests/test_line_webhook.py`). The live send needs a channel and a public
endpoint, which are the operator's — not this repo's.

## What you need

- A LINE **Official Account** with a **Messaging API** channel (LINE Developers
  Console, free). A *personal* LINE account will not work — it has no sanctioned
  API to read messages, and the client-automation route violates LINE's ToS.
- The channel **secret** (Console → your channel → Basic settings). That is all
  the read path uses.
- A public HTTPS endpoint to the receiver (a tunnel such as an SSH reverse
  tunnel or an ngrok-style service pointed at the receiver's port).

## Security — non-negotiable

- The channel secret is a credential. **Never** paste it into a chat, a commit,
  or any file in the repo. It lives in an environment variable on your machine:
  `export LINE_CHANNEL_SECRET=...`
- The **access token** (which sends/pushes messages) is deliberately **not used
  and not imported** anywhere in `core/line_webhook.py`. The ghost reads; it
  cannot send, because the send credential was never wired. Read-only is enforced
  by absence, not by a promise — `tests/test_line_webhook.py` asserts no token
  reference exists.

## Steps

1. `export LINE_CHANNEL_SECRET=...` (your channel secret; never hard-coded).
2. `python examples/line_live_receiver.py` — listens on `:8080/webhook`,
   read-only, signature-verified. It refuses to start without the env var.
3. Expose it: point a public HTTPS tunnel at port 8080.
4. In the LINE Developers Console, set the channel's **Webhook URL** to
   `https://<your-tunnel>/webhook` and enable "Use webhook".
5. Send your OA a message (e.g. `CF C01 x2 ค่ะ`). The receiver verifies the
   `X-Line-Signature`, maps the event to a candidate, runs it through OrderBook,
   and appends the result to `examples/receipts/line_live.txt`.

A message with a bad/absent signature is rejected and produces no candidate — an
unverifiable request is never trusted.

## What this proves, and what it still does not

- **Proven once you run it:** a real LINE message, over the real signed webhook,
  becomes an exact-text candidate and an order event with no app, no scrape, no
  OCR. The "Thai OCR is slow and wrong" objection never applies, because there is
  no OCR on this path — the bytes arrive exact from the API.
- **Still frozen:** fleet scale (`[2b-1]`), business/seller-hour outcomes
  (`[3]`), and any surface other than the one you connected. One live receipt is
  a live receipt — not a claim about 50k of them. See
  [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md).

## Why this shrinks the hardware worry

The webhook arrives on an ordinary server. There is no phone in this path at all
— see the "one account, many instances" note in
[`SURFACE_CAPABILITY_MATRIX.md`](SURFACE_CAPABILITY_MATRIX.md). The image of a
warehouse of 50k phones was answering a question the API path doesn't ask.
