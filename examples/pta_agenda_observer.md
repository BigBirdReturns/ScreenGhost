# PTA Agenda Observer — Screen Ghost's first production consumer

[`axm-tools/pta-tracker`](https://github.com/BigBirdReturns/axm-tools) is a
zero-maintenance legislation tracker for a school-district PTA. It runs
entirely in GitHub Actions — which means it has exactly the problem this
project was built for: the district's board-agenda system (Simbli) sits
behind Incapsula bot protection. A datacenter scraper will never get in.
A parent looking at the page on their phone always will, because the screen
is the one interface the vendor cannot revoke.

So the tracker doesn't scrape Simbli. It exposes a seam:
`pta-tracker/data/observed.json` — items collected out-of-band, in the same
schema as its feed items, merged by the nightly job without keyword
filtering. Anything that can read a screen and write JSON can fill it.
That's observer mode.

## The loop

1. **Phone shows the agenda.** Open the Simbli meeting listing in the
   phone's browser (bookmark it), tap into the upcoming meeting's agenda.
   Navigator mode can do the opening for you if you want
   (`--goal "Open Chrome and go to the AUSD board agenda"`), but for a
   monthly task, a human tap is fine — the win is the extraction.
2. **Observer mode reads it.** `observe()` turns the visible agenda into
   structured elements. Scroll, observe again, repeat until the agenda is
   covered.
3. **Convert to tracker items and commit.** Each substantive agenda line
   becomes one `observed.json` item. Commit it to the axm-tools repo; the
   nightly Action merges it into the tracker page with everything else.

## Runnable sketch

```python
"""Read the on-screen Simbli agenda into pta-tracker's observed.json.

Run with the phone attached over USB and showing the agenda page.
Scroll manually between captures, or wire in swipes via the executor.
"""
import json
from pathlib import Path

from screenghost import observe

SIMBLI = (
    "https://simbli.eboardsolutions.com/SB_Meetings/SB_MeetingListing.aspx"
    "?S=36030512"
)
OUT = Path("../axm-tools/pta-tracker/data/observed.json")  # local checkout

state = observe(save_screenshot=True)  # keep the pixels as provenance

items = []
for el in state.elements:
    text = " ".join(filter(None, [el.label, el.value or ""])).strip()
    if len(text) < 12:
        continue  # buttons, page chrome, dates without substance
    items.append(
        {
            "source": "Simbli agenda (observed)",
            "title": text[:200],
            "link": SIMBLI,
            "published": state.timestamp.date().isoformat(),
            "priority": "hot",
            "scope": "district",
        }
    )

existing = json.loads(OUT.read_text()) if OUT.exists() else {"items": []}
existing["items"] = items  # replace: each capture supersedes the last
OUT.write_text(json.dumps(existing, indent=2))
print(f"wrote {len(items)} agenda items — review, then commit & push")
```

Then:

```bash
cd ../axm-tools
git add pta-tracker/data/observed.json
git commit -m "pta-tracker: observed Simbli agenda for the upcoming meeting"
git push
```

The next nightly run folds the items into `items.json`; the PTA page shows
them tagged AUSD/hot, and the monthly report generator includes them.

## Honest limitations

- **Review before you commit.** Moondream2 misreads screens; the tracker's
  own provenance rule applies doubly here — the screenshot saved alongside
  the run is your check.
- The `len(text) < 12` filter is a heuristic, not understanding. Expect to
  delete a stray navigation label now and then.
- This is a monthly, human-in-the-loop task by design. The point isn't
  removing the human — it's that the human's phone does the reading, and
  the tracker's data outlives the skim.

This is the pattern in miniature: the vendor blocked the protocol, so the
integration moved to the interface they can't block. No adapter to
maintain, nothing for Incapsula to fingerprint — just a screen, observed.
