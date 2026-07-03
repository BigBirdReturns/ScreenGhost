# The Pop-proof demo

An adversarial proof theater. You supply the seed and try to make it lie. The
population is generated fresh from your seed (no golden replay), the real
evaluators run, and every failure your seed produces is attributed to exactly
one cause. It claims neither hardware nor business proof — `[2b]` and `[3]` are
scoped out by decision, not left as unpaid debt.

## Run it

```bash
python examples/pop_proof_demo.py --seed "pop-picks-this-live" --sellers 1000 --mode adversarial
python examples/pop_proof_demo.py --seed "anything"           --sellers 50   --cha-mode
```

Any seed works. A different seed builds a different labeled world and prints a
different failure surface.

## What prints

| act | shows |
|-----|-------|
| I   | the ten objection classes + the three live rules; ends on **"Pick a seed. Pick a mode. Try to make it lie."** |
| II  | the world built from *your* seed — sellers, cohorts, total vs order-bearing messages (denominators kept separate) |
| III | pipeline receipt `[1]` — recall, precision, latency, corruptions, ledger reproduction |
| IV  | **failure attribution** — every miss/false-positive your seed produced, single-caused into `{resolver, row_grouping, dedupe, window_loss, unicode_corruption}`; header echoes your seed |
| V   | boundary conditions — busy / hot / firehose / window-overflow / pathological overlap |
| VI  | objection matrix — allowed claim, forbidden claim, next proof per objection |
| VII | final ledger + verdict |

## What a valid failure looks like

Failures are the point, not a defect:

- **Act III/IV** — adversarial suites bleed: spelling recall drops, stock-out and
  repeat-CF inject false positives. All attributed to `resolver` (parser/catalog),
  because the capture-seam recall delta is ~0.
- **Act V** — `firehose past window`, `visible-window overflow`, and
  `pathological row overlap` print **FAILS**. That is correct behavior.

## What makes it self-auditing (it aborts nonzero)

- no boundary failure exercised → `Demo invalid: no boundary failure was exercised.`
- adversarial labels present but zero bleed → `Demo invalid: adversarial suites produced no bleed. Attribution table not exercised.`
- an error counted under two causes → `Demo invalid: multi-cause attribution detected.`
- any `ARCHITECTURAL`/`FROZEN` row upgraded to `PASS` → `Demo invalid: forbidden status upgrade attempted.`

A demo that can pass by not testing is the exact failure this stack exists to
make impossible.

## `--cha-mode` ends on exactly three lines

```
The objection named ten problems and asserted one verdict.
Here are the ten, separated.
The verdict does not survive the separation.
```

## For a skeptical reader

> Run it with any seed.
>
> The demo does not claim hardware proof.
> The demo does not claim business proof.
> It separates the objection into its component claims and prints what each one
> can and cannot prove.
>
> If the system fails, the failure is attributed.
> If the boundary is not exercised, the demo aborts.
> If a category is frozen, it stays frozen.
>
> The point is not that everything is green.
> The point is that the original verdict does not survive separation.

A canonical transcript is saved at
[`examples/receipts/pop_proof_demo_seed_pop-picks-this-live.txt`](examples/receipts/pop_proof_demo_seed_pop-picks-this-live.txt).
