# The admin panel, wired to the real detector

Merged from the `frontend` branch. It lives at `web/` rather than `frontend/`, because `frontend/` was
already the fusion console — nothing was renamed to make room.

`src/admin/api.ts` said what to do, and this is it:

> When a backend exists, implement the same interface with fetch calls and swap it in here; nothing in
> the components needs to change.

So nothing in the components changed. `src/admin/api.live.ts` implements the same `IsisiApi` against
`src/server.py`, the seeded mock is still there as `mockApi`, and `api.ts` picks between them on one line.

## Running it

```
python src/server.py --port 8000                      # the detector
cd web && VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

Open `/admin/`. The header pill reads **LIVE** instead of SAMPLE, and every number under it came from a
real detection.

**Leave `VITE_API_URL` unset and the mock answers**, exactly as before — the landing page still builds and
demos with no backend running, which is why the mock was kept rather than deleted.

## What each part of the panel is now

| the panel asks for | it gets |
|---|---|
| `detect(audio)` | a real `POST /detect` — the two-stage fusion, verifier included when the primaries do not settle it |
| `calls('24h')` | every verdict the endpoint has produced, from the call log |
| `stats('24h')` | p50 latency per 30-minute bucket, from the same log |
| `health()` | the live layer registry, the deployed model's real calibration date, and 24 h uptime |

`evidence.conversational` is the behaviour layer — the panel's name for it, kept as-is rather than
renamed on either side. A layer that abstained, or that the gate never asked, reads **0**: it had no
opinion, and inventing one would be worse than showing none.

## The call log

`calls()` and `stats()` need something the detector does not have on its own: memory. `src/store.py` is
that — SQLite, one file at `outputs/fusion/calls.db`, created on first use, no server and no credentials.
Every `/detect` appends one row after the response is computed, so a write failure can never cost a
detection.

**No audio is stored.** Only scores, verdicts and timings. The audio arrives, is scored, and is gone —
the dataset terms say not to redistribute the recordings, and a database of them is exactly that.

```
python src/store.py --hours 24      # read the log
python src/store.py --prune         # drop rows older than 30 days
```

## Two fields the server refuses to fake

The panel's health card has a row for each, and both are answered honestly rather than dressed up:

- **streaming: false** — this detector scores a complete call. It does not decide as audio arrives.
- **queue_depth: 0** — requests are served synchronously. There is no queue to be deep.

## One field it does compute, rather than guess

`decided_at_s` — "seconds into the call when it stopped being ambiguous". The acoustic layer scores 4 s
chunks of caller speech and averages their log-odds, so the running average after *k* chunks is exactly
what it would have answered having heard only those *k*. `store.decided_at()` walks that forward and
returns the end of the first chunk after which the verdict never flips again. It is `None` when the layer
abstained or never settles — not a placeholder.

## `trap` and `queue`

- **`queue`** is the caller's own label for where a call came from. `POST /detect` accepts an optional
  `"queue"` field; anything that just posts audio lands in `api`.
- **`trap`** is always `none` today. The semantic module *does* have a trap bank (`semantic/traps.py`
  locates the agent's scripted moments — the deliberately wrong digit, the interruption, the product the
  caller may not have), but its `/detect` does not report which ones the caller fell for. Exposing that
  is a change to that module, so it is not faked here.
