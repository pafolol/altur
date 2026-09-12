# Animation plans

| # | Plan | Severity | Status |
|---|---|---|---|
| 001 | [Make the lock break, not shrink](001-lock-break-choreography.md) | HIGH | DONE |
| 002 | [Give the lock hover and press feedback](002-lock-hover-press.md) | HIGH | DONE |
| 003 | [Fade the transcript when the caller changes](003-demo-transcript-crossfade.md) | LOW | DONE |

Execution order: 001, 002, 003. No dependencies between them.

Noted, not planned: the seven hand-typed easing curves could become `--ease-*` tokens, and the
tap-hint pulse animates `box-shadow`. Both are deliberate prototype tunings on rarely-hit
elements, so they are left as they are.
