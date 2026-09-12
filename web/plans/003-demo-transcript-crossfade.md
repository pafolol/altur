# 003 — Fade the transcript when the caller changes

- **Status**: DONE
- **Commit**: no git repository; written 2026-09-12
- **Severity**: LOW
- **Category**: Missed opportunities (jarring state change)
- **Estimated scope**: 2 files, ~5 lines

## Problem

Switching "Real caller" / "Cloned voice" eases the three bars over 800 ms but swaps the
transcript instantly, so the left half of the demo teleports while the right half moves.

```tsx
// src/sections/TrapDemo.tsx:35 — current
<div className="script">
```

## Target

Remount the transcript on mode change and fade it up 4 px over 220 ms with the burst's ease-out.

```tsx
<div className="script" key={mode}>
```

```css
/* src/index.css — target, next to .script */
.script{animation:fadeup .22s cubic-bezier(.05,.7,.2,1) both}
@keyframes fadeup{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
```

## Steps

1. Add `key={mode}` to the `.script` div in `src/sections/TrapDemo.tsx`.
2. Add the animation and keyframes in `src/index.css` beside the existing `.script` rule.

## Boundaries

- Do NOT touch the bar transition or the verdict block.

## Verification

- **Mechanical**: `npx tsc --noEmit` clean.
- **Feel check**: toggle the caller: the transcript fades up as the bars move; under `?reduced-motion` it swaps instantly (the global animation kill handles it).
- **Done when**: the transcript no longer snaps while the bars ease.
