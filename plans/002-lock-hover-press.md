# 002 — Give the lock hover and press feedback

- **Status**: DONE
- **Commit**: no git repository; written 2026-09-12
- **Severity**: HIGH
- **Category**: Missed opportunities / Physicality (press feedback)
- **Estimated scope**: 1 file, ~10 lines

## Problem

The lock is the only thing on the page you can act on, and it does nothing when the pointer
reaches it or presses it. The `<button class="tapzone">` is invisible and sits after the lock
in the DOM, so the lock itself has never had a hover or active state.

```css
/* src/index.css:114 — current */
.tapzone{position:absolute;left:50%;top:50%;width:260px;height:300px;transform:translate(-50%,-50%);
  background:none;border:0;cursor:pointer;z-index:5;border-radius:24px}
.tapzone[disabled]{cursor:default}
```

## Target

Hover lifts the lock 3% and brightens its pool of light; press squashes it to 0.96 in 160 ms.
Hover is gated to real pointers, and both are dropped under reduced motion. `:has()` reaches
the lock from the button that follows it.

```css
/* src/index.css — target, placed right after .tapzone[disabled] */
.lock3d{transition:transform .35s cubic-bezier(.2,.7,.2,1)}
.glowunder{transition:filter .35s ease}
@media (hover:hover) and (pointer:fine){
  html:not(.reduced-motion) .stage:has(.tapzone:not([disabled]):hover) .lock3d{transform:scale(1.03)}
  html:not(.reduced-motion) .stage:has(.tapzone:not([disabled]):hover) .glowunder{filter:blur(14px) brightness(1.45)}
}
html:not(.reduced-motion) .stage:has(.tapzone:not([disabled]):active) .lock3d{transform:scale(.96);transition-duration:.16s}
```

## Repo conventions to follow

- Reduced motion is the `html.reduced-motion` class (set by `src/lib/useReducedMotion.ts`), not a media query.
- `.glowunder` already carries `filter:blur(14px)`; the hover value must keep the blur.
- The break animation (WAAPI, `fill: forwards`) targets `.lock3d` and wins over these transitions once it runs.

## Steps

1. Add the block above to `src/index.css` after `.tapzone[disabled]{cursor:default}`.

## Boundaries

- Do NOT move or restyle `.tapzone`; do NOT change `.spin` (its transform is scroll-driven).
- Do NOT add dependencies.

## Verification

- **Mechanical**: `npm run build` clean.
- **Feel check**: hover the lock: it lifts and the glow warms; press and hold: it dips; release: it returns. After the break nothing reacts (the button is disabled). `?reduced-motion`: no hover or press movement.
- **Done when**: all of the above at 1440 wide with a mouse.
