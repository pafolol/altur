# 001 — Make the lock break, not shrink

- **Status**: DONE
- **Commit**: no git repository; written 2026-09-12
- **Severity**: HIGH
- **Category**: Physicality & origin / Easing & duration
- **Estimated scope**: 2 files, ~40 lines

## Problem

The tap on the lock is the page's one high-emotion moment and it reads as the lock being
sucked inward: it swells to 1.13 then collapses to `scale(.2)`, with no wind-up before the
release. The flash is a flat full-viewport overlay at 0.7 opacity that decays for 950 ms
(a screen blink rather than light coming out of the lock), and the particles fly one straight
leg with no gravity, so they read as confetti rather than debris.

```ts
// src/scenes/ActLock.tsx:49 — current
lock?.animate([{ transform: 'scale(1)', opacity: 1, filter: 'brightness(1)' },
               { transform: 'scale(1.13)', opacity: 1, filter: 'brightness(3)', offset: 0.18 },
               { transform: 'scale(.2)', opacity: 0, filter: 'brightness(3.4)' }],
              { duration: 660, easing: 'cubic-bezier(.3,0,.2,1)', fill: 'forwards' })
glowRef.current?.animate([{ opacity: 1 }, { opacity: 2.4, offset: 0.2 }, { opacity: 0 }], { duration: 1200, fill: 'forwards' })
flashRef.current?.animate([{ opacity: 0 }, { opacity: 0.7, offset: 0.12 }, { opacity: 0 }], { duration: 950, easing: 'ease-out' })
// ring: scale(.2) → scale(16) over 1200 ms; dots: one leg, translate + scale(0), 950–1900 ms
// hand-off: setTimeout(scrollNext, 580)
```

```css
/* src/index.css — current */
.flash{position:absolute;inset:0;background:var(--color-glint);opacity:0;pointer-events:none;z-index:9}
```

## Target

Three beats in 640 ms: a 90 ms squash (ease-in, the wind-up), a 110 ms snap to 1.16 with a
−3° kick and the brightness spike, then a 440 ms evaporate: the lock grows to 1.4, drifts up
22 px and fades. The flash becomes a radial burst centred on the burst origin (50% 44%, the
same point `.burst` uses), peaking at 0.6 in the snap and gone by 700 ms. Particles get a
second leg where gravity takes over (46 px drop, easing in-out). The ring caps at 12×. The
scroll hand-off waits for the evaporate (620 ms).

```ts
// src/scenes/ActLock.tsx — target
const out = 'cubic-bezier(.05,.7,.2,1)'      // the burst's ease-out, already used by the dots
lock?.animate([
  { offset: 0,    transform: 'scale(1)',                                    opacity: 1, filter: 'brightness(1)',  easing: 'cubic-bezier(.4,0,1,1)' },
  { offset: 0.14, transform: 'scale(.94)',                                  opacity: 1, filter: 'brightness(.9)', easing: out },
  { offset: 0.31, transform: 'scale(1.16) rotate(-3deg)',                   opacity: 1, filter: 'brightness(3)',  easing: out },
  { offset: 1,    transform: 'scale(1.4) translateY(-22px) rotate(1deg)',   opacity: 0, filter: 'brightness(4)' },
], { duration: 640, fill: 'forwards' })
glowRef.current?.animate([{ opacity: 1 }, { opacity: 2.6, offset: 0.12 }, { opacity: 0 }], { duration: 1100, fill: 'forwards' })
flashRef.current?.animate([{ opacity: 0 }, { opacity: 0.6, offset: 0.16 }, { opacity: 0 }], { duration: 700, delay: 80, easing: out })
ring.animate([{ transform: 'scale(.2)', opacity: 0.9 }, { transform: 'scale(12)', opacity: 0 }],
             { duration: 1000, delay: 90, easing: 'cubic-bezier(.1,.8,.2,1)', fill: 'forwards' })
// per dot, with x = cos(ang)*dist and y = sin(ang)*dist*0.82:
dEl.animate([
  { transform: 'translate(0,0) scale(1)', opacity: 1, easing: out },
  { transform: `translate(${x * 0.78}px,${y * 0.78}px) scale(.7)`, opacity: 0.9, offset: 0.55, easing: 'cubic-bezier(.4,0,.6,1)' },
  { transform: `translate(${x}px,${y + 46}px) scale(0)`, opacity: 0 },
], { duration: 900 + Math.random() * 900, delay: 90 + Math.random() * 40, fill: 'both' })
setTimeout(scrollNext, 620)
```

```css
/* src/index.css — target */
.flash{position:absolute;inset:0;opacity:0;pointer-events:none;z-index:9;
  background:radial-gradient(circle at 50% 44%,var(--color-glint) 0%,rgba(127,196,255,.55) 22%,transparent 62%)}
```

## Repo conventions to follow

- All break motion is WAAPI in `breakLock()` in `src/scenes/ActLock.tsx`; keep it there, no state.
- The reduced-motion branch above it (opacity fade, immediate scroll) stays untouched.
- Colours come from `--color-*` tokens in `src/index.css`; the two dot tints are literals by design.

## Steps

1. In `src/scenes/ActLock.tsx`, inside `breakLock()` after the reduced-motion early return, replace the lock, glow and flash `animate` calls with the target above.
2. Replace the ring `animate` call with the target (delay 90, scale 12, 1000 ms).
3. In the dot loop, compute `x` and `y` once, then use the three-keyframe target with the random delay.
4. Change `setTimeout(scrollNext, 580)` to `620`.
5. In `src/index.css`, replace the `.flash` background with the radial gradient.

## Boundaries

- Do NOT touch the scroll choreography (`draw()`), `LockSequence`, or the tap-hint logic.
- Do NOT change dot count, sizes, colours or spread; only their keyframes.
- Do NOT add dependencies.

## Verification

- **Mechanical**: `npx tsc --noEmit` clean; `npm run build` clean.
- **Feel check**: load the page, scroll to the settle, tap. In DevTools run
  `document.getAnimations().forEach(a => a.playbackRate = 0.15)` right after tapping and confirm:
  - the lock dips before it swells (the squash is visible for a few frames);
  - the flash is a bloom around the lock, not a uniform screen change;
  - particles curve downward at the end of their flight;
  - the page starts scrolling only after the lock has gone.
  - `?reduced-motion`: a plain opacity fade, no particles, immediate jump.
- **Done when**: the four checks above hold and the console is clean.
