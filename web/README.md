# ISISI — landing page

Vite + React 18 + TypeScript, Tailwind v4 (CSS-first `@theme`), `motion` for scroll
values, `lenis` for smooth scroll. The single-file prototype this was ported from is
kept verbatim at [`prototype/index.html`](prototype/index.html); in dev it is also
served at `http://localhost:5173/prototype/index.html` for side-by-side checks.

## Run

```bash
npm i && npm run dev
```

`npm run build` type-checks (`tsc --noEmit`) and builds to `dist/`; `npm run preview`
serves the build.

## Turntable frames

The 120 frames are rendered from [`blender/padlock.blend`](blender/padlock.blend) into

```
public/turntable/frame_0001.webp … frame_0120.webp
```

They ship with the repo (2.7 MB total, ~22 KB a frame). To re-render after changing the
scene:

```bash
npm run render
```

(That calls `/Applications/Blender.app/Contents/MacOS/Blender -b blender/padlock.blend
--python blender/render_turntable.py`; edit the path in `package.json` if Blender lives
elsewhere. Rendered with Blender 5.2 LTS, Eevee, about a minute on an Apple Silicon
laptop.) The script pins 900×1125, transparent film, WebP RGBA at quality 90, and takes
`-- --frames 1,60 --scale 50 --out DIR --format PNG` for quick tests.

- The canvas is 900×1125 internally and CSS-sized to the 196px lock box (150px under
  600px wide): sharp on retina without a 2x payload.
- All frames are fetched and `decode()`d before the first draw; the canvas then fades in.
  Until then the stage shows only the glow.
- If any frame fails to load, the CSS 3D lock from the prototype renders instead.
- Frame count lives in `FRAMES` and the number of revolutions in `TURNS` (2, eased, like the
  prototype's CSS lock) at the top of [`src/scenes/LockSequence.tsx`](src/scenes/LockSequence.tsx).
  The spin completes at 70% of the hero (`SETTLE` in [`src/scenes/ActLock.tsx`](src/scenes/ActLock.tsx)) and holds.
- Nothing below the hero exists until the lock is tapped: `App` mounts the name act and the
  reading sections on `onBroken`, and the nav link stays invisible until then. There is no
  scroll-triggered break any more; the page simply ends at the settled lock.
- Memory note: 120 decoded 900×1125 bitmaps are ~480 MB if the browser keeps them all
  resident. Desktop browsers cope; if low-end phones stutter, halve the frame size.

## Copy

Everything editable lives in `src/data/`, never inline in JSX:

| File | What |
|---|---|
| [`src/data/copy.ts`](src/data/copy.ts) | nav, hero, signals cards + diagram labels, demo chrome, pipeline steps + timeline, endpoint request/response + specs, closing + footer |
| [`src/data/calls.ts`](src/data/calls.ts) | the two Spanish transcripts, analyst notes, per-signal scores, verdicts |
| [`src/data/acronym.ts`](src/data/acronym.ts) | the five meanings; the wordmark letter is each word's initial |

Paragraph-level fields (`lede`, `text`, `note`, list items, spec text) may contain inline
HTML — `&nbsp;`, `<i>`, `<code>` — exactly as in the prototype; they are injected via
[`src/lib/Html.tsx`](src/lib/Html.tsx). Headings and labels are plain strings.

Every number on the page is read from the repository's own reports (README.md Results,
reports/ACOUSTIC_LEARNING_SUMMARY.md, behaviour/reports/, semantic/AUDIT.md). The demo's two
callers are real held-out calls (`call_569ffb0869eb`, `call_0847d7417bb1`) with the scores every
layer produced for them; the long version with sources is `reports/ISISI_PRESENTER_GUIDE.pdf`.
The page `<title>` and `<meta name="description">` are in [`index.html`](index.html).

## Reduced motion

`prefers-reduced-motion: reduce` gives: no scroll scrub, no bob, no particles, a static
lock (last turntable frame, or the tilted CSS lock), all five acronym meanings stacked
and visible, native scrolling (Lenis is not created), the sea drawn once. Tapping the
lock fades it and jumps to the name act.

To test without touching OS settings, open `http://localhost:5173/?reduced-motion`.

## Small conveniences

- The ISISI logo in the nav is a link back to the top (`#hero`), smooth under Lenis.
- While the 120 frames decode, the lock box shows a skeleton silhouette with a frame counter
  (`Loading the lock 37 / 120`); it fades under the canvas once decoding finishes. Static
  under reduced motion.
- The endpoint example has a Copy button; it uses the Clipboard API and falls back to a
  selection copy where that is unavailable (plain http) or denied.

## Admin panel

`http://localhost:5173/admin/` (and `dist/admin/` after a build) is a dashboard for the
detector: six KPI tiles, verdicts by hour, the confidence distribution, latency p50, the
calls stream with per-signal scores and a detail panel, an endpoint tester, and endpoint
health.

It is frontend only. Nothing leaves the browser: the data layer in
[`src/admin/api.ts`](src/admin/api.ts) is a seeded mock behind the `IsisiApi` interface
(`calls`, `stats`, `health`, `detect`), badged "Sample data · not connected" in the header.
When a backend exists, implement that interface with fetch calls and swap it in there;
the components don't change. KPIs, the hourly stack and the histogram are derived on the
client from the calls list (`summarize()`), so a backend only has to return calls to light
most of the page. Labels live in [`src/admin/copy.ts`](src/admin/copy.ts). Chart colours
(human, synthetic, abstained) were validated for colour-vision safety and contrast against
the card surface; every chart has a hover readout and a table view.

## Motion notes

Animation audit plans live in [`plans/`](plans/README.md). Executed so far: the lock break
(squash, snap, evaporate, radial flash, particles with a gravity leg), hover and press
feedback on the lock, and a fade on the demo transcript when the caller changes.

## Layout

```
src/scenes/    LockSequence (canvas scrubber, CSS-lock fallback)
               ActLock (hero choreography, tap, burst)
               ActName (wordmark, meanings, ticks)
src/sections/  Signals, TrapDemo, Pipeline, Endpoint, Closing
src/data/      copy.ts, calls.ts, acronym.ts
src/lib/       useLenis, useReducedMotion, Html, math
src/index.css  @theme tokens + the prototype's stylesheet
```

Tokens: `--color-abyss`, `--color-deep`, `--color-panel`, `--color-foam`, `--color-body`,
`--color-dim`, `--color-glint` (plus `--color-edge` for the hairlines), `--font-display`
(Instrument Serif: h1, h2, h3, the wordmark, set in caps at leading 1), `--font-sans`
(Hanken Grotesk: running text, buttons, nav), `--font-mono` (DM Mono: the small labels,
`pre`, `code`, the verdict). Use them as `bg-abyss`, `text-glint`, `fill-body`,
`font-mono`… A hex literal in a `className` is a bug.

## Decisions you did not specify

1. **No ocean.** The prototype's canvas sea was never visible (its opaque body background
   painted over the fixed canvas), and you asked for it to go, so the sea, the click ripple
   and the dim overlay under the reading sections are removed. The page background is the
   flat abyss the prototype showed.
2. **The prototype moved to `prototype/index.html`.** Vite needs the root `index.html` as
   its entry.
3. **The tuned CSS stayed CSS.** The prototype's stylesheet is kept nearly verbatim inside
   Tailwind's `@layer components`, with colours and fonts pointed at the `@theme` tokens.
   Translating every `clamp()` into arbitrary-value utilities would have risked the
   fidelity you asked for. Tailwind utilities are used where they are natural (SVG
   `fill-*`/`stroke-*`, `font-sans`).
4. **Turntable vs. CSS lock transforms.** The frames have the camera baked in, so with the
   turntable the spin wrapper only scales (0.86 → 1). The `rotateX` tilt and the 720°
   `rotateY` apply only to the CSS fallback. The break animation hits whichever is showing.
5. **Nothing is drawn while frames decode** (per your spec). On a slow connection that is a
   few seconds of empty stage; the glow is still there. If you would rather show the CSS
   lock meanwhile, render `<CssLock/>` in the `loading` state of `LockSequence`.
6. **Reduced motion is a class, not a media block.** `useReducedMotion` puts
   `reduced-motion` on `<html>` from the media query or the `?reduced-motion` URL flag, and
   the stylesheet keys off that class, so one CSS block serves both. Under reduced motion
   all five ticks light up (the prototype left them unlit under an all-visible stack) and
   the dim overlay still follows scroll, since it is opacity, not motion.
7. **Frame decoding falls back to the load event** if `decode()` rejects (Safari does that
   under big batches); only a real load failure switches to the CSS lock.
8. **Lenis.** Created only when motion is allowed, with `anchors: true` so the nav's
   `#api` link scrolls smoothly; the post-tap scroll uses `lenis.scrollTo`. Its stylesheet is
   imported in `index.css`. The prototype's `scroll-behavior: smooth` is dropped, as Lenis
   requires.
9. **The endpoint JSON is data.** `endpoint.request` / `endpoint.response` are typed
   objects rendered by a ten-line highlighter, so the numbers are editable and carry TODOs.
   Diagram labels and timeline marker positions moved to `copy.ts` for the same reason.
10. **Section ids** (`#hero`, `#name`, `#how`, `#demo`, `#pipeline`, `#api`) match the
    prototype. The hero scrolls to its next sibling after the tap rather than to a
    hard-coded id.
11. **Burst particles are plain DOM nodes** appended into an empty React-owned `div`, as in
    the prototype; putting 58 one-shot dots through React state would only add re-renders.
12. **Versions.** React 18.3.1 as asked (React 19 is current), TypeScript 5.9 rather than
    the new 7.x line, Vite 8, Tailwind 4.3, motion 13, lenis 1.3.
13. **Not included:** ESLint/Prettier, tests, self-hosted fonts (Google Fonts link kept),
    the video scrubber path from the prototype (replaced by the frame sequence).
14. **Name act arrival.** After the tap, the five letters rise in on a timer (1.1 s after the
    tap, 70 ms stagger) as the hand-off scroll lands, the first word is live on arrival, and
    a "Scroll to read the name" cue sits at the bottom until you move a step. The section is
    460vh (one word per 72vh of scroll, a flick each). Each letter is a `<button>` that jumps
    to the middle of its word's stretch. Reduced motion: everything stacked, no cue.
15. **Type direction (after the trials).** Titles, subtitles and the wordmark are Instrument
    Serif at 400 in caps with line-height 1 and no tracking, after mijares.mx; the small
    labels (signal/step/spec labels, speaker tags, bar labels, timeline ticks, tap hint,
    footer, nav strap, SVG diagram labels) are DM Mono in caps; running text, buttons and
    the nav are Hanken Grotesk, the Apple register. One mono serves labels and JSON alike.
    The label rules sit at the end of the components layer so they win over the per-block
    sizes above them. Swap any of the three in `@theme` and the page follows.
16. **Turntable render.** `padlock.blend` (the render-ready scene from WhatsApp) is copied
    into `blender/`; `padlock_source.blend` is the same scene and was left out. The scene's
    key on `Lock_Body` reaches 360° at frame 121 (loop convention), so frame 120 would sit
    3° short of front, which is the pose held while the tap hint shows. The render script
    samples the curve so frames 1 and 120 are both exactly front. The scrubber plays the
    sequence twice with the prototype's easing (`TURNS = 2`), so the motion matches the
    CSS lock's 720°. The scene still contains Blender's default Cube/Light/Camera
    collection, unlinked from the view layer, so it does not render.
