import { forwardRef, useEffect, useRef, useState } from 'react'
import { useMotionValueEvent, type MotionValue } from 'motion/react'
import { hero } from '../data/copy'
import { clamp, lerp } from '../lib/math'

/** public/turntable/frame_0001.webp … frame_0120.webp — one full revolution, frame 120 = frame 1 */
export const FRAMES = 120
/** Full turns over the scroll, eased like the prototype's CSS lock (it spun 720°). */
const TURNS = 2
const frameUrl = (i: number) => `${import.meta.env.BASE_URL}turntable/frame_${String(i).padStart(4, '0')}.webp`

type Props = {
  progress: MotionValue<number>
  /** Fraction of the hero's scroll at which the spin completes and holds. */
  settle: number
  /** Once the lock is broken the spin stops following scroll. */
  frozen: boolean
  reduced: boolean
}

type Mode = 'loading' | 'frames' | 'css'

/** decode() is what makes the first pass flicker-free; Safari can reject it under a
 *  large batch even though the image loaded fine, so fall back to the load event
 *  and only treat a genuine load failure as missing. */
function decoded(img: HTMLImageElement) {
  return img.decode().catch(() => new Promise<void>((res, rej) => {
    if (img.complete) return img.naturalWidth ? res() : rej()
    img.onload = () => res()
    img.onerror = () => rej()
  }))
}

/**
 * Scroll-scrubbed turntable. Every frame is decoded before the first draw —
 * setting src and drawing immediately makes the browser decode lazily and the
 * lock flickers on first pass. The canvas is 900×1125 internally and CSS-sized
 * to the 196px lock box: sharp on retina without a 2x payload. If any frame
 * fails to load, the CSS 3D lock from the prototype renders instead.
 * The forwarded ref is the element the break animation hits.
 */
export const LockSequence = forwardRef<HTMLDivElement, Props>(function LockSequence(
  { progress, settle, frozen, reduced },
  ref,
) {
  const [mode, setMode] = useState<Mode>('loading')
  const spinRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frames = useRef<HTMLImageElement[] | null>(null)
  const last = useRef(-1)
  const frozenRef = useRef(frozen)
  frozenRef.current = frozen
  const counterRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    let alive = true, done = 0
    const imgs = Array.from({ length: FRAMES }, (_, i) => Object.assign(new Image(), { src: frameUrl(i + 1) }))
    const tick = () => { if (alive && counterRef.current) counterRef.current.textContent = `${++done} / ${FRAMES}` }
    Promise.all(imgs.map((img) => decoded(img).then(tick))).then(
      () => { if (alive) { frames.current = imgs; setMode('frames') } },
      () => { if (alive) setMode('css') },
    )
    return () => { alive = false }
  }, [])

  function drawFrame(idx: number) {
    const cv = canvasRef.current, imgs = frames.current
    if (!cv || !imgs || idx === last.current) return
    last.current = idx
    const ctx = cv.getContext('2d')!
    ctx.clearRect(0, 0, cv.width, cv.height)   // frames are transparent; without this they pile up
    ctx.drawImage(imgs[idx], 0, 0, cv.width, cv.height)
  }

  function paint(p: number) {
    if (frozenRef.current) return
    const t = clamp(p / settle, 0, 1)
    const e = 1 - Math.pow(1 - t, 2.6)
    const spin = spinRef.current
    if (spin) {
      // the turntable has the rotation baked in; the CSS lock is rotated live
      spin.style.transform = frames.current
        ? `scale(${lerp(0.86, 1, e).toFixed(3)})`
        : `rotateX(${lerp(-4, -12, e).toFixed(2)}deg) rotateY(${(e * 720).toFixed(2)}deg) scale(${lerp(0.86, 1, e).toFixed(3)})`
    }
    drawFrame(Math.round(((e * TURNS) % 1) * (FRAMES - 1)))
  }

  // written straight to the DOM — no React state on the scroll path
  useMotionValueEvent(progress, 'change', (p) => { if (!reduced) paint(p) })

  useEffect(() => {
    if (mode === 'loading') return
    if (reduced) {
      if (spinRef.current) spinRef.current.style.transform = mode === 'css' ? 'rotateX(-12deg)' : ''
      drawFrame(FRAMES - 1)
    } else {
      paint(progress.get())
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, reduced])

  return (
    <div className="spin" ref={spinRef}>
      <div className="lock3d" ref={ref}>
        {mode !== 'css' && (
          <div className={mode === 'frames' ? 'lockskel off' : 'lockskel'} role="status" aria-live="polite">
            <i className="s" /><i className="b" />
            {mode === 'loading' && <span className="count">{hero.loading} <span ref={counterRef}>0 / {FRAMES}</span></span>}
          </div>
        )}
        {mode === 'frames' && <canvas ref={canvasRef} className="lockcanvas" width={900} height={1125} aria-hidden="true" />}
        {mode === 'css' && <CssLock />}
      </div>
    </div>
  )
})

/* ═══ the CSS lock ═══════════════════════════════════════════
   43 stacked slices. Switch METAL to 'brass' for the gold lock. */
const METAL = 'steel' as 'steel' | 'brass'
const TONE = METAL === 'brass'
  ? { a: [38, 26, 6], b: [108, 78, 18], c: [226, 190, 96], d: [150, 112, 34], e: [52, 36, 10],
      L: [214, 180, 92], T: [140, 104, 32], R: [46, 32, 9] }
  : { a: [10, 20, 32], b: [32, 58, 86], c: [196, 220, 246], d: [74, 108, 146], e: [16, 32, 52],
      L: [168, 200, 234], T: [70, 104, 142], R: [14, 28, 46] }
const BODY_N = 26, SHACK_N = 17, GAPZ = 1.25, SHACK_Z = 6.6
const shade = (c: number[], d: number) => 'rgb(' + c.map((v) => Math.round(v * d)).join(',') + ')'
const dim = (t: number) => 1 - Math.pow(t, 0.7) * 0.6
function metal(t: number) {
  const k = dim(t)
  return 'linear-gradient(106deg,' + shade(TONE.a, k) + ' 0%,' + shade(TONE.b, k) + ' 16%,' +
    shade(TONE.c, k) + ' 36%,' + shade(TONE.d, k) + ' 54%,' + shade(TONE.e, k) + ' 78%,' + shade(TONE.a, k) + ' 100%)'
}

function CssLock() {
  const shackle = []
  for (let i = SHACK_N - 1; i >= 0; i--) {
    const t = i / (SHACK_N - 1), k = dim(t)
    shackle.push(
      <div key={i} className="slab s" style={{
        borderWidth: 19,
        borderLeftColor: shade(TONE.L, k),
        borderTopColor: shade(TONE.T, k),
        borderRightColor: shade(TONE.R, k),
        transform: 'translateZ(' + (-(SHACK_Z + i * GAPZ)).toFixed(2) + 'px)',
      }} />,
    )
  }
  const body = []
  for (let j = BODY_N - 1; j >= 0; j--) {
    const u = j / (BODY_N - 1)
    body.push(
      <div key={j} className="slab b" style={{ background: metal(u), transform: 'translateZ(' + (-j * GAPZ).toFixed(2) + 'px)' }} />,
    )
  }
  return (
    <>
      <div className="part shacklePart">{shackle}</div>
      <div className="part bodyPart">
        {body}
        <div className="face" style={{ transform: 'translateZ(1.5px)' }}>
          <div className="keyhole" />
          <div className="shine" />
        </div>
      </div>
    </>
  )
}
