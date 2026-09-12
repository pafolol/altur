import { useEffect, useRef, useState } from 'react'
import { useMotionValueEvent, useScroll } from 'motion/react'
import { hero } from '../data/copy'
import { Html } from '../lib/Html'
import { clamp, lerp } from '../lib/math'
import { scrollToElement } from '../lib/useLenis'
import { LockSequence } from './LockSequence'

/** The spin completes at 70% of the hero and holds. */
const SETTLE = 0.7

type Props = { reduced: boolean; onBroken: () => void }

export function ActLock({ reduced, onBroken }: Props) {
  const heroRef = useRef<HTMLElement>(null)
  const copyRef = useRef<HTMLDivElement>(null)
  const glowRef = useRef<HTMLDivElement>(null)
  const lockRef = useRef<HTMLDivElement>(null)
  const burstRef = useRef<HTMLDivElement>(null)
  const taphintRef = useRef<HTMLDivElement>(null)
  const hintRef = useRef<HTMLDivElement>(null)
  const flashRef = useRef<HTMLDivElement>(null)
  const [broken, setBroken] = useState(false)
  const brokenRef = useRef(false)
  const hintShown = useRef(false)

  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end end'] })

  // the rest of the page mounts once the lock is broken; the name act is whatever follows the hero
  function scrollNext() {
    const next = heroRef.current?.nextElementSibling as HTMLElement | null
    if (next) scrollToElement(next)
  }

  function breakLock() {
    if (brokenRef.current) return
    brokenRef.current = true
    setBroken(true)
    onBroken()
    taphintRef.current?.classList.remove('on')
    const lock = lockRef.current

    if (reduced) {
      lock?.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 400, fill: 'forwards' })
      setTimeout(scrollNext, 0)
      return
    }

    // squash (wind-up) → snap with a kick → evaporate: the lock breaks outward, it is not sucked in
    const out = 'cubic-bezier(.05,.7,.2,1)'
    lock?.animate([
      { offset: 0,    transform: 'scale(1)',                                  opacity: 1, filter: 'brightness(1)',  easing: 'cubic-bezier(.4,0,1,1)' },
      { offset: 0.14, transform: 'scale(.94)',                                opacity: 1, filter: 'brightness(.9)', easing: out },
      { offset: 0.31, transform: 'scale(1.16) rotate(-3deg)',                 opacity: 1, filter: 'brightness(3)',  easing: out },
      { offset: 1,    transform: 'scale(1.4) translateY(-22px) rotate(1deg)', opacity: 0, filter: 'brightness(4)' },
    ], { duration: 640, fill: 'forwards' })
    glowRef.current?.animate([{ opacity: 1 }, { opacity: 2.6, offset: 0.12 }, { opacity: 0 }], { duration: 1100, fill: 'forwards' })
    flashRef.current?.animate([{ opacity: 0 }, { opacity: 0.6, offset: 0.16 }, { opacity: 0 }], { duration: 700, delay: 80, easing: out })

    const burst = burstRef.current
    if (burst) {
      const ring = document.createElement('div'); ring.className = 'ring'; burst.appendChild(ring)
      ring.animate([{ transform: 'scale(.2)', opacity: 0.9 }, { transform: 'scale(12)', opacity: 0 }],
                   { duration: 1000, delay: 90, easing: 'cubic-bezier(.1,.8,.2,1)', fill: 'forwards' })

      for (let i = 0; i < 58; i++) {
        const dEl = document.createElement('div'); dEl.className = 'dot'
        const size = 2 + Math.random() * 7
        dEl.style.width = dEl.style.height = size + 'px'
        dEl.style.left = (-size / 2) + 'px'; dEl.style.top = (-size / 2) + 'px'
        if (Math.random() > 0.5) dEl.style.background = '#C9E6FF'
        if (Math.random() > 0.86) dEl.style.background = '#FFFFFF'
        burst.appendChild(dEl)
        const ang = Math.random() * Math.PI * 2, dist = 90 + Math.random() * 440
        const x = Math.cos(ang) * dist, y = Math.sin(ang) * dist * 0.82
        // a fast first leg, then gravity takes over
        dEl.animate([
          { transform: 'translate(0,0) scale(1)', opacity: 1, easing: out },
          { transform: `translate(${x * 0.78}px,${y * 0.78}px) scale(.7)`, opacity: 0.9, offset: 0.55, easing: 'cubic-bezier(.4,0,.6,1)' },
          { transform: `translate(${x}px,${y + 46}px) scale(0)`, opacity: 0 },
        ], { duration: 900 + Math.random() * 900, delay: 90 + Math.random() * 40, fill: 'both' })
      }
    }
    setTimeout(scrollNext, 620)
  }

  function draw(p: number) {
    const copy = copyRef.current, hint = hintRef.current, glow = glowRef.current, taphint = taphintRef.current
    if (!copy || !hint || !glow || !taphint) return
    if (!brokenRef.current) {
      const t = clamp(p / SETTLE, 0, 1), e = 1 - Math.pow(1 - t, 2.6)
      glow.style.opacity = String(lerp(0.3, 1, e))
      const show = t >= 0.995
      if (show !== hintShown.current) { hintShown.current = show; taphint.classList.toggle('on', show) }
    }
    const f = clamp((p - 0.05) / 0.28, 0, 1)
    copy.style.opacity = String(1 - f)
    copy.style.transform = 'translateY(' + (-f * 36) + 'px)'
    hint.style.opacity = String(1 - clamp(p / 0.12, 0, 1))
  }

  useMotionValueEvent(scrollYProgress, 'change', (p) => { if (!reduced) draw(p) })
  useEffect(() => {
    if (!reduced) draw(scrollYProgress.get())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced])

  return (
    <section className="hero" id="hero" ref={heroRef}>
      <div className="stick">
        <div className="hero-copy" ref={copyRef}>
          <h1>{hero.title}</h1>
          <Html html={hero.sub} />
        </div>

        <div className="stage">
          <div className="glowunder" ref={glowRef} />
          <div className="float">
            <LockSequence ref={lockRef} progress={scrollYProgress} settle={SETTLE} frozen={broken} reduced={reduced} />
          </div>
          <div className="burst" ref={burstRef} />
          <button className="tapzone" type="button" aria-label={hero.tapLabel} disabled={broken} onClick={breakLock} />
        </div>

        <div className="taphint" ref={taphintRef}><span>{hero.tapHint}</span></div>
        <div className="hint" ref={hintRef}>{hero.scrollHint}<i /></div>
        <div className="flash" ref={flashRef} />
      </div>
    </section>
  )
}
