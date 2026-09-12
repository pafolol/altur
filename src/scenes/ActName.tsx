import { useEffect, useRef } from 'react'
import { useMotionValueEvent, useScroll } from 'motion/react'
import { acronym } from '../data/acronym'
import { nameAct } from '../data/copy'
import { Html } from '../lib/Html'
import { clamp } from '../lib/math'
import { scrollToY } from '../lib/useLenis'

const STEPS = acronym.length
/** The hand-off scroll from the hero takes ~1.8 s; the name forms as it lands, so hold the first word until then. */
const ARRIVE_MS = 1500

export function ActName({ reduced }: { reduced: boolean }) {
  const ref = useRef<HTMLElement>(null)
  const letters = useRef<HTMLElement[]>([])
  const means = useRef<HTMLElement[]>([])
  const ticks = useRef<HTMLElement[]>([])
  const hintRef = useRef<HTMLDivElement>(null)
  const current = useRef(-1)
  const armed = useRef(false)

  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] })

  // one word per fifth of the scroll; the first is live on arrival
  function draw(p: number) {
    const idx = clamp(Math.floor(p * STEPS), 0, STEPS - 1)
    if (hintRef.current) hintRef.current.style.opacity = idx >= 1 ? '0' : '1'
    if (idx === current.current) return
    current.current = idx
    letters.current.forEach((el, k) => el.classList.toggle('on', k === idx))
    means.current.forEach((el, m) => el.classList.toggle('live', m === idx))
    ticks.current.forEach((el, q) => el.classList.toggle('live', q <= idx))
  }

  // a letter is a jump to the middle of its word's stretch of scroll
  function scrollToStep(i: number) {
    const el = ref.current
    if (!el) return
    scrollToY(el.offsetTop + ((i + 0.5) / STEPS) * (el.offsetHeight - innerHeight))
  }

  useMotionValueEvent(scrollYProgress, 'change', (p) => { if (!reduced && armed.current) draw(p) })
  useEffect(() => {
    if (reduced) return
    const t = setTimeout(() => { armed.current = true; draw(scrollYProgress.get()) }, ARRIVE_MS)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced])

  const on = reduced ? 'on' : undefined
  const live = reduced ? 'live' : undefined

  return (
    <section className="name" id="name" ref={ref}>
      <div className="stick">
        <div className="stagebox">
          <div className="wordmark" role="group" aria-label={acronym.map((m) => m.word[0]).join('')}>
            {acronym.map((m, i) => (
              <button
                key={m.word}
                type="button"
                className={on}
                style={{ '--i': i } as React.CSSProperties}
                aria-label={m.word}
                onClick={() => scrollToStep(i)}
                ref={(el) => { if (el) letters.current[i] = el }}
              >{m.word[0]}</button>
            ))}
          </div>
          <div className="meaning">
            {acronym.map((m, i) => (
              <div key={m.word} className={live ? 'mean live' : 'mean'} ref={(el) => { if (el) means.current[i] = el }}>
                <h2><em>{m.word}</em></h2>
                <Html html={m.text} />
              </div>
            ))}
          </div>
          <div className="tick" aria-hidden="true">
            {acronym.map((m, i) => <i key={m.word} className={live} ref={(el) => { if (el) ticks.current[i] = el }} />)}
          </div>
        </div>
        <div className="hint" ref={hintRef} style={{ opacity: 0 }}>{nameAct.hint}<i /></div>
      </div>
    </section>
  )
}
