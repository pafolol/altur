import { signals } from '../data/copy'
import { Html } from '../lib/Html'

/* ═══ card diagrams ═══ */
function rng(seed: number) {
  return () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
}
const SPEC = (() => {
  const r = rng(11)
  return Array.from({ length: 44 }, (_, x) => ({ x: x * 4.5, h: (4 + r() * 44) * (x < 29 ? 1 : 0.32), hi: x >= 29 }))
})()

const { overlap, sem } = signals.diagramLabels
const diagrams = {
  spec: (
    <g>
      {SPEC.map((b, i) => (
        <rect key={i} x={b.x} y={70 - b.h} width={2.6} height={b.h} rx={1.3} className={b.hi ? 'fill-glint' : 'fill-body'} opacity={b.hi ? 1 : 0.5} />
      ))}
      <line x1={128} y1={0} x2={128} y2={70} className="stroke-glint" strokeWidth={1} strokeDasharray="3 3" />
    </g>
  ),
  overlap: (
    <g>
      <rect x={0} y={14} width={110} height={13} rx={6.5} className="fill-body" opacity={0.45} />
      <rect x={94} y={42} width={106} height={13} rx={6.5} className="fill-body" opacity={0.45} />
      <rect x={94} y={14} width={16} height={41} rx={6} className="fill-glint" />
      <text x={116} y={10} className="font-mono fill-glint" fontSize={8.5} letterSpacing=".1em">{overlap.bargeIn}</text>
    </g>
  ),
  sem: (
    <g>
      <circle cx={32} cy={34} r={15} fill="none" className="stroke-body" strokeWidth={1.5} opacity={0.5} />
      <path d="M47 34 H94" className="stroke-body" strokeWidth={1.5} strokeDasharray="4 4" opacity={0.4} />
      <circle cx={110} cy={34} r={15} fill="none" className="stroke-glint" strokeWidth={1.5} />
      <path d="M104 28 L116 40 M116 28 L104 40" className="stroke-glint" strokeWidth={1.5} />
      <path d="M125 34 H180" className="stroke-glint" strokeWidth={1.5} />
      <path d="M172 28 L180 34 L172 40" fill="none" className="stroke-glint" strokeWidth={1.5} />
      <text x={16} y={64} className="font-mono fill-body" fontSize={8.5} opacity={0.6} letterSpacing=".1em">{sem.asked}</text>
      <text x={96} y={64} className="font-mono fill-glint" fontSize={8.5} letterSpacing=".1em">{sem.missing}</text>
    </g>
  ),
}

export function Signals() {
  return (
    <section className="band" id="how">
      <h2>{signals.heading}</h2>
      <Html className="lede" html={signals.lede} />
      <div className="signals">
        {signals.cards.map((c) => (
          <article className="sig" key={c.what}>
            <svg viewBox="0 0 200 70" aria-hidden="true">{diagrams[c.diagram]}</svg>
            <p className="what">{c.what}</p>
            <h3>{c.title}</h3>
            <Html html={c.text} />
            <ul>{c.points.map((p) => <Html as="li" key={p} html={p} />)}</ul>
          </article>
        ))}
      </div>
    </section>
  )
}
