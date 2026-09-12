import { useId, useState, type ReactNode } from 'react'
import { admin } from './copy'
import type { HourBucket } from './api'

/*
  Three SVG charts, no library. Marks follow the dataviz spec: bars ≤ 24px with a 4px
  rounded data-end and a 2px surface gap between stacked segments, 2px lines, ≥ 8px
  end markers with a 2px surface ring, solid hairline grid, text in text tokens only.
  Every chart has a hover tooltip (one readout, every series) and a table twin.
*/

const W = 720, H = 220
const PAD = { t: 14, r: 14, b: 26, l: 40 }
const PLOT_W = W - PAD.l - PAD.r, PLOT_H = H - PAD.t - PAD.b

function niceMax(v: number) {
  if (v <= 5) return 5
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  const m = v / p
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p
}

/** A column with rounded top corners and a square base. */
function column(x: number, y: number, w: number, h: number, r = 4) {
  if (h <= 0) return ''
  const rr = Math.min(r, h, w / 2)
  return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`
}

type TipRow = { key?: 'human' | 'synthetic' | 'abstained' | 'accent'; value: string; label: string }
type Tip = { x: number; y: number; rows: TipRow[] }

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return (
    <div className="tip" style={{ left: `${(tip.x / W) * 100}%`, top: `${(tip.y / H) * 100}%` }} role="status">
      {tip.rows.map((r, i) => (
        <div className="tip-row" key={i}>
          {r.key && <i className={`key key-${r.key}`} />}
          <b>{r.value}</b><span>{r.label}</span>
        </div>
      ))}
    </div>
  )
}

function ChartCard({ title, sub, legend, chart, table }: { title: string; sub: string; legend?: ReactNode; chart: ReactNode; table: ReactNode }) {
  const [view, setView] = useState<'chart' | 'table'>('chart')
  const id = useId()
  return (
    <section className="card chart" aria-labelledby={id}>
      <header className="chart-head">
        <div><h3 id={id}>{title}</h3><p>{sub}</p></div>
        <div className="seg" role="group">
          <button type="button" aria-pressed={view === 'chart'} onClick={() => setView('chart')}>{admin.charts.chart}</button>
          <button type="button" aria-pressed={view === 'table'} onClick={() => setView('table')}>{admin.charts.table}</button>
        </div>
      </header>
      {legend}
      {view === 'chart' ? <div className="plot">{chart}</div> : <div className="table-wrap twin">{table}</div>}
    </section>
  )
}

/* ─── verdicts by hour: stacked columns ───────────────────────────── */

const SERIES: Array<'human' | 'synthetic' | 'abstained'> = ['human', 'synthetic', 'abstained']

export function HourlyVerdicts({ data }: { data: HourBucket[] }) {
  const [tip, setTip] = useState<Tip | null>(null)
  const yMax = niceMax(Math.max(1, ...data.map((d) => d.human + d.synthetic + d.abstained)))
  const slot = PLOT_W / data.length
  const bw = Math.min(24, slot * 0.62)
  const y = (v: number) => PAD.t + PLOT_H * (1 - v / yMax)
  const show = (i: number) => {
    const d = data[i]
    setTip({ x: PAD.l + i * slot + slot / 2, y: y(d.human + d.synthetic + d.abstained), rows: [
      { key: 'human', value: String(d.human), label: admin.verdict.human },
      { key: 'synthetic', value: String(d.synthetic), label: admin.verdict.synthetic },
      { key: 'abstained', value: String(d.abstained), label: admin.verdict.abstained },
      { value: d.hour, label: admin.charts.hour },
    ] })
  }
  const legend = (
    <ul className="legend">
      {SERIES.map((s) => <li key={s}><i className={`key key-${s}`} />{admin.verdict[s]}</li>)}
    </ul>
  )
  const chart = (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={admin.charts.hourlySub}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(yMax * f)} y2={y(yMax * f)} className="grid" />
            <text x={PAD.l - 8} y={y(yMax * f) + 4} className="axis" textAnchor="end">{Math.round(yMax * f)}</text>
          </g>
        ))}
        {data.map((d, i) => {
          const x = PAD.l + i * slot + (slot - bw) / 2
          let cum = 0
          const segs = SERIES.map((s) => {
            const v = d[s]; const top = y(cum + v), bottom = y(cum); cum += v
            return { s, v, top, bottom }
          }).filter((seg) => seg.v > 0)
          const dim = tip && tip.x !== PAD.l + i * slot + slot / 2
          return (
            <g key={d.hour} className={dim ? 'dim' : undefined}>
              {segs.map((seg, k) => {
                const isTop = k === segs.length - 1
                const h = seg.bottom - seg.top - (k === 0 ? 0 : 2)   // 2px surface gap above the segment below
                return isTop
                  ? <path key={seg.s} d={column(x, seg.top, bw, h)} className={`fill-${seg.s}`} />
                  : <rect key={seg.s} x={x} y={seg.top} width={bw} height={Math.max(0, h)} className={`fill-${seg.s}`} />
              })}
              {i % 6 === 0 && <text x={x + bw / 2} y={H - 8} className="axis" textAnchor="middle">{d.hour}</text>}
              <rect x={PAD.l + i * slot} y={PAD.t} width={slot} height={PLOT_H} className="hit" tabIndex={0}
                aria-label={`${d.hour}: ${d.human} ${admin.verdict.human}, ${d.synthetic} ${admin.verdict.synthetic}, ${d.abstained} ${admin.verdict.abstained}`}
                onMouseEnter={() => show(i)} onFocus={() => show(i)} onMouseLeave={() => setTip(null)} onBlur={() => setTip(null)} />
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </>
  )
  const table = (
    <table>
      <thead><tr><th>{admin.charts.hour}</th><th>{admin.verdict.human}</th><th>{admin.verdict.synthetic}</th><th>{admin.verdict.abstained}</th></tr></thead>
      <tbody>{data.map((d) => <tr key={d.hour}><td>{d.hour}</td><td>{d.human}</td><td>{d.synthetic}</td><td>{d.abstained}</td></tr>)}</tbody>
    </table>
  )
  return <ChartCard title={admin.charts.hourly} sub={admin.charts.hourlySub} legend={legend} chart={chart} table={table} />
}

/* ─── confidence histogram: one series, one hue ──────────────────── */

export function ConfidenceHistogram({ bins }: { bins: number[] }) {
  const [tip, setTip] = useState<Tip | null>(null)
  const yMax = niceMax(Math.max(1, ...bins))
  const slot = PLOT_W / bins.length
  const bw = Math.min(24, slot * 0.62)
  const y = (v: number) => PAD.t + PLOT_H * (1 - v / yMax)
  const peak = bins.indexOf(Math.max(...bins))
  const label = (i: number) => `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`
  const chart = (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={admin.charts.confidenceSub}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(yMax * f)} y2={y(yMax * f)} className="grid" />
            <text x={PAD.l - 8} y={y(yMax * f) + 4} className="axis" textAnchor="end">{Math.round(yMax * f)}</text>
          </g>
        ))}
        {bins.map((v, i) => {
          const x = PAD.l + i * slot + (slot - bw) / 2
          return (
            <g key={i} className={tip && tip.x !== x + bw / 2 ? 'dim' : undefined}>
              <path d={column(x, y(v), bw, y(0) - y(v))} className="fill-synthetic" />
              {i === peak && v > 0 && <text x={x + bw / 2} y={y(v) - 6} className="value" textAnchor="middle">{v}</text>}
              {(i === 0 || i === 5 || i === 9) && <text x={x + bw / 2} y={H - 8} className="axis" textAnchor="middle">{i === 9 ? '1.0' : (i / 10).toFixed(1)}</text>}
              <rect x={PAD.l + i * slot} y={PAD.t} width={slot} height={PLOT_H} className="hit" tabIndex={0}
                aria-label={`${label(i)}: ${v} ${admin.charts.calls}`}
                onMouseEnter={() => setTip({ x: x + bw / 2, y: y(v), rows: [{ key: 'accent', value: String(v), label: admin.charts.calls }, { value: label(i), label: admin.charts.bin }] })}
                onFocus={() => setTip({ x: x + bw / 2, y: y(v), rows: [{ key: 'accent', value: String(v), label: admin.charts.calls }, { value: label(i), label: admin.charts.bin }] })}
                onMouseLeave={() => setTip(null)} onBlur={() => setTip(null)} />
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </>
  )
  const table = (
    <table>
      <thead><tr><th>{admin.charts.bin}</th><th>{admin.charts.calls}</th></tr></thead>
      <tbody>{bins.map((v, i) => <tr key={i}><td>{label(i)}</td><td>{v}</td></tr>)}</tbody>
    </table>
  )
  return <ChartCard title={admin.charts.confidence} sub={admin.charts.confidenceSub} chart={chart} table={table} />
}

/* ─── latency p50: a line with a wash and a crosshair ────────────── */

export function LatencySpark({ series }: { series: number[] }) {
  const [tip, setTip] = useState<(Tip & { i: number }) | null>(null)
  const lo = Math.min(...series), hi = Math.max(...series)
  const yMin = Math.floor((lo - 20) / 50) * 50, yMax = Math.ceil((hi + 20) / 50) * 50
  const x = (i: number) => PAD.l + (PLOT_W * i) / (series.length - 1)
  const y = (v: number) => PAD.t + PLOT_H * (1 - (v - yMin) / (yMax - yMin))
  const path = series.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')
  const area = `${path} L${x(series.length - 1)},${y(yMin)} L${x(0)},${y(yMin)} Z`
  const last = series.length - 1
  const bucketLabel = (i: number) => `${String(Math.floor(i / 2)).padStart(2, '0')}:${i % 2 ? '30' : '00'}`
  const at = (i: number) => setTip({ i, x: x(i), y: y(series[i]), rows: [{ key: 'accent', value: `${series[i]} ${admin.charts.ms}`, label: admin.charts.latency }, { value: bucketLabel(i), label: admin.charts.bucket }] })
  const chart = (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={admin.charts.latencySub} tabIndex={0}
        onFocus={() => at(last)} onBlur={() => setTip(null)}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; at(Math.round(Math.min(last, Math.max(0, ((px - PAD.l) / PLOT_W) * last)))) }}
        onMouseLeave={() => setTip(null)}>
        {[yMin, (yMin + yMax) / 2, yMax].map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={PAD.l - 8} y={y(v) + 4} className="axis" textAnchor="end">{v}</text>
          </g>
        ))}
        <path d={area} className="fill-synthetic wash" />
        <path d={path} className="stroke-synthetic line" />
        {tip && <line x1={tip.x} x2={tip.x} y1={PAD.t} y2={PAD.t + PLOT_H} className="crosshair" />}
        {tip && <circle cx={tip.x} cy={tip.y} r={4} className="fill-synthetic marker" />}
        <circle cx={x(last)} cy={y(series[last])} r={4} className="fill-synthetic marker" />
        <text x={x(last) - 10} y={y(series[last]) - 10} className="value" textAnchor="end">{series[last]} {admin.charts.ms}</text>
        {[0, 24, 47].map((i) => <text key={i} x={x(i)} y={H - 8} className="axis" textAnchor={i === 0 ? 'start' : i === 47 ? 'end' : 'middle'}>{bucketLabel(i)}</text>)}
      </svg>
      <Tooltip tip={tip} />
    </>
  )
  const table = (
    <table>
      <thead><tr><th>{admin.charts.bucket}</th><th>{admin.charts.latency} ({admin.charts.ms})</th></tr></thead>
      <tbody>{series.map((v, i) => <tr key={i}><td>{bucketLabel(i)}</td><td>{v}</td></tr>)}</tbody>
    </table>
  )
  return <ChartCard title={admin.charts.latency} sub={admin.charts.latencySub} chart={chart} table={table} />
}
