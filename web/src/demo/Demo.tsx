/**
 * Detection latency — how EARLY the verdict was reachable, not just what it was.
 *
 * Every number here comes from the model. The acoustic layer scores 4 s chunks of caller speech and
 * aggregates their log-odds by mean, so the aggregate of the first k chunks is exactly what the deployed
 * model would have answered having heard only those k: the server walks that sequence and this draws it.
 * It is a prefix evaluation of a non-streaming model, which the footer says rather than implies.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, configured, type Analysis, type Benchmark, type CallRow, type Summary } from './api'

const s1 = (v?: number | null, unit = 's') => (v === null || v === undefined ? '—' : `${v.toFixed(1)}${unit}`)
const pct = (v?: number | null) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`)

export function Demo() {
  const [calls, setCalls] = useState<CallRow[]>([])
  const [id, setId] = useState('')
  const [model, setModel] = useState('specialist')
  const [data, setData] = useState<Analysis | null>(null)
  const [bench, setBench] = useState<Benchmark | null>(null)
  const [busy, setBusy] = useState(false)
  const [benchBusy, setBenchBusy] = useState(false)
  const [error, setError] = useState('')
  const [cursor, setCursor] = useState<number | null>(null)
  const heroRef = useRef<HTMLDivElement>(null)
  const firedRef = useRef(false)

  useEffect(() => {
    if (!configured) return
    api.calls().then((c) => { setCalls(c); setId(c[0]?.anon_id ?? '') }).catch((e) => setError(e.message))
  }, [])

  // The benchmark also tells the page which calls are worth opening, so it loads quietly in the
  // background. It is cached server-side after the first run.
  useEffect(() => {
    if (!configured) return
    setBench(null)
    api.benchmark(model).then(setBench).catch(() => {})
  }, [model])

  const analyse = useCallback(async (callId: string, m: string) => {
    if (!callId) return
    setBusy(true); setError(''); firedRef.current = false; setCursor(null)
    try { setData(await api.analyse(callId, m)) } catch (e) { setError((e as Error).message); setData(null) }
    setBusy(false)
  }, [])

  const flash = () => {
    const el = heroRef.current
    if (!el) return
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash')
  }

  useEffect(() => { if (data?.metrics.confident_detection_at_call_sec !== null) flash() }, [data])

  if (!configured) {
    return (
      <>
        <Header model={model} setModel={setModel} calls={[]} id="" setId={() => {}} onRun={() => {}} busy={false} />
        <main><div className="card empty">
          <b>VITE_API_URL is not set.</b><br />
          This page shows measured detection latency, so it has no mock to fall back on — a fabricated
          latency number is the one thing it must never display.<br /><br />
          <span className="mono small">VITE_API_URL=http://127.0.0.1:8000 npm run dev</span>
        </div></main>
      </>
    )
  }

  const m = data?.metrics
  const detected = !!m && m.confident_detection_at_call_sec !== null
  const synthetic = m?.early ? m.early.verdict === 'synthetic' : !!data?.final.is_synthetic

  return (
    <>
      <Header model={model} setModel={setModel} calls={calls} id={id} setId={setId}
              onRun={() => analyse(id, model)} busy={busy} />
      {error && <div className="error">{error}</div>}

      <main>
        {/* ── which call ─────────────────────────────────────────── */}
        <div className="card">
          <h3>Pick a held-out call</h3>
          <p className="sub">
            The 71 validation calls, speaker-disjoint from everything the model trained on. The truth label
            is revealed only after the analysis — the model is given the audio and nothing else.
          </p>
          {bench && <Picks bench={bench} onPick={(cid) => { setId(cid); analyse(cid, model) }} />}
        </div>

        {data && m && (
          <>
            {/* ── the verdict, and when ──────────────────────────── */}
            <div className={`hero ${synthetic ? 'synthetic' : 'human'}`} ref={heroRef}>
              <div className="face">
                <div className="label">{detected ? 'confident detection' : 'no confident detection'}</div>
                <div className="what">
                  {detected ? (synthetic ? 'Synthetic voice detected' : 'Human caller confirmed')
                            : 'Never left the uncertain band'}
                </div>
                <div className="pct">
                  {detected ? pct(m.confident_detection_probability) : pct(data.final.probability)}
                </div>
                <div className="label">
                  {detected ? 'synthetic probability at the moment of detection'
                            : 'final probability — the confidence band was never crossed'}
                </div>
              </div>
              <div className="grid">
                <Cell k="Detection time" v={detected ? s1(m.confident_detection_at_call_sec) : '—'}
                      sub={detected ? 'into the call' : undefined} />
                <Cell k="Caller speech required"
                      v={detected ? s1(m.confident_detection_after_caller_speech_sec) : '—'} />
                <Cell k="Call duration" v={s1(m.call_duration_sec)} />
                <Cell k="Detected before the call ended" lead
                      v={detected ? s1(m.seconds_before_call_end) : '—'}
                      sub={detected ? 'earlier' : undefined} />
              </div>
            </div>
            <p className="small muted" style={{ marginTop: -6 }}>
              {detected
                ? `Reachable after ${s1(m.confident_detection_after_caller_speech_sec)} of caller speech — `
                  + `${m.percentage_of_call_elapsed_at_detection}% of the way through a ${s1(m.call_duration_sec)} call. `
                  + `Inference itself took ${m.inference_ms} ms; the rest is waiting for the caller to talk.`
                : `The running verdict stayed between ${pct(m.thresholds.confident_human)} and `
                  + `${pct(m.thresholds.confident_synthetic)} for the whole call, so nothing here claims a detection.`}
            </p>

            {/* ── the call, end to end ───────────────────────────── */}
            <div className="card">
              <h3>Where the detection sits in the conversation</h3>
              <p className="sub">
                The caller's channel. Shaded bands are the speech the voice-activity detector found; the
                blocks are the 4 s chunks the classifier scored, shaded by <b>that chunk's own score</b>.
                These are model responses per segment — not located artefacts and not anomalies. The mark
                is the first moment the running verdict crossed the confidence band.
              </p>
              <Timeline data={data} cursor={cursor} />
              <div className="axis"><span>0 s</span><span>{data.duration_s.toFixed(1)} s</span></div>
              <audio
                controls preload="none" src={api.audio(data.anon_id)}
                onTimeUpdate={(e) => {
                  const t = e.currentTarget.currentTime
                  setCursor(t)
                  const at = m.confident_detection_at_call_sec
                  if (at !== null && !firedRef.current && t >= at) { firedRef.current = true; flash() }
                }}
                onSeeked={() => { firedRef.current = false }}
              />
              <div className="legend">
                <span><i style={{ background: '#24415C' }} />caller waveform</span>
                <span><i style={{ background: 'rgba(127,196,255,.22)' }} />speech region (VAD)</span>
                <span><i style={{ background: 'var(--color-synthetic)' }} />chunk scored synthetic</span>
                <span><i style={{ background: 'var(--color-human)' }} />chunk scored human</span>
                <span><i style={{ background: 'var(--color-mark)' }} />first confident detection</span>
              </div>
            </div>

            {/* ── how the confidence built ───────────────────────── */}
            <div className="card">
              <h3>What it would have said, after each chunk</h3>
              <p className="sub">
                The classifier averages the log-odds of every chunk, so the average of the first <i>k</i> is
                exactly what the deployed model would have answered having heard only those <i>k</i> — same
                aggregation, same calibration, nothing re-run. The bands are the decision threshold and the
                confidence band; where the line enters the band is the detection.
              </p>
              <div className="evo">
                <Chart data={data} />
                <div>
                  <table>
                    <thead><tr><th>caller speech</th><th>into call</th><th>this chunk</th><th>running</th></tr></thead>
                    <tbody>
                      {data.steps.map((s) => (
                        <tr key={s.chunk} className={s.call_time_s === m.confident_detection_at_call_sec ? 'hit' : ''}>
                          <td>{s1(s.caller_speech_s)}</td><td>{s1(s.call_time_s)}</td>
                          <td>{pct(s.chunk_probability)}</td><td><b>{pct(s.probability)}</b></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="small muted" style={{ marginTop: 12 }}>
                    {data.display} · {data.vad} VAD · {data.steps.length} chunks over {s1(m.caller_speech_sec)} of
                    caller speech. Confidence band {pct(m.thresholds.confident_human)}–
                    {pct(m.thresholds.confident_synthetic)}, decision threshold {pct(m.thresholds.decision)}.
                  </p>
                </div>
              </div>
            </div>

            {/* ── early against final ────────────────────────────── */}
            <div className="card">
              <h3>Was the early answer the right one?</h3>
              <p className="sub">
                The verdict at the moment of detection, beside the verdict after the whole conversation.
                When they disagree it is shown, not hidden — it is the number that decides whether an early
                answer can be acted on.
              </p>
              <div className="versus">
                <div className="vbox">
                  <div className="label">early — at detection</div>
                  <div className={`w ${m.early?.verdict ?? ''}`}>{m.early ? m.early.verdict : '—'}</div>
                  <div className="p">{m.early ? pct(m.early.probability) : 'the band was never crossed'}</div>
                  {m.early && <div className="when">
                    after {s1(m.early.after_caller_speech_sec)} of caller speech, {s1(m.early.at_call_sec)} into the call
                  </div>}
                </div>
                <div className="arrow">→</div>
                <div className="vbox">
                  <div className="label">final — whole call</div>
                  <div className={`w ${m.final?.verdict ?? ''}`}>{m.final ? m.final.verdict : '—'}</div>
                  <div className="p">{pct(m.final?.probability)}</div>
                  {m.final && <div className="when">
                    after the whole call — {s1(m.final.after_caller_speech_sec)} of caller speech
                  </div>}
                </div>
              </div>
              {m.early_agrees_with_final === false && m.early && m.final && (
                <div className="warn">
                  <b>The early verdict and the final one disagree.</b> The model committed to{' '}
                  <b>{m.early.verdict}</b> at {s1(m.early.at_call_sec)} and ended on <b>{m.final.verdict}</b>.
                  Shown rather than hidden: it is the honest measure of what an early answer is worth here.
                </div>
              )}
            </div>

            {/* ── the scorecard ──────────────────────────────────── */}
            <div className="card">
              <h3>Call analysis complete</h3>
              <div className="kpis" style={{ marginTop: 14 }}>
                <Tile k="Verdict" v={m.final?.verdict ?? '—'} />
                <Tile k="Final score" v={pct(m.final?.probability)} />
                <Tile k="First prediction" v={s1(m.first_prediction_after_caller_speech_sec)} note="of caller speech" />
                <Tile k="First confident detection"
                      v={detected ? s1(m.confident_detection_after_caller_speech_sec) : '—'} note="of caller speech" />
                <Tile k="Detected" v={detected ? s1(m.confident_detection_at_call_sec) : '—'} note="into the call" />
                <Tile k="Call duration" v={s1(m.call_duration_sec)} />
                <Tile k="Lead time" v={detected ? s1(m.seconds_before_call_end) : '—'} note="before the call ended" lead />
                <Tile k="Inference latency" v={`${m.inference_ms}`} note="milliseconds" />
              </div>
              <p className="small muted" style={{ marginTop: 14 }}>
                Truth for this call:{' '}
                <b style={{ color: data.label === 'synthetic' ? '#8FC0FF' : '#6FD3AE' }}>{data.label}</b> — the
                final verdict was <b>{(m.final?.verdict === 'synthetic') === (data.label === 'synthetic') ? 'correct' : 'WRONG'}</b>.
                Revealed only now.
              </p>
            </div>
          </>
        )}

        {/* ── across all 71 ────────────────────────────────────── */}
        <div className="card">
          <h3>Detection latency across all 71 calls</h3>
          <p className="sub">
            The same calculation over every held-out call — medians, so one long call cannot move the
            figure. Nothing here is asserted: it is computed from the same walk, on demand.
          </p>
          <div className="picks">
            <button className="btn primary" disabled={benchBusy}
                    onClick={async () => {
                      setBenchBusy(true)
                      try { setBench(await api.benchmark(model)) } catch (e) { setError((e as Error).message) }
                      setBenchBusy(false)
                    }}>
              {benchBusy ? 'scoring 71 calls…' : bench ? 'Recompute' : 'Run the benchmark'}
            </button>
            {bench && <span className="small muted">
              {bench.display} · band {pct(bench.thresholds.confident_human)}–{pct(bench.thresholds.confident_synthetic)} · {bench.generated}
            </span>}
          </div>
          {bench && <>
            <Bench title="Synthetic callers — the attack" s={bench.synthetic} />
            <Bench title="Human callers" s={bench.human} />
          </>}
        </div>
      </main>

      <footer>
        This is a <b>prefix evaluation</b>: the detector scores a finished recording, and the timeline is
        what it would have concluded having heard only the first <i>k</i> chunks. The aggregation and the
        calibration are the deployed ones, so the numbers are real — but the model is not streaming, and
        this page does not pretend otherwise. The confidence band is a policy choice set in{' '}
        <span className="mono">config.py</span>, not a value tuned on these 71 calls.
      </footer>
    </>
  )
}

/* ────────────────────────────────────────────────────────────── pieces */

function Header({ model, setModel, calls, id, setId, onRun, busy }: {
  model: string; setModel: (v: string) => void; calls: CallRow[]
  id: string; setId: (v: string) => void; onRun: () => void; busy: boolean
}) {
  return (
    <header className="top">
      <div className="brandline">
        <a className="brand" href="/">ISISI</a>
        <span className="sep">·</span>
        <span className="section">Detection latency</span>
      </div>
      <div className="controls">
        <label className="field">
          <span>call</span>
          <select value={id} onChange={(e) => setId(e.target.value)} disabled={!calls.length}>
            {calls.map((c, i) => (
              <option key={c.anon_id} value={c.anon_id}>
                {String(i + 1).padStart(2, '0')} · {c.anon_id.replace('call_', '')} · {c.duration_s.toFixed(0)}s
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>acoustic model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="specialist">V1 — specialist (in the fusion)</option>
            <option value="robust_v2">V2 — phone-hardened</option>
          </select>
        </label>
        <button className="btn primary" onClick={onRun} disabled={busy || !id}>
          {busy ? 'analysing…' : 'Analyse'}
        </button>
        <a className="btn" href="/admin/">Admin</a>
      </div>
    </header>
  )
}

function Cell({ k, v, sub, lead }: { k: string; v: string; sub?: string; lead?: boolean }) {
  return (
    <div className={`cell ${lead ? 'lead' : ''}`}>
      <div className="label">{k}</div>
      <div className="value">{v} {sub && <small>{sub}</small>}</div>
    </div>
  )
}

function Tile({ k, v, note, lead }: { k: string; v: string; note?: string; lead?: boolean }) {
  return (
    <div className={`tile ${lead ? 'lead' : ''}`}>
      <div className="label">{k}</div>
      <div className="value">{v}</div>
      {note && <div className="note">{note}</div>}
    </div>
  )
}

/** Quick picks decided by the measurement: on this split most calls are settled by the first chunk. */
function Picks({ bench, onPick }: { bench: Benchmark; onPick: (id: string) => void }) {
  const n = bench.notable
  if (!n) return null
  return (
    <div className="picks">
      <span className="small muted">
        {n.n_flat} of {n.n_total} calls are decided by the first chunk. The ones that are not:
      </span>
      {n.most_gradual.slice(0, 3).map((c) => (
        <button key={c.anon_id} className="btn small" onClick={() => onPick(c.anon_id)}>
          {c.anon_id.replace('call_', '').slice(0, 8)} · moves {(c.spread * 100).toFixed(0)} pts
        </button>
      ))}
      {n.early_disagreed_with_final.slice(0, 2).map((c) => (
        <button key={c.anon_id} className="btn small mark" onClick={() => onPick(c.anon_id)}>
          {c.anon_id.replace('call_', '').slice(0, 8)} · early verdict was wrong
        </button>
      ))}
    </div>
  )
}

function Bench({ title, s }: { title: string; s?: Summary }) {
  if (!s) return null
  return (
    <>
      <h3 style={{ margin: '20px 0 10px' }}>{title}</h3>
      {s.n_detected ? (
        <div className="kpis">
          <Tile k="Calls detected" v={`${s.n_detected}/${s.n}`} />
          <Tile k="Median caller speech" v={`${s.median_caller_speech_sec}`} note="seconds" />
          <Tile k="Median time into call" v={`${s.median_call_time_sec}`} note="seconds" />
          <Tile k="Median lead time" v={`${s.median_lead_time_sec}`} note="seconds before the end" lead />
          <Tile k="Detected before halfway" v={`${s.detected_before_halfway_pct}%`} />
          <Tile k="Early verdict held" v={`${s.early_agrees_with_final_pct}%`} note="agreed with the final one" />
        </div>
      ) : <p className="small muted">None of these calls crossed the confidence band.</p>}
    </>
  )
}

/* ────────────────────────────────────────────────────────────── canvas + svg */

function Timeline({ data, cursor }: { data: Analysis; cursor: number | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const x = c.getContext('2d')!
    const W = c.width, H = c.height, dur = data.duration_s || 1
    const T = (t: number) => (t / dur) * W
    x.clearRect(0, 0, W, H)
    const top = 16, waveH = 182, mid = top + waveH / 2
    x.fillStyle = 'rgba(127,196,255,.13)'
    data.speech_regions.forEach(([a, b]) => x.fillRect(T(a), top, Math.max(T(b) - T(a), 1), waveH))
    x.fillStyle = '#24415C'
    const bw = W / (data.envelope.length || 1)
    data.envelope.forEach((v, i) => {
      const h = Math.max(v * (waveH / 2), 1)
      x.fillRect(i * bw, mid - h, Math.max(bw - 0.5, 0.6), h * 2)
    })
    const bTop = top + waveH + 18, bH = 84
    data.steps.forEach((s, i) => {
      const [a, b] = data.chunk_spans[i] ?? [s.chunk_start_s, s.call_time_s]
      const p = s.chunk_probability, w = Math.max(T(b) - T(a), 3)
      const rgb = p >= 0.5 ? '57,135,229' : '25,158,112'
      const strength = Math.min(Math.abs(p - 0.5) * 2, 1)   // 0.5 = no opinion, 0/1 = emphatic
      x.fillStyle = `rgba(${rgb},${0.2 + strength * 0.7})`
      x.fillRect(T(a), bTop, w, bH)
      if (w > 48) {
        x.fillStyle = '#03080E'
        x.font = "600 23px 'DM Mono', monospace"
        x.textAlign = 'center'
        x.fillText(`${(p * 100).toFixed(0)}%`, T(a) + w / 2, bTop + bH / 2 + 8)
      }
    })
    x.strokeStyle = 'rgba(191,217,242,.13)'; x.lineWidth = 1
    x.strokeRect(0.5, bTop + 0.5, W - 1, bH)
    const at = data.metrics.confident_detection_at_call_sec
    if (at !== null) {
      const px = T(at)
      x.strokeStyle = '#C98500'; x.lineWidth = 3
      x.beginPath(); x.moveTo(px, 4); x.lineTo(px, bTop + bH); x.stroke()
      x.fillStyle = '#C98500'
      x.beginPath(); x.moveTo(px, 4); x.lineTo(px - 11, 23); x.lineTo(px + 11, 23); x.closePath(); x.fill()
      x.font = "600 22px 'DM Mono', monospace"
      x.textAlign = px > W * 0.62 ? 'right' : 'left'
      x.fillText(`DETECTED  ${at.toFixed(1)}s`, px + (px > W * 0.62 ? -16 : 16), 42)
    }
    if (cursor !== null) {
      x.strokeStyle = 'rgba(230,240,251,.85)'; x.lineWidth = 2
      x.beginPath(); x.moveTo(T(cursor), top); x.lineTo(T(cursor), bTop + bH); x.stroke()
    }
  }, [data, cursor])
  return <canvas className="tl" ref={ref} width={2400} height={310} />
}

function Chart({ data }: { data: Analysis }) {
  const { steps, metrics: m } = data
  if (!steps.length) return <svg className="chart" viewBox="0 0 720 300" />
  const W = 720, H = 300, L = 52, R = 16, Tp = 18, B = 44
  const maxX = Math.max(...steps.map((s) => s.caller_speech_s)) || 1
  const X = (v: number) => L + (v / maxX) * (W - L - R)
  const Y = (p: number) => Tp + (1 - p) * (H - Tp - B)
  const line = steps.map((s, i) => `${i ? 'L' : 'M'}${X(s.caller_speech_s)},${Y(s.probability)}`).join(' ')
  const bands: [number, number, string][] = [
    [m.thresholds.confident_synthetic, 1, 'rgba(57,135,229,.12)'],
    [0, m.thresholds.confident_human, 'rgba(25,158,112,.12)'],
  ]
  const lines: [number, string, string, string][] = [
    [m.thresholds.decision, '#6F8BA8', '4 4', 'decision 50%'],
    [m.thresholds.confident_synthetic, '#3987E5', '2 5', `confident synthetic ${pct(m.thresholds.confident_synthetic)}`],
    [m.thresholds.confident_human, '#199E70', '2 5', `confident human ${pct(m.thresholds.confident_human)}`],
  ]
  return (
    <svg className="chart" viewBox="0 0 720 300" preserveAspectRatio="none">
      {bands.map(([lo, hi, fill], i) => (
        <rect key={i} x={L} y={Y(hi)} width={W - L - R} height={Math.max(Y(lo) - Y(hi), 0)} fill={fill} />
      ))}
      {lines.map(([p, col, dash, label], i) => (
        <g key={i}>
          <line x1={L} y1={Y(p)} x2={W - R} y2={Y(p)} stroke={col} strokeDasharray={dash} />
          <text x={W - R} y={Y(p) - 5} fill={col} fontSize="9.5" textAnchor="end" fontFamily="DM Mono">{label}</text>
        </g>
      ))}
      {[0, .25, .5, .75, 1].map((p) => (
        <text key={p} x={L - 8} y={Y(p) + 3.5} fill="#6F8BA8" fontSize="10" textAnchor="end" fontFamily="DM Mono">
          {p * 100}%
        </text>
      ))}
      <path d={line} fill="none" stroke="#7FC4FF" strokeWidth="2.5" strokeLinejoin="round" />
      {steps.map((s) => {
        const hit = m.confident_detection_at_call_sec !== null && s.call_time_s === m.confident_detection_at_call_sec
        return (
          <g key={s.chunk}>
            {hit && <>
              <line x1={X(s.caller_speech_s)} y1={Y(s.probability)} x2={X(s.caller_speech_s)} y2={H - B}
                    stroke="#C98500" strokeDasharray="3 3" />
              <text x={X(s.caller_speech_s)} y={H - B + 27} fill="#C98500" fontSize="10.5" textAnchor="middle"
                    fontFamily="DM Mono">first confident · {s.caller_speech_s.toFixed(1)}s</text>
            </>}
            <circle cx={X(s.caller_speech_s)} cy={Y(s.probability)} r={hit ? 6.5 : 4}
                    fill={hit ? '#C98500' : '#7FC4FF'} stroke="#081522" strokeWidth={hit ? 2 : 1} />
            <text x={X(s.caller_speech_s)} y={H - B + 13} fill="#6F8BA8" fontSize="9.5" textAnchor="middle"
                  fontFamily="DM Mono">{s.caller_speech_s.toFixed(1)}</text>
          </g>
        )
      })}
      <text x={(L + W - R) / 2} y={H - 5} fill="#6F8BA8" fontSize="10" textAnchor="middle" fontFamily="DM Mono">
        seconds of caller speech heard
      </text>
    </svg>
  )
}
