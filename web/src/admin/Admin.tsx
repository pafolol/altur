import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { api, summarize, type Call, type DetectResponse, type Health, type Range, type Stats, type Verdict } from './api'
import { admin as t } from './copy'
import { ConfidenceHistogram, HourlyVerdicts, LatencySpark } from './charts'
import { fetchTelemetry, telemetryConfigured, type Telemetry } from './telemetry'

const fmtTime = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
const fmtDur = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
const pct = (n: number, d: number) => (d ? `${((100 * n) / d).toFixed(1)}%` : '—')

export function Admin() {
  const [range, setRange] = useState<Range>('24h')
  const [calls, setCalls] = useState<Call[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<Verdict | 'all'>('all')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string | null>(null)

  async function load() {
    setLoading(true); setError(null)
    fetchTelemetry().then(setTelemetry)   // never rejects, and a detector that is down must not hide it
    try {
      const [c, s, h] = await Promise.all([api.calls(range), api.stats(range), api.health()])
      setCalls(c); setStats(s); setHealth(h)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { load() }, [range]) // eslint-disable-line react-hooks/exhaustive-deps

  const summary = useMemo(() => summarize(calls, Date.now()), [calls])
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return calls.filter((c) => (filter === 'all' || c.verdict === filter) && (!q || c.id.includes(q) || c.queue.includes(q)))
  }, [calls, filter, query])
  const current = calls.find((c) => c.id === selected) ?? null

  return (
    <>
      <header className="top">
        <div className="brandline">
          <a className="brand" href="/">{t.brand}</a>
          <span className="sep" aria-hidden="true">·</span>
          <span className="section">{t.title}</span>
          <span className={api.mock ? 'pill sample' : 'pill live'}>{api.mock ? t.sample : t.live}</span>
        </div>
        <div className="controls">
          <label className="field">
            <span>{t.range.label}</span>
            <select value={range} onChange={(e) => setRange(e.target.value as Range)}>
              {t.range.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <button type="button" className="btn" onClick={load} disabled={loading}>{loading ? t.refreshing : t.refresh}</button>
          <a className="btn" href="/demo/">Latencia</a><a className="btn ghost" href="/">{t.backToSite}</a>
        </div>
      </header>

      {error && <p className="error" role="alert">{error}</p>}

      <main className={loading && calls.length ? 'stale' : undefined}>
        <section className="kpis" aria-label="Cifras clave">
          <Tile label={t.tiles.calls} value={String(summary.calls)} />
          <Tile label={t.tiles.synthetic} value={String(summary.synthetic)} note={`${pct(summary.synthetic, summary.calls)} ${t.tiles.ofCalls}`} tone="synthetic" />
          <Tile label={t.tiles.abstained} value={String(summary.abstained)} note={`${pct(summary.abstained, summary.calls)} ${t.tiles.ofCalls}`} tone="abstained" />
          <Tile label={t.tiles.latency} value={`${summary.latency_p50_ms} ms`} />
          <Tile label={t.tiles.latency95} value={`${summary.latency_p95_ms} ms`} />
          <Tile label={t.tiles.decided} value={`${summary.decided_p50_s.toFixed(1)} s`} />
        </section>

        <section className="charts">
          <HourlyVerdicts data={summary.hourly} />
          <ConfidenceHistogram bins={summary.confidence_bins} />
          {stats && <LatencySpark series={stats.latency_series} />}
        </section>

        <section className="stream">
          <div className="card">
            <header className="stream-head">
              <h3>{t.stream.title} <span className="count">{t.stream.showing} {rows.length} / {calls.length}</span></h3>
              <div className="chips" role="group" aria-label={t.stream.cols.verdict}>
                {(['all', 'human', 'synthetic', 'abstained'] as const).map((v) => (
                  <button key={v} type="button" className="chip" aria-pressed={filter === v} onClick={() => setFilter(v)}>
                    {v !== 'all' && <i className={`key key-${v}`} />}{v === 'all' ? t.stream.all : t.verdict[v]}
                  </button>
                ))}
              </div>
              <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t.stream.search} aria-label={t.stream.search} />
            </header>
            <div className="table-wrap">
              <table className="calls">
                <thead>
                  <tr>
                    <th>{t.stream.cols.time}</th><th>{t.stream.cols.id}</th><th>{t.stream.cols.queue}</th><th>{t.stream.cols.duration}</th>
                    <th>{t.stream.cols.verdict}</th><th>{t.stream.cols.confidence}</th><th>{t.stream.cols.signals}</th>
                    <th>{t.stream.cols.decided}</th><th>{t.stream.cols.latency}</th><th>{t.stream.cols.trap}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.id} tabIndex={0} aria-selected={c.id === selected} onClick={() => setSelected(c.id)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(c.id) } }}>
                      <td>{fmtTime(c.at)}</td>
                      <td className="mono">{c.id}</td>
                      <td>{c.queue}</td>
                      <td className="mono">{fmtDur(c.duration_s)}</td>
                      <td><VerdictBadge v={c.verdict} /></td>
                      <td className="mono">{c.confidence.toFixed(2)}</td>
                      <td><Signals e={c.evidence} /></td>
                      <td className="mono">{c.evidence.decided_at_s === null ? '—' : `${c.evidence.decided_at_s.toFixed(1)} s`}</td>
                      <td className="mono">{c.latency_ms} ms</td>
                      <td>{t.trap[c.trap]}</td>
                    </tr>
                  ))}
                  {!rows.length && <tr><td colSpan={10} className="empty">{t.stream.empty}</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
          <Detail call={current} onClose={() => setSelected(null)} />
        </section>

        <section className="tools">
          <Tester />
          <div className="side">
            {health && <HealthCard h={health} />}
            <TelemetryCard d={telemetry} />
          </div>
        </section>
      </main>
    </>
  )
}

function Tile({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: Verdict }) {
  return (
    <div className={`tile${tone ? ' tone-' + tone : ''}`}>
      <p className="label">{label}</p>
      <p className="value">{value}</p>
      {note && <p className="note">{note}</p>}
    </div>
  )
}

function VerdictBadge({ v }: { v: Verdict }) {
  return <span className={`badge badge-${v}`}><i className={`key key-${v}`} />{t.verdict[v]}</span>
}

function Signals({ e }: { e: Call['evidence'] }) {
  const vals = [e.acoustic, e.conversational, e.semantic]
  return (
    <span className="mini" aria-label={`${t.detail.signals.acoustic} ${e.acoustic.toFixed(2)}, ${t.detail.signals.conversational} ${e.conversational.toFixed(2)}, ${t.detail.signals.semantic} ${e.semantic.toFixed(2)}`}>
      {vals.map((v, i) => <i key={i} style={{ '--v': `${Math.round(v * 100)}%` } as CSSProperties} />)}
    </span>
  )
}

function Detail({ call, onClose }: { call: Call | null; onClose: () => void }) {
  if (!call) return <aside className="card detail idle"><h3>{t.detail.title}</h3><p className="muted">{t.detail.pick}</p></aside>
  const e = call.evidence
  const response = { is_synthetic: call.verdict === 'synthetic', confidence: +call.confidence.toFixed(2), evidence: { acoustic: +e.acoustic.toFixed(2), conversational: +e.conversational.toFixed(2), semantic: +e.semantic.toFixed(2), decided_at_s: e.decided_at_s === null ? null : +e.decided_at_s.toFixed(1) } }
  return (
    <aside className="card detail" aria-live="polite">
      <header className="detail-head">
        <div><h3>{t.detail.title} <span className="mono">{call.id}</span></h3><p className="muted">{fmtTime(call.at)} · {call.queue} · {fmtDur(call.duration_s)}</p></div>
        <button type="button" className="btn ghost small" onClick={onClose}>{t.detail.close}</button>
      </header>
      <VerdictBadge v={call.verdict} />
      <dl className="facts">
        <dt>{t.detail.channels}</dt><dd>{call.channels === 2 ? t.detail.stereo : t.detail.mono}</dd>
        <dt>{t.detail.confidence}</dt><dd><Meter v={call.confidence} /></dd>
        {(['acoustic', 'conversational', 'semantic'] as const).map((k) => <Row key={k} label={t.detail.signals[k]} v={e[k]} />)}
        <dt>{t.detail.decided}</dt><dd className="mono">{e.decided_at_s === null ? t.detail.notDecided : `${e.decided_at_s.toFixed(1)} s`}</dd>
        <dt>{t.detail.latency}</dt><dd className="mono">{call.latency_ms} ms</dd>
        <dt>{t.detail.trap}</dt><dd>{t.trap[call.trap]}</dd>
      </dl>
      <p className="label">{t.detail.response}</p>
      <pre>{JSON.stringify(response, null, 2)}</pre>
      <button type="button" className="btn" title={t.detail.reviewNote} onClick={() => alert(t.detail.reviewNote)}>{t.detail.review}</button>
    </aside>
  )
}

function Row({ label, v }: { label: string; v: number }) {
  return <><dt>{label}</dt><dd><Meter v={v} /></dd></>
}

function Meter({ v }: { v: number }) {
  return <span className="meter"><i style={{ '--v': `${Math.round(v * 100)}%` } as CSSProperties} /><b className="mono">{v.toFixed(2)}</b></span>
}

function Tester() {
  const [sample, setSample] = useState<'human' | 'synthetic'>('synthetic')
  const [file, setFile] = useState<{ name: string; b64: string } | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<{ res: DetectResponse; took: number } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function pick(f: File | undefined) {
    if (!f) { setFile(null); return }
    const buf = new Uint8Array(await f.arrayBuffer())
    let bin = ''; for (const b of buf) bin += String.fromCharCode(b)
    setFile({ name: f.name, b64: btoa(bin) })
  }
  async function run() {
    setRunning(true); setErr(null); setResult(null)
    const payload = file ? file.b64 : `sample:${sample}`
    const started = performance.now()
    try { setResult({ res: await api.detect(payload), took: Math.round(performance.now() - started) }) }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setRunning(false) }
  }
  const requestPreview = file ? `<base64 ${file.name}, ${Math.round((file.b64.length * 3) / 4).toLocaleString()} bytes>` : `<sample: ${t.tester.samples[sample]}>`
  const curl = `curl -X POST https://isisi.example/detect \\\n  -H 'Content-Type: application/json' \\\n  -d '{"audio":"<base64 stereo WAV, 8 kHz>"}'`

  return (
    <section className="card tester">
      <header className="chart-head"><div><h3>{t.tester.title}</h3><p>{t.tester.sub}</p></div></header>
      <div className="tester-grid">
        <div className="tester-form">
          <div className="seg" role="group" aria-label={t.tester.sample}>
            {(['human', 'synthetic'] as const).map((s) => (
              <button key={s} type="button" aria-pressed={sample === s && !file} onClick={() => { setSample(s); setFile(null); if (fileRef.current) fileRef.current.value = '' }}>{t.tester.samples[s]}</button>
            ))}
          </div>
          <label className="file">
            <span>{t.tester.file}</span>
            <input ref={fileRef} type="file" accept="audio/wav,audio/x-wav,audio/*" onChange={(e) => pick(e.target.files?.[0])} />
          </label>
          <button type="button" className="btn primary" onClick={run} disabled={running}>{running ? t.tester.running : t.tester.run}</button>
          {api.mock && <p className="muted small">{t.tester.mockNote}</p>}
        </div>
        <div className="tester-io">
          <p className="label">{t.tester.request}</p>
          <pre>{`POST /detect\n{\n  "audio": "${requestPreview}"\n}`}</pre>
          <p className="label">{t.tester.response}{result && <span className="mono took"> · {t.tester.took} {result.took} ms</span>}</p>
          <pre className={result ? 'ok' : undefined}>{err ? err : result ? JSON.stringify(result.res, null, 2) : '—'}</pre>
          <p className="label">{t.tester.curl}</p>
          <pre>{curl}</pre>
        </div>
      </div>
    </section>
  )
}

function HealthCard({ h }: { h: Health }) {
  const tone = h.endpoint === 'up' ? 'good' : h.endpoint === 'degraded' ? 'warning' : 'critical'
  return (
    <section className="card health">
      <header className="chart-head"><div><h3>{t.health.title}</h3></div><span className={`status status-${tone}`}><i />{t.health.status[h.endpoint]}</span></header>
      <dl className="facts">
        <dt>{t.health.model}</dt><dd className="mono">{h.model}</dd>
        <dt>{t.health.calibrated}</dt><dd className="mono">{h.calibrated_on}</dd>
        <dt>{t.health.streaming}</dt><dd>{h.streaming ? t.health.on : t.health.off}</dd>
        <dt>{t.health.uptime}</dt><dd className="mono">{h.uptime_24h.toFixed(2)}%</dd>
        <dt>{t.health.queue}</dt><dd className="mono">{h.queue_depth}</dd>
        <dt>{t.health.checked}</dt><dd className="mono">{fmtTime(h.checked_at)}</dd>
      </dl>
    </section>
  )
}

function TelemetryCard({ d }: { d: Telemetry | null }) {
  const state = !telemetryConfigured ? 'off' : d ? 'up' : 'down'
  const tone = state === 'up' ? 'good' : state === 'down' ? 'critical' : 'warning'
  return (
    <section className="card health">
      <header className="chart-head"><div><h3>{t.telemetry.title}</h3><p>{t.telemetry.sub}</p></div><span className={`status status-${tone}`}><i />{t.telemetry.status[state]}</span></header>
      {d ? (
        <dl className="facts">
          <dt>{t.telemetry.stored}</dt><dd className="mono">{d.stats.total_detections}</dd>
          <dt>{t.telemetry.split}</dt><dd className="mono">{d.stats.synthetic_detections} / {d.stats.human_detections}</dd>
          <dt>{t.telemetry.confidence}</dt><dd className="mono">{d.stats.average_confidence == null ? '—' : `${(100 * d.stats.average_confidence).toFixed(1)}%`}</dd>
          <dt>{t.telemetry.latency}</dt><dd className="mono">{d.stats.average_latency_ms == null ? '—' : `${Math.round(d.stats.average_latency_ms)} ms`}</dd>
          <dt>{t.telemetry.last}</dt><dd className="mono">{d.latest ? `${fmtTime(d.latest.timestamp)} · ${d.latest.call_id}` : '—'}</dd>
        </dl>
      ) : (
        <p className="muted small">{state === 'off' ? t.telemetry.offNote : t.telemetry.downNote}</p>
      )}
    </section>
  )
}
