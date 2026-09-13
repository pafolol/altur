/**
 * The admin panel's data layer. Frontend only: `api` is a seeded mock behind the
 * `IsisiApi` interface. When a backend exists, implement the same interface with
 * fetch calls and swap it in here; nothing in the components needs to change.
 */
import { configured, liveApi } from './api.live'

export type Verdict = 'human' | 'synthetic' | 'abstained'
export type Trap = 'refused' | 'answered' | 'none'
export type Range = '24h'

export interface Evidence {
  acoustic: number
  conversational: number
  semantic: number
  /** Seconds into the call when confidence crossed the commit threshold; null when it abstained. */
  decided_at_s: number | null
}

export interface Call {
  id: string
  at: string
  duration_s: number
  channels: 1 | 2
  verdict: Verdict
  confidence: number
  evidence: Evidence
  latency_ms: number
  trap: Trap
  queue: string
}

export interface Stats {
  /** p50 latency per 30-minute bucket, oldest first, 48 points for 24 h. */
  latency_series: number[]
}

export interface Health {
  endpoint: 'up' | 'degraded' | 'down'
  model: string
  calibrated_on: string
  streaming: boolean
  uptime_24h: number
  queue_depth: number
  checked_at: string
}

export interface DetectResponse {
  is_synthetic: boolean
  confidence: number
  evidence: Evidence
}

export interface IsisiApi {
  readonly mock: boolean
  calls(range: Range): Promise<Call[]>
  stats(range: Range): Promise<Stats>
  health(): Promise<Health>
  detect(audioBase64: string): Promise<DetectResponse>
}

/* ─── mock ─────────────────────────────────────────────────────────── */

function rng(seed: number) {
  return () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
}
const QUEUES = ['collections', 'cards', 'onboarding', 'support']
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function makeCalls(now: number): Call[] {
  const r = rng(11)
  const g = () => (r() + r() + r()) / 3   // bell-ish 0..1
  const out: Omit<Call, 'id'>[] = []
  for (let i = 0; i < 138; i++) {
    // busier in office hours: pull ages toward 2–12 h ago
    const age = (r() < 0.7 ? 2 + g() * 10 : r() * 24) * 3600e3
    const roll = r()
    const verdict: Verdict = roll < 0.12 ? 'synthetic' : roll < 0.18 ? 'abstained' : 'human'
    const channels: 1 | 2 = r() < 0.9 ? 2 : 1
    let evidence: Evidence, confidence: number, trap: Trap
    if (verdict === 'synthetic') {
      evidence = { acoustic: 0.5 + g() * 0.45, conversational: 0.55 + g() * 0.4, semantic: 0.7 + g() * 0.29, decided_at_s: 14 + g() * 16 }
      confidence = 0.74 + g() * 0.22; trap = r() < 0.78 ? 'answered' : 'none'
    } else if (verdict === 'human') {
      evidence = { acoustic: g() * 0.32, conversational: g() * 0.25, semantic: g() * 0.15, decided_at_s: 14 + g() * 16 }
      confidence = 0.82 + g() * 0.16; trap = r() < 0.6 ? 'refused' : 'none'
    } else {
      evidence = { acoustic: 0.3 + g() * 0.4, conversational: 0.3 + g() * 0.4, semantic: 0.3 + g() * 0.3, decided_at_s: null }
      confidence = 0.5 + g() * 0.1; trap = 'none'
    }
    if (channels === 1) evidence.conversational = 0
    out.push({
      at: new Date(now - age).toISOString(),
      duration_s: Math.round(60 + g() * 210),   // the dataset's calls run 61–274 s
      channels, verdict, confidence, evidence,
      latency_ms: Math.round(880 + Math.exp(g() * 1.2) * 120),  // median ≈ 1.0 s with a tail: the primaries settle most calls in ~1.0 s (README.md)
      trap,
      queue: QUEUES[Math.floor(r() * QUEUES.length)],
    })
  }
  out.sort((a, b) => (a.at < b.at ? 1 : -1))
  return out.map((c, i) => ({ id: String(538 - i).padStart(4, '0'), ...c }))
}

// The same two real held-out calls the landing's demo shows (README.md, the five escalations). `decided_at_s`
// is computed by the server from the acoustic chunks and was not recorded for these, so it stays null.
const SAMPLE: Record<'human' | 'synthetic', DetectResponse> = {
  synthetic: { is_synthetic: true, confidence: 0.748, evidence: { acoustic: 1.0, conversational: 0.435, semantic: 0.952, decided_at_s: null } },
  human: { is_synthetic: false, confidence: 0.575, evidence: { acoustic: 0.0, conversational: 0.964, semantic: 0.046, decided_at_s: null } },
}

export const mockApi: IsisiApi = {
  mock: true,
  async calls() { await sleep(180); return makeCalls(Date.now()) },
  async stats() {
    await sleep(120)
    const r = rng(23)
    return { latency_series: Array.from({ length: 48 }, (_, i) => Math.round(1000 + 60 * Math.sin((i / 48) * Math.PI * 2 - 1.2) + (r() - 0.5) * 80)) }
  },
  async health() {
    await sleep(90)
    // Same shape and the same two honest fields the real /api/health returns: no streaming, no queue.
    return { endpoint: 'up', model: 'acoustic 0.50 + behaviour 0.50 + semantic 0.15', calibrated_on: '2026-09-12', streaming: false, uptime_24h: 100, queue_depth: 0, checked_at: new Date().toISOString() }
  },
  async detect(audio) {
    await sleep(950 + Math.random() * 150)   // ~1.0 s, what a call the primaries settle takes (README.md)
    return SAMPLE[audio.includes('synthetic') ? 'synthetic' : 'human']
  },
}

/* ─── the swap ─────────────────────────────────────────────────────── */
// The live client talks to the fusion server (src/server.py). It defaults to the deployed backend;
// VITE_API_URL overrides it, and an explicitly empty value selects the seeded mock.
// `api.mock` is what flips the header pill between SAMPLE and LIVE.
export const api: IsisiApi = configured ? liveApi : mockApi

/* ─── derived, computed on the client from the calls list ──────────── */

export interface HourBucket { hour: string; human: number; synthetic: number; abstained: number }

export interface Summary {
  calls: number
  synthetic: number
  abstained: number
  latency_p50_ms: number
  latency_p95_ms: number
  decided_p50_s: number
  hourly: HourBucket[]
  confidence_bins: number[]
}

const pct = (sorted: number[], p: number) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : 0

export function summarize(calls: Call[], now: number): Summary {
  const hourly: HourBucket[] = []
  const start = new Date(now); start.setMinutes(0, 0, 0)
  for (let i = 23; i >= 0; i--) {
    const h = new Date(start.getTime() - i * 3600e3)
    hourly.push({ hour: String(h.getHours()).padStart(2, '0') + ':00', human: 0, synthetic: 0, abstained: 0 })
  }
  const bins = Array(10).fill(0) as number[]
  for (const c of calls) {
    const idx = 23 - Math.floor((start.getTime() - new Date(c.at).getTime()) / 3600e3)
    if (idx >= 0 && idx < 24) hourly[idx][c.verdict]++
    bins[Math.min(9, Math.floor(c.confidence * 10))]++
  }
  const lat = calls.map((c) => c.latency_ms).sort((a, b) => a - b)
  const dec = calls.map((c) => c.evidence.decided_at_s).filter((v): v is number => v !== null).sort((a, b) => a - b)
  return {
    calls: calls.length,
    synthetic: calls.filter((c) => c.verdict === 'synthetic').length,
    abstained: calls.filter((c) => c.verdict === 'abstained').length,
    latency_p50_ms: pct(lat, 0.5),
    latency_p95_ms: pct(lat, 0.95),
    decided_p50_s: pct(dec, 0.5),
    hourly,
    confidence_bins: bins,
  }
}
