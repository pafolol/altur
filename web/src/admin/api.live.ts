/**
 * The real `IsisiApi`, talking to the fusion server in `src/server.py`.
 *
 * `api.ts` said: "When a backend exists, implement the same interface with fetch calls and swap it in
 * here; nothing in the components needs to change." This is that implementation — the mock is untouched
 * and still the fallback, and `api.ts` picks between them on one line.
 *
 * Point it at the server with `VITE_API_URL`:
 *
 *     VITE_API_URL=http://127.0.0.1:8000 npm run dev
 *
 * With the variable unset it uses the deployed backend. Set it to an empty string to use the seeded mock.
 * `mock: false` here is what flips the header pill from SAMPLE to LIVE.
 *
 * Two fields the server deliberately does not pretend to have, and the panel shows them as they are:
 *   streaming    false — the detector scores a complete call; it does not decide as audio arrives.
 *   queue_depth  0 — requests are served synchronously, so there is no queue.
 */
import type { Call, DetectResponse, Health, IsisiApi, Range, Stats } from './api'

const DEFAULT_API_URL = 'https://hack.pafodev.com'
export const BASE = (import.meta.env.VITE_API_URL ?? DEFAULT_API_URL).replace(/\/$/, '')
export const configured = BASE !== ''

/** The server always answers /detect, even on failure, so a non-2xx here is a real transport problem. */
async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`)
  return (await r.json()) as T
}

export const liveApi: IsisiApi = {
  mock: false,

  calls(range: Range) {
    return get<Call[]>(`/api/calls?range=${encodeURIComponent(range)}`)
  },

  stats(range: Range) {
    return get<Stats>(`/api/stats?range=${encodeURIComponent(range)}`)
  },

  health() {
    return get<Health>('/api/health')
  },

  async detect(audioBase64: string): Promise<DetectResponse> {
    // The challenge contract is {is_synthetic, confidence}; everything the panel draws underneath it
    // lives in `details.layers`, one entry per detection layer. `conversational` is the behaviour layer.
    const raw = await get<{
      is_synthetic: boolean
      confidence: number
      details?: {
        layers?: { key: string; probability: number; abstained?: boolean; scored?: boolean }[]
        decided_at_s?: number | null
      }
    }>('/detect', { method: 'POST', body: JSON.stringify({ audio: audioBase64, queue: 'admin' }) })

    const layers = raw.details?.layers ?? []
    // A layer that abstained or was never asked has no opinion — 0, not a made-up score.
    const score = (key: string) => {
      const l = layers.find((x) => x.key === key)
      return !l || l.abstained || l.scored === false ? 0 : l.probability
    }
    return {
      is_synthetic: raw.is_synthetic,
      confidence: raw.confidence,
      evidence: {
        acoustic: score('acoustic'),
        conversational: score('behaviour'),
        semantic: score('semantic'),
        decided_at_s: raw.details?.decided_at_s ?? null,
      },
    }
  },
}
