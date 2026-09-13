/**
 * The detection-latency API, served by the fusion server (src/server.py, GET /demo/*).
 *
 * Same convention as the admin panel: VITE_API_URL points at the backend. Unset and the page says so
 * rather than inventing a timeline — there is no mock here, because a fabricated latency number is the
 * one thing this page must never show.
 */
export const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
export const configured = BASE !== ''

export interface Step {
  chunk: number
  call_time_s: number
  chunk_start_s: number
  caller_speech_s: number
  chunk_score: number
  /** what this chunk says on its own */
  chunk_probability: number
  /** what the model would have answered having heard only the chunks up to here */
  probability: number
  confident: boolean
}

export interface Verdict {
  verdict: 'synthetic' | 'human'
  probability: number
  at_call_sec?: number
  after_caller_speech_sec: number
}

export interface Metrics {
  call_duration_sec: number | null
  n_chunks: number
  caller_speech_sec: number
  first_prediction_at_call_sec: number | null
  first_prediction_after_caller_speech_sec: number | null
  first_prediction_probability: number | null
  confident_detection_at_call_sec: number | null
  confident_detection_after_caller_speech_sec: number | null
  confident_detection_probability: number | null
  confident_detection_chunk: number | null
  inference_ms: number | null
  seconds_before_call_end: number | null
  percentage_of_call_elapsed_at_detection: number | null
  early: Verdict | null
  final: Verdict | null
  /** null when there is nothing to compare; false is the case the page must not hide */
  early_agrees_with_final: boolean | null
  never_confident: boolean
  thresholds: { decision: number; confident_synthetic: number; confident_human: number }
}

export interface Analysis {
  anon_id: string
  label: 'human' | 'synthetic'
  model: string
  display: string
  vad: string
  duration_s: number
  /** peak envelope of the caller channel, for drawing only */
  envelope: number[]
  speech_regions: [number, number][]
  chunk_spans: [number, number][]
  final: { probability: number; is_synthetic: boolean; raw_score: number; speech_s: number }
  steps: Step[]
  metrics: Metrics
}

export interface CallRow { index: number; anon_id: string; label: string; duration_s: number }

export interface Notable {
  anon_id: string
  label: string
  n_chunks: number
  spread: number
  first: number | null
  final: number | null
  early_agrees_with_final: boolean | null
}

export interface Summary {
  n: number
  n_detected: number
  n_never_confident?: number
  median_caller_speech_sec?: number
  median_call_time_sec?: number
  median_lead_time_sec?: number
  median_percent_of_call_at_detection?: number
  detected_before_halfway_pct?: number
  early_agrees_with_final_pct?: number
}

export interface Benchmark {
  all: Summary
  synthetic?: Summary
  human?: Summary
  display: string
  generated: string
  thresholds: { decision: number; confident_synthetic: number; confident_human: number }
  notable: { most_gradual: Notable[]; early_disagreed_with_final: Notable[]; n_flat: number; n_total: number }
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`)
  const j = await r.json()
  if (j?.error) throw new Error(j.error)
  return j as T
}

export const api = {
  calls: () => get<CallRow[]>('/demo/calls'),
  analyse: (id: string, model: string) => get<Analysis>(`/demo/call/${id}?model=${model}`),
  benchmark: (model: string) => get<Benchmark>(`/demo/benchmark?model=${model}`),
  audio: (id: string) => `${BASE}/validation/${id}/audio`,
}
