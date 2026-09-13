/**
 * The scene's API, served by the fusion server (src/server.py).
 *
 * VITE_API_URL points at it (the project .env sets it). No mock: a fabricated verdict or latency is the
 * one thing this page must never show, so without a backend it says so and stops.
 */
export const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
export const configured = BASE !== ''

export interface Step {
  chunk: number
  call_time_s: number
  chunk_start_s: number
  caller_speech_s: number
  chunk_score: number
  chunk_probability: number
  probability: number
  confident: boolean
}

/** The FUSION's running verdict at a chunk boundary: acoustic prefix + behaviour re-run on the prefix. */
export interface FusionStep {
  chunk: number
  call_time_s: number
  chunk_start_s: number
  caller_speech_s: number
  acoustic: number
  behaviour: number | null
  behaviour_state: 'voted' | 'abstained' | 'absent'
  probability: number
  confident: boolean
  n_layers_used: number
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
  early_agrees_with_final: boolean | null
  never_confident: boolean
  thresholds: { decision: number; confident_synthetic: number; confident_human: number }
}

export interface LayerVote {
  key: string
  display: string
  probability: number
  abstained: boolean
  scored: boolean
  consulted: boolean
  role: string | null
  share: number
  reason: string
  latency_ms?: number
}

export interface FinalVerdict {
  is_synthetic: boolean
  confidence: number
  probability: number
  decisive: boolean
  note: string
  verifiers_consulted: boolean
  layers: LayerVote[]
}

export interface Scene {
  anon_id?: string
  label?: 'human' | 'synthetic'
  acoustic_model: string | null
  acoustic_display: string | null
  vad: string
  weights: Record<string, number>
  roles: Record<string, string>
  duration_s: number
  sample_rate: number
  channels: number
  envelope: number[]
  agent_envelope: number[]
  speech_regions: [number, number][]
  agent_speech_regions: [number, number][]
  chunk_spans: [number, number][]
  steps: Step[]
  metrics: Metrics
  fusion_steps: FusionStep[]
  fusion_metrics: Metrics
  final: FinalVerdict
}

export interface CallRow { index: number; anon_id: string; label: string; duration_s: number }

export interface Notable {
  anon_id: string; label: string; n_chunks: number; spread: number
  first: number | null; final: number | null; early_agrees_with_final: boolean | null
}

export interface Summary {
  n: number; n_detected: number; n_never_confident?: number
  median_caller_speech_sec?: number; median_call_time_sec?: number; median_lead_time_sec?: number
  median_percent_of_call_at_detection?: number; detected_before_halfway_pct?: number
  early_agrees_with_final_pct?: number
}

export interface Benchmark {
  all: Summary; synthetic?: Summary; human?: Summary
  display: string; generated: string
  thresholds: { decision: number; confident_synthetic: number; confident_human: number }
  notable: { most_gradual: Notable[]; early_disagreed_with_final: Notable[]; n_flat: number; n_total: number }
}

/** What POST /detect answers - the challenge contract plus every layer's own score under details. */
export interface DetectResponse {
  is_synthetic: boolean
  confidence: number
  details?: {
    synthetic_probability: number
    layers: LayerVote[]
    weights: Record<string, number>
    latency_ms: number
    n_layers_used: number
    warning?: string
  }
  error?: string
}

export interface TwilioConfig {
  configured: boolean; missing: string[]; acoustic_model: string; from: string; model_available: boolean
  /** Masked destination, and whether TWILIO_TO_NUMBER is set at all - the one-button call needs it. */
  default_to: string; has_default_to: boolean
}

export interface TwilioJob {
  call_sid: string; to: string; state: 'queued' | 'ringing' | 'recording' | 'scoring' | 'done' | 'failed'
  detail: string; elapsed_s: number; call_status: string; duration_s: number
  verdict: null | { is_synthetic: boolean; confidence: number; synthetic_probability: number; acoustic_model: string; note: string; layers: LayerVote[] }
  error: string
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init)
  const j = await r.json().catch(() => ({}))
  if (!r.ok || j?.error) throw new Error(j?.error || `${path} → HTTP ${r.status}`)
  return j as T
}
const json = (body: unknown): RequestInit =>
  ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  calls: () => get<CallRow[]>('/demo/calls'),
  call: (id: string) => get<Scene>(`/demo/call/${id}`),
  analyseFile: (file: Blob) => {
    const fd = new FormData(); fd.append('file', file, 'call.wav')
    return get<Scene>('/demo/analyse', { method: 'POST', body: fd })
  },
  analyseBase64: (audio: string) => get<Scene>('/demo/analyse', json({ audio })),
  detect: (audio: string, queue = 'live') => get<DetectResponse>('/detect', json({ audio, queue })),
  benchmark: () => get<Benchmark>('/demo/benchmark?model=specialist'),
  audio: (id: string) => `${BASE}/validation/${id}/audio`,
  twilio: {
    config: () => get<TwilioConfig>('/twilio/config'),
    // No `to` means the server dials TWILIO_TO_NUMBER from the project .env - which is what the demo's
    // "Try being the caller" button does, so no phone number is ever typed into, or returned to, the page.
    call: (seconds = 30, to?: string) => get<TwilioJob>('/twilio/call', json(to ? { to, seconds } : { seconds })),
    status: (sid: string) => get<TwilioJob>(`/twilio/status/${sid}`),
    analysis: (sid: string) => get<Scene>(`/twilio/analysis/${sid}`),
    recording: (sid: string) => `${BASE}/twilio/recording/${sid}`,
  },
}
