/**
 * The Tiger telemetry service (tiger-telemetry/): a copy of every verdict, kept in Tiger Data. The panel
 * reads it straight from the browser — the service allows the dev origins listed in
 * TELEMETRY_ALLOWED_ORIGINS — and only when VITE_TELEMETRY_URL is set; without it nothing here runs.
 */
export interface TelemetryStats {
  total_detections: number
  synthetic_detections: number
  human_detections: number
  average_confidence: number | null
  average_latency_ms: number | null
}
export interface TelemetryEvent {
  timestamp: string
  call_id: string
  is_synthetic: boolean
  confidence: number
  final_score: number | null
  latency_ms: number | null
}
export interface Telemetry { stats: TelemetryStats; latest: TelemetryEvent | null }

const BASE = (import.meta.env.VITE_TELEMETRY_URL ?? '').replace(/\/$/, '')
export const telemetryConfigured = BASE !== ''

/** null when not configured or not answering: the card says which, and nothing else in the panel cares. */
export async function fetchTelemetry(): Promise<Telemetry | null> {
  if (!telemetryConfigured) return null
  try {
    const [s, d] = await Promise.all([fetch(`${BASE}/stats`), fetch(`${BASE}/detections`)])
    if (!s.ok || !d.ok) return null
    const events = (await d.json()) as TelemetryEvent[]
    return { stats: (await s.json()) as TelemetryStats, latest: events[0] ?? null }
  } catch {
    return null
  }
}
