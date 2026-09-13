import { useEffect, useRef, useState } from 'react'
import { api, configured, type DetectResponse, type Scene, type TwilioConfig, type TwilioJob } from './api'

type Mode = 'live' | 'prerecorded'
type CaptureState = 'idle' | 'waiting' | 'recording' | 'analysing'
/** What the phone call is doing right now, so the stage can say it. `live` = the person is on the line. */
type Phone = { label: string; live: boolean } | null

const pct = (value?: number | null) => value == null ? '--' : `${(value * 100).toFixed(1)}%`
const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`
const within = (regions: [number, number][], time: number) => regions.some(([start, end]) => time >= start && time <= end)

function resample(input: Float32Array, sourceRate: number, targetRate = 8000) {
  if (sourceRate === targetRate) return input
  const output = new Float32Array(Math.round(input.length * targetRate / sourceRate))
  const ratio = sourceRate / targetRate
  for (let i = 0; i < output.length; i++) {
    const at = i * ratio
    const before = Math.floor(at)
    const after = Math.min(before + 1, input.length - 1)
    output[i] = input[before] + (input[after] - input[before]) * (at - before)
  }
  return output
}

function stereoCallerWav(samples: Float32Array, sourceRate: number) {
  const caller = resample(samples, sourceRate)
  const buffer = new ArrayBuffer(44 + caller.length * 4)
  const view = new DataView(buffer)
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i))
  }
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE')
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 2, true); view.setUint32(24, 8000, true); view.setUint32(28, 32000, true)
  view.setUint16(32, 4, true); view.setUint16(34, 16, true); text(36, 'data')
  view.setUint32(40, caller.length * 4, true)
  let offset = 44
  caller.forEach((sample) => {
    const value = Math.max(-1, Math.min(1, sample))
    view.setInt16(offset, value < 0 ? value * 32768 : value * 32767, true)
    view.setInt16(offset + 2, 0, true)
    offset += 4
  })
  return new Blob([buffer], { type: 'audio/wav' })
}

async function blobBase64(blob: Blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(binary)
}

export function Demo() {
  const [mode, setMode] = useState<Mode>('live')
  const [scene, setScene] = useState<Scene | null>(null)
  const [liveResult, setLiveResult] = useState<DetectResponse | null>(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [capture, setCapture] = useState<CaptureState>('idle')
  const [captureProgress, setCaptureProgress] = useState(0)
  const [meter, setMeter] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [base64, setBase64] = useState('')
  const [fileName, setFileName] = useState('')
  const [showBase64, setShowBase64] = useState(false)
  const [phone, setPhone] = useState<Phone>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const captureStop = useRef<null | (() => void)>(null)

  useEffect(() => () => {
    captureStop.current?.()
    if (audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  const installAudio = (url: string) => {
    setAudioUrl((old) => { if (old.startsWith('blob:')) URL.revokeObjectURL(old); return url })
    setTime(0); setPlaying(false)
  }

  const analyseBlob = async (file: Blob) => {
    setBusy(true); setError(''); setLiveResult(null); setScene(null)
    installAudio(URL.createObjectURL(file))
    try {
      // Paint the deployed verdict first; the richer prefix scene is deliberately lower priority.
      setLiveResult(await api.detect(await blobBase64(file), 'demo'))
      setScene(await api.analyseFile(file))
    } catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  const analyseText = async () => {
    const clean = base64.trim().replace(/^data:audio\/[\w.+-]+;base64,/, '')
    if (!clean) return
    setBusy(true); setError(''); setLiveResult(null); setScene(null)
    try {
      const raw = atob(clean); const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0))
      installAudio(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })))
      setLiveResult(await api.detect(clean, 'demo'))
      setScene(await api.analyseBase64(clean))
    } catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  const startLive = async () => {
    setError(''); setLiveResult(null); setScene(null); setCaptureProgress(0)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false })
      const context = new AudioContext()
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(2048, 1, 1)
      const mute = context.createGain(); mute.gain.value = 0
      const chunks: Float32Array[] = []
      let count = 0; let started = false; let closed = false
      const needed = Math.ceil(context.sampleRate * 5)
      const cleanup = () => {
        if (closed) return
        closed = true; processor.disconnect(); source.disconnect(); mute.disconnect()
        stream.getTracks().forEach((track) => track.stop()); void context.close()
        captureStop.current = null; setMeter(0)
      }
      captureStop.current = () => { cleanup(); setCapture('idle'); setCaptureProgress(0) }
      processor.onaudioprocess = async (event) => {
        if (closed) return
        const input = event.inputBuffer.getChannelData(0)
        const rms = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0) / input.length)
        setMeter(Math.min(rms * 9, 1))
        if (!started && rms > 0.025) { started = true; setCapture('recording') }
        if (!started) return
        const copy = new Float32Array(input); chunks.push(copy); count += copy.length
        setCaptureProgress(Math.min(count / needed, 1))
        if (count < needed) return
        cleanup(); setCapture('analysing')
        const joined = new Float32Array(count); let at = 0
        chunks.forEach((chunk) => { joined.set(chunk, at); at += chunk.length })
        const wav = stereoCallerWav(joined.subarray(0, needed), context.sampleRate)
        installAudio(URL.createObjectURL(wav))
        try { setLiveResult(await api.detect(await blobBase64(wav), 'live')) }
        catch (e) { setError((e as Error).message) }
        finally { setCapture('idle') }
      }
      source.connect(processor); processor.connect(mute); mute.connect(context.destination)
      setCapture('waiting')
    } catch (e) { setError((e as Error).message); setCapture('idle') }
  }

  const final = scene?.final ?? (liveResult ? {
    is_synthetic: liveResult.is_synthetic,
    confidence: liveResult.confidence,
    probability: liveResult.details?.synthetic_probability ?? (liveResult.is_synthetic ? liveResult.confidence : 1 - liveResult.confidence),
    decisive: true, note: liveResult.details?.warning ?? '', verifiers_consulted: false,
    layers: liveResult.details?.layers ?? [],
  } : null)
  const latencyMs = liveResult?.details?.layers.find((layer) => layer.key === 'acoustic')?.latency_ms
    ?? scene?.final.layers.find((layer) => layer.key === 'acoustic')?.latency_ms
    ?? null
  const liveSpeaking = capture === 'recording'

  return (
    <div className="demo-shell">
      <header className="top">
        <a className="brand" href="/" aria-label="ISISI home"><b>ISISI</b></a>
        <span className="product-name">VOICE DETECTION</span>
        <div className="controls">
          <a className="nav-link" href="/">Back</a>
        </div>
      </header>

      {!configured && <div className="error">Detection service unavailable.</div>}
      {error && <div className="error">{error}</div>}

      <main>
        <section className="mode-panel">
          <div className="seg" aria-label="Demo mode">
            <button aria-pressed={mode === 'live'} onClick={() => setMode('live')}>Live</button>
            <button aria-pressed={mode === 'prerecorded'} onClick={() => setMode('prerecorded')}>Upload</button>
          </div>
          {mode === 'live' ? (
            <div className="live-controls">
              <div className="mode-state">{capture === 'waiting' ? 'Listening' : capture === 'recording' ? 'Listening' : capture === 'analysing' ? 'Analyzing' : 'Microphone'}</div>
              <div className="capture-action">
                {capture === 'idle' || capture === 'analysing' ? <button className="btn primary record" onClick={startLive} disabled={!configured || capture === 'analysing'}><span />{capture === 'analysing' ? 'Analyzing' : 'Start'}</button>
                  : <button className="btn record stop" onClick={() => captureStop.current?.()}><span />Stop</button>}
                <div className="capture-meter"><i style={{ width: `${Math.max(meter, captureProgress) * 100}%` }} /></div>
                <span className="mono small">{capture === 'recording' ? `${(captureProgress * 5).toFixed(1)} / 5.0 s` : capture === 'waiting' ? 'Listening' : ''}</span>
              </div>
            </div>
          ) : (
            <div className="upload-controls">
              <div className="file-name">{fileName || 'WAV audio'}</div>
              <div className="row">
                <label className="btn primary upload">{busy ? 'Analyzing' : 'Choose WAV'}<input type="file" accept="audio/wav,.wav" disabled={busy} onChange={(e) => { const file = e.target.files?.[0]; if (file) { setFileName(file.name); void analyseBlob(file) } }} /></label>
                <button className="btn quiet" onClick={() => setShowBase64((value) => !value)}>Base64</button>
              </div>
              {showBase64 && <div className="base64-box"><textarea value={base64} onChange={(e) => setBase64(e.target.value)} placeholder="Paste WAV base64" /><button className="btn primary" disabled={busy || !base64.trim()} onClick={analyseText}>Analyze</button></div>}
            </div>
          )}
          <CallerCta
            onPhase={setPhone}
            onError={setError}
            onStart={() => { captureStop.current?.(); setError(''); setLiveResult(null); setScene(null) }}
            onScene={(next, sid) => { setLiveResult(null); setScene(next); installAudio(api.twilio.recording(sid)) }} />
        </section>

        {final && <InstantResult final={final} latencyMs={latencyMs} />}

        <CallStage scene={scene} final={final} time={time} playing={playing} liveSpeaking={liveSpeaking} meter={meter} capture={capture} phone={phone} />

        {audioUrl && <section className="transport-card">
          <button className="play" aria-label={playing ? 'Pause' : 'Play'} onClick={() => { const audio = audioRef.current; if (!audio) return; playing ? audio.pause() : void audio.play() }}>{playing ? 'Ⅱ' : '▶'}</button>
          <div className="track"><Waveform values={scene?.envelope ?? []} progress={scene ? time / scene.duration_s : 0} /><input aria-label="Playback position" type="range" min="0" max={scene?.duration_s ?? audioRef.current?.duration ?? 5} step="0.01" value={time} onChange={(e) => { if (audioRef.current) audioRef.current.currentTime = Number(e.target.value); setTime(Number(e.target.value)) }} /></div>
          <span className="mono small">{clock(time)} / {clock(scene?.duration_s ?? audioRef.current?.duration ?? 5)}</span>
          <audio ref={audioRef} src={audioUrl} onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
        </section>}
      </main>
      <footer><span>ISISI</span><span>Voice authenticity</span></footer>
    </div>
  )
}

function CallStage({ scene, final, time, playing, liveSpeaking, meter, capture, phone }: { scene: Scene | null; final: Scene['final'] | null; time: number; playing: boolean; liveSpeaking: boolean; meter: number; capture: CaptureState; phone: Phone }) {
  const callerSpeaking = liveSpeaking || !!phone?.live || !!scene && within(scene.speech_regions, time)
  const agentSpeaking = !!scene && within(scene.agent_speech_regions, time)
  // On a phone call there is no meter to read - the audio is on Twilio's side, not in this browser - so the
  // caller ring pulses at level 0 rather than inventing a level, and its label says "on the call", not "speaking".
  const callerLevel = liveSpeaking ? meter : phone?.live ? 0 : envelopeAt(scene?.envelope, time, scene?.duration_s)
  const agentLevel = envelopeAt(scene?.agent_envelope, time, scene?.duration_s)
  const steps = scene?.fusion_steps.length ? scene.fusion_steps : scene?.steps ?? []
  const running = [...steps].reverse().find((step) => step.call_time_s <= time)
  const visibleFinal = final && (!scene || (!playing && time === 0) || time >= scene.duration_s - .08)
  const probability = visibleFinal ? final.probability : running?.probability
  const synthetic = probability != null ? probability >= .5 : final?.is_synthetic
  const confidence = probability == null ? null : synthetic ? probability : 1 - probability
  const listening = !!phone || capture === 'waiting' || capture === 'recording' || (!!scene && time < scene.duration_s)
  const tone = probability == null ? '' : synthetic ? 'synthetic' : 'human'
  const state = phone?.label ?? (capture === 'analysing' ? 'Analyzing' : listening ? 'Listening' : 'Ready')

  return <section className={`stage ${tone}`}>
    <div className="stage-in">
      <Person kind="caller" speaking={callerSpeaking} level={callerLevel} verdict={tone || undefined}
              stateLabel={phone?.live ? 'On the call' : undefined} />
      <div className="system">
        <div className={`wire ${listening ? 'busy' : ''}`} />
        <div className={`verdict ${tone}`} aria-live="polite">
          {probability == null ? <><div className={`status-mark ${listening ? 'active' : ''}`} /><div className="waiting">{state}</div></>
            : <><div className="word">{synthetic ? 'Synthetic' : 'Human'}</div><div className="confidence"><b>{pct(confidence)}</b> confidence</div></>}
          {capture === 'recording' && <LiveWave level={meter} />}
        </div>
      </div>
      <Person kind="agent" speaking={agentSpeaking} level={agentLevel} />
    </div>
    {visibleFinal && final && <AnalysisDetails final={final} scene={scene} />}
  </section>
}

const CALLER_IMAGE_URL = 'https://static1.personalitydatabase.net/2/pdb-images-prod/fab844b5/profile_images/b55f58820f3b4fe18f95e0a3adb42fc1.png'

function Person({ kind, speaking, level, verdict, stateLabel }: { kind: 'agent' | 'caller'; speaking: boolean; level: number; verdict?: string; stateLabel?: string }) {
  const transform = speaking ? `scale(${1 + Math.min(level, 1) * .055})` : undefined
  const shadow = speaking ? `0 0 0 ${6 + level * 8}px color-mix(in srgb, var(--who-color) ${22 + level * 28}%, transparent), 0 0 ${35 + level * 35}px var(--who-glow)` : undefined
  return <div className={`who ${kind} ${speaking ? 'speaking' : ''} ${verdict ?? ''}`}>
    <div className="avatar" style={{ transform, boxShadow: shadow }}>
      <div className="ring" />
      {kind === 'agent' ? <AgentIcon /> : <img src={CALLER_IMAGE_URL} alt="Caller" />}
    </div>
    <div className="name">{kind === 'agent' ? 'Altur Agent' : 'Caller'}</div>
    <div className="state">{stateLabel ?? (speaking ? 'Speaking' : '')}</div>
  </div>
}

function AgentIcon() { return <svg viewBox="0 0 80 80" fill="none" aria-hidden><rect x="13" y="19" width="54" height="45" rx="16" stroke="currentColor" strokeWidth="3"/><path d="M40 19V10M31 48h18M27 36h.1M53 36h.1" stroke="currentColor" strokeWidth="5" strokeLinecap="round"/><circle cx="40" cy="8" r="4" fill="currentColor"/></svg> }

function AnalysisDetails({ final, scene }: { final: Scene['final']; scene: Scene | null }) {
  const labels: Record<string, string> = { acoustic: 'Voice', behaviour: 'Conversation', semantic: 'Content' }
  const layers = final.layers.filter((layer) => layer.scored && ['acoustic', 'behaviour'].includes(layer.key))
  const duration = scene?.duration_s ?? 5
  // The fusion metrics only exist when the fusion prefixes were walked; a live phone call skips them (they
  // would cost a paid re-run per chunk), and their caller_speech_sec is then a real 0, not a missing value
  // `??` would fall through. The acoustic walk measured the same seconds either way, so read that instead.
  const measured = scene && (scene.fusion_steps.length ? scene.fusion_metrics : scene.metrics)
  const speech = measured ? measured.caller_speech_sec : 5
  return <div className="analysis-details">
    <div className="audio-facts">
      <span><small>Audio</small><b>{duration.toFixed(1)} s</b></span>
      <span><small>Caller speech</small><b>{speech.toFixed(1)} s</b></span>
    </div>
    <div className="signal-facts">
      {layers.map((layer) => {
        if (layer.abstained) return <span key={layer.key}><small>{labels[layer.key] ?? layer.display}</small><b className="neutral">No signal</b></span>
        const synthetic = layer.probability >= .5
        return <span key={layer.key}><small>{labels[layer.key] ?? layer.display}</small><b className={synthetic ? 'synthetic' : 'human'}>{synthetic ? 'Synthetic' : 'Human'} · {pct(synthetic ? layer.probability : 1 - layer.probability)}</b></span>
      })}
    </div>
  </div>
}

function envelopeAt(values: number[] | undefined, time: number, duration: number | undefined) {
  if (!values?.length || !duration) return 0
  return values[Math.min(Math.floor(time / duration * values.length), values.length - 1)] ?? 0
}

function Waveform({ values, progress }: { values: number[]; progress: number }) {
  const bars = values.length ? Array.from({ length: 90 }, (_, i) => values[Math.floor(i / 90 * values.length)] ?? 0) : Array.from({ length: 90 }, (_, i) => .12 + Math.abs(Math.sin(i * .61)) * .2)
  return <div className="waveform">{bars.map((height, i) => <i key={i} className={i / bars.length <= progress ? 'passed' : ''} style={{ height: `${Math.max(8, height * 100)}%` }} />)}</div>
}

function LiveWave({ level }: { level: number }) {
  return <div className="live-wave" aria-hidden>{[.45, .8, .6, 1, .7, .5, .85, .4].map((weight, index) => <i key={index} style={{ transform: `scaleY(${Math.max(.12, level * weight)})` }} />)}</div>
}

function InstantResult({ final, latencyMs }: { final: Scene['final']; latencyMs: number | null }) {
  const tone = final.is_synthetic ? 'synthetic' : 'human'
  return <section className={`instant-result ${tone}`} aria-live="polite">
    <div className="instant-verdict"><span>Result</span><strong>{final.is_synthetic ? 'Synthetic' : 'Human'}</strong></div>
    <div className="instant-stat"><span>Confidence</span><b>{pct(final.confidence)}</b></div>
    <div className="instant-stat"><span>Voice latency</span><b>{latencyMs == null ? '--' : `${Math.round(latencyMs)} ms`}</b></div>
  </section>
}

// ----------------------------------------------------------------------------- "Try being the caller"
//
// One button, and no phone number anywhere on the page: the server dials TWILIO_TO_NUMBER from the
// project .env, says a line, records what the person answers, and scores THAT recording. It is the only
// path here where the audio has genuinely been through a telephone network rather than a simulation of
// one, which is why the server scores it with Robust V2 - down a real line the V1 specialist falls to
// 77.5 % while V2 holds at 100 %.
//
// Nothing has to reach this machine. src/twilio_demo.py carries the TwiML inline and polls Twilio's REST
// API for the recording, so there is no webhook, no tunnel and no public URL to fail on stage.

const PHONE_PHASE: Record<TwilioJob['state'], { label: string; live: boolean }> = {
  queued: { label: 'Calling you', live: false },
  ringing: { label: 'Ringing', live: false },
  recording: { label: 'Speak now', live: true },
  scoring: { label: 'Analyzing', live: false },
  done: { label: 'Analyzing', live: false },      // held until the scene it explains is in hand
  failed: { label: 'Call failed', live: false },
}

function PhoneIcon() {
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden width="13" height="13"><path d="M6.5 3.5h3l1.5 4-2 1.4a11 11 0 0 0 5.1 5.1l1.4-2 4 1.5v3a1.5 1.5 0 0 1-1.6 1.5C10.2 17.6 6.4 13.8 5 5.1A1.5 1.5 0 0 1 6.5 3.5Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/></svg>
}

function CallerCta({ onScene, onPhase, onStart, onError }: {
  onScene: (scene: Scene, sid: string) => void
  onPhase: (phase: Phone) => void
  onStart: () => void
  onError: (message: string) => void
}) {
  const [config, setConfig] = useState<TwilioConfig | null>(null)
  const [job, setJob] = useState<TwilioJob | null>(null)
  const [starting, setStarting] = useState(false)

  useEffect(() => { api.twilio.config().then(setConfig).catch(() => setConfig(null)) }, [])

  // Poll while the call is in flight: ringing -> recording -> scoring is what the job reports, and one
  // request a second is the rate those states actually change at.
  useEffect(() => {
    if (!job || job.state === 'done' || job.state === 'failed') return
    const timer = window.setTimeout(async () => {
      try { setJob(await api.twilio.status(job.call_sid)) }
      catch (e) { onError((e as Error).message); onPhase(null); setJob(null) }
    }, 1000)
    return () => clearTimeout(timer)
  }, [job])

  // The verdict appears only once the scene that explains it is in hand, so the page never paints a
  // result it cannot then show the working for.
  useEffect(() => {
    if (!job) return
    if (job.state === 'failed') {
      onError(job.error || 'the call ended without a recording'); onPhase(null); setJob(null); return
    }
    onPhase(PHONE_PHASE[job.state])
    if (job.state !== 'done') return
    let cancelled = false
    api.twilio.analysis(job.call_sid)
      .then((scene) => { if (!cancelled) onScene(scene, job.call_sid) })
      .catch((e) => { if (!cancelled) onError((e as Error).message) })
      .finally(() => { if (!cancelled) { onPhase(null); setJob(null) } })
    return () => { cancelled = true }
  }, [job?.state])

  const start = async () => {
    setStarting(true); onStart(); onPhase(PHONE_PHASE.queued)
    try { setJob(await api.twilio.call(30)) }
    catch (e) { onError((e as Error).message); onPhase(null) }
    finally { setStarting(false) }
  }

  const unavailable = !configured ? 'Detection service unavailable'
    : config === null ? ''
    : !config.configured ? `Needs ${config.missing.join(', ')}`
    : !config.has_default_to ? 'Needs TWILIO_TO_NUMBER'
    : ''
  const busy = starting || !!job
  const label = job ? PHONE_PHASE[job.state].label : starting ? 'Calling you' : 'Try being the caller'

  return <div className="caller-cta">
    <button className={`btn caller${busy ? ' busy' : ''}`} onClick={start} disabled={!!unavailable || busy}
            title={unavailable || `Altur phones ${config?.default_to ?? 'you'} and scores your own voice`}>
      <PhoneIcon />{label}
    </button>
    <span className="caller-note">
      {unavailable || (busy ? `Calling ${config?.default_to ?? ''}` : 'We call you · Robust V2')}
    </span>
  </div>
}
