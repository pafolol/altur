import { useState, type CSSProperties } from 'react'
import { calls, signalNames, type Mode } from '../data/calls'
import { demo } from '../data/copy'
import { Html } from '../lib/Html'

const modes = Object.keys(demo.buttons) as Mode[]

export function TrapDemo() {
  const [mode, setMode] = useState<Mode>('human')
  const d = calls[mode]
  return (
    <section className="band" id="demo">
      <h2>{demo.heading}</h2>
      <Html className="lede" html={demo.lede} />

      <div className="demo" data-mode={mode}>
        <div className="demo-head">
          <p>{d.meta}</p>
          <div className="switch" role="group" aria-label={demo.switchLabel}>
            {modes.map((m) => (
              <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)}>{demo.buttons[m]}</button>
            ))}
          </div>
        </div>
        <div className="demo-body">
          <div className="script" key={mode}>
            {d.turns.map((t, i) => (
              <div className={'turn ' + t.who.toLowerCase()} key={i}>
                <b>{t.who}</b>
                <div>
                  <i>{t.text}</i>
                  {t.note && <span className="note">{t.note}</span>}
                </div>
              </div>
            ))}
          </div>
          <div className="readout">
            {signalNames.map((name, i) => (
              <div className="bar" key={name}>
                <span><u>{name}</u><em>{d.scores[i].toFixed(2)}</em></span>
                <i style={{ '--v': (d.scores[i] * 100).toFixed(0) + '%' } as CSSProperties} />
              </div>
            ))}
            <div className="verdict">
              {'{\n  '}<span className="k">"is_synthetic"</span>: <span className="v">{String(d.verdict.is_synthetic)}</span>
              {',\n  '}<span className="k">"confidence"</span>: <span className="v">{d.verdict.confidence.toFixed(3)}</span>
              {'\n}'}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
