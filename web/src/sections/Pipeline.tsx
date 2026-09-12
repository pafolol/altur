import { pipeline } from '../data/copy'
import { Html } from '../lib/Html'

export function Pipeline() {
  const tl = pipeline.timeline
  return (
    <section className="band" id="pipeline">
      <h2>{pipeline.heading}</h2>
      <Html className="lede" html={pipeline.lede} />
      <div className="flow">
        {pipeline.steps.map((s) => (
          <div className="node" key={s.label}>
            <b>{s.label}</b>
            <h3>{s.title}</h3>
            <Html html={s.text} />
          </div>
        ))}
      </div>
      <div className="timeline">
        <h3>{tl.heading}</h3>
        <div className="tl">
          <b style={{ width: tl.verdictAt + '%' }} />
          {tl.markers.map((m) => <i key={m.label} style={{ left: m.at + '%' }} />)}
          <em style={{ left: tl.verdictAt + '%' }} />
          {tl.markers.map((m) => <u key={m.label} style={{ left: m.at + '%' }}>{m.label}</u>)}
        </div>
        <div className="tl-axis">{tl.axis.map((a) => <span key={a}>{a}</span>)}</div>
        <Html className="lede" html={tl.note} />
      </div>
    </section>
  )
}
