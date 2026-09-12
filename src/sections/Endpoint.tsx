import { useRef, useState, type ReactNode } from 'react'
import { endpoint } from '../data/copy'
import { Html } from '../lib/Html'

/** Colours JSON the way the prototype did: strings in .s, numbers and booleans in .n. */
function json(value: unknown): ReactNode[] {
  const src = JSON.stringify(value, null, 2)
  const re = /"(?:[^"\\]|\\.)*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?/g
  const out: ReactNode[] = []
  let last = 0, m: RegExpExecArray | null
  while ((m = re.exec(src))) {
    out.push(src.slice(last, m.index))
    out.push(<span key={m.index} className={m[0][0] === '"' ? 's' : 'n'}>{m[0]}</span>)
    last = m.index + m[0].length
  }
  out.push(src.slice(last))
  return out
}

export function Endpoint() {
  const preRef = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)
  function copy() {
    const pre = preRef.current
    if (!pre) return
    const ok = () => { setCopied(true); setTimeout(() => setCopied(false), 1600) }
    // plain-http (a LAN demo) has no navigator.clipboard, and some embedded browsers deny it
    const legacy = () => {
      const range = document.createRange(); range.selectNodeContents(pre)
      const sel = getSelection(); sel?.removeAllRanges(); sel?.addRange(range)
      const done = document.execCommand('copy'); sel?.removeAllRanges()
      if (done) ok()
    }
    if (navigator.clipboard) navigator.clipboard.writeText(pre.textContent ?? '').then(ok, legacy)
    else legacy()
  }
  return (
    <section className="band" id="api">
      <h2>{endpoint.heading}</h2>
      <div className="api">
        <div className="codebox">
          <button type="button" className="copy" onClick={copy} aria-live="polite">{copied ? endpoint.copied : endpoint.copy}</button>
          <pre ref={preRef}>
            <span className="c">{endpoint.method}</span> {endpoint.path}{'\n'}
            <span className="c">Content-Type: {endpoint.contentType}</span>{'\n\n'}
            {json(endpoint.request)}{'\n\n'}
            <span className="c">{endpoint.status}</span>{'\n'}
            {json(endpoint.response)}
          </pre>
        </div>
        <div>
          <ul className="specs">
            {endpoint.specs.map((s) => (
              <li key={s.label}><b>{s.label}</b><Html as="span" html={s.text} /></li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}
