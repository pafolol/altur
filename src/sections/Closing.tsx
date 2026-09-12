import { closing } from '../data/copy'
import { Html } from '../lib/Html'

export function Closing() {
  return (
    <section className="close">
      <div className="inner">
        <h2>{closing.heading}</h2>
        <Html className="lede" html={closing.lede} />
      </div>
      <footer>{closing.footer.map((f) => <span key={f}>{f}</span>)}</footer>
    </section>
  )
}
