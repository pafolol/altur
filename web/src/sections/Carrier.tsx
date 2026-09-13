import { carrier } from '../data/copy'
import { Html } from '../lib/Html'

/**
 * The carrier-portability chart. Every number is the real call-level accuracy at the deployed
 * threshold, read from outputs/robust/evaluation_matrix.csv — the same table reports/
 * TELEPHONE_ROBUSTNESS_REPORT.md §8 prints. Nothing here is rounded in our favour.
 *
 * Colour: two series in a deliberate hierarchy, not two co-equal categories — the old model
 * recedes, ours leads. Validated on the dark surface for CVD separation (ΔE 17.3, target ≥8),
 * normal-vision separation (18.1, floor 15) and ≥3:1 contrast. Identity is never carried by
 * colour alone: there is a legend AND a value printed on every single bar.
 */
const BEFORE = '#5E7A96'
const AFTER = '#5BB0F5'

export function Carrier() {
  return (
    <section className="band" id="carrier">
      <h2>{carrier.heading}</h2>
      <Html className="lede" html={carrier.lede} />

      <figure className="chart">
        <figcaption className="chart-legend">
          <span><i style={{ background: BEFORE }} />{carrier.legend.before}</span>
          <span><i style={{ background: AFTER }} />{carrier.legend.after}</span>
        </figcaption>

        <div className="chart-rows">
          {carrier.rows.map((r) => (
            <div className="chart-row" key={r.label}>
              <div className="chart-label">
                <b>{r.label}</b>
                <span>{r.note}</span>
              </div>
              <div className="chart-bars">
                <div className="chart-bar">
                  <i style={{ width: `${r.before}%`, background: BEFORE }} />
                  <b style={{ color: BEFORE }}>{r.before.toFixed(1)}%</b>
                </div>
                <div className="chart-bar">
                  <i style={{ width: `${r.after}%`, background: AFTER }} />
                  <b style={{ color: AFTER }}>{r.after.toFixed(1)}%</b>
                </div>
              </div>
            </div>
          ))}
        </div>

        <table className="chart-table">
          <caption>{carrier.tableCaption}</caption>
          <thead>
            <tr>
              <th scope="col">{carrier.tableCols.domain}</th>
              <th scope="col">{carrier.legend.before}</th>
              <th scope="col">{carrier.legend.after}</th>
            </tr>
          </thead>
          <tbody>
            {carrier.rows.map((r) => (
              <tr key={r.label}>
                <th scope="row">{r.label}</th>
                <td>{r.before.toFixed(1)}%</td>
                <td>{r.after.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </figure>

      <Html className="chart-note" html={carrier.note} />
    </section>
  )
}
