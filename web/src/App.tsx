import { useState } from 'react'
import { nav } from './data/copy'
import { useLenis } from './lib/useLenis'
import { useReducedMotion } from './lib/useReducedMotion'
import { ActLock } from './scenes/ActLock'
import { ActName } from './scenes/ActName'
import { Signals } from './sections/Signals'
import { TrapDemo } from './sections/TrapDemo'
import { Carrier } from './sections/Carrier'
import { Pipeline } from './sections/Pipeline'
import { Endpoint } from './sections/Endpoint'
import { Closing } from './sections/Closing'

export default function App() {
  const reduced = useReducedMotion()
  useLenis(!reduced)
  // the rest of the page only exists once the lock has been tapped
  const [broken, setBroken] = useState(false)

  return (
    <>
      <nav className="nav">
        <a href="#hero" className="brand" aria-label={nav.brandLabel}><b>{nav.brand}</b></a>
        <span className="sub">{nav.sub}</span>
        <a href={nav.link.href} style={{ visibility: broken ? 'visible' : 'hidden' }}>{nav.link.label}</a>
      </nav>

      <ActLock reduced={reduced} onBroken={() => setBroken(true)} />

      {broken && (
        <>
          <ActName reduced={reduced} />
          <main>
            <Signals />
            <div className="rule" />
            <TrapDemo />
            <div className="rule" />
            <Carrier />
            <div className="rule" />
            <Pipeline />
            <div className="rule" />
            <Endpoint />
          </main>
          <Closing />
        </>
      )}
    </>
  )
}
