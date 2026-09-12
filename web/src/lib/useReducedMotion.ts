import { useEffect, useState } from 'react'

const mq = matchMedia('(prefers-reduced-motion: reduce)')
/** `?reduced-motion` in the URL forces it on, so the static layout can be checked without touching OS settings. */
const forced = new URLSearchParams(location.search).has('reduced-motion')

const current = () => forced || mq.matches
/** The stylesheet keys the static layout off this class (see index.css). */
const apply = () => document.documentElement.classList.toggle('reduced-motion', current())
apply()

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(current)
  useEffect(() => {
    const update = () => {
      apply()
      setReduced(current())
    }
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [])
  return reduced
}
