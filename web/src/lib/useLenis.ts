import Lenis from 'lenis'
import { useEffect } from 'react'

let lenis: Lenis | null = null

/** Smooth scroll. Off under reduced motion, where the page uses native scrolling. */
export function useLenis(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return
    lenis = new Lenis({ autoRaf: true, anchors: true })
    return () => {
      lenis?.destroy()
      lenis = null
    }
  }, [enabled])
}

export function scrollToElement(el: HTMLElement) {
  if (lenis) lenis.scrollTo(el)
  else el.scrollIntoView()
}

export function scrollToY(y: number) {
  if (lenis) lenis.scrollTo(y)
  else window.scrollTo({ top: y })
}
