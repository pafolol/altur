import type { ElementType, HTMLAttributes } from 'react'

type Props = HTMLAttributes<HTMLElement> & { as?: ElementType; html: string }

/**
 * Copy in src/data may carry inline markup (&nbsp;, <i>, <code>), exactly as
 * it was written in the prototype. It is our own text, so it goes in as-is.
 */
export function Html({ as: Tag = 'p', html, ...rest }: Props) {
  return <Tag dangerouslySetInnerHTML={{ __html: html }} {...rest} />
}
