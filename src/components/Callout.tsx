import type { ReactNode } from 'react'

export type CalloutKind = 'analogy' | 'engineer' | 'note'

interface CalloutProps {
  kind: CalloutKind
  title: string
  children: ReactNode
}

export function Callout({ kind, title, children }: CalloutProps) {
  return (
    <aside className={`callout callout-${kind}`}>
      <p className="callout-title">{title}</p>
      <div className="callout-body">{children}</div>
    </aside>
  )
}
