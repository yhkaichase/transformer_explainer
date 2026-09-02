import type { ComponentType } from 'react'
import type { InteractiveKind, Section } from '../content/types'
import { useAudience } from '../state/audience-context'
import { Callout } from './Callout'
import { TokenizerDemo } from './TokenizerDemo'

const INTERACTIVES: Record<InteractiveKind, ComponentType> = {
  tokenizer: TokenizerDemo,
}

export const ENGINEER_CALLOUT_TITLE = '엔지니어를 위한 더 깊은 설명'

export function SectionView({ section }: { section: Section }) {
  const { audience } = useAudience()
  const Interactive = section.interactive ? INTERACTIVES[section.interactive] : null
  const headingId = `${section.id}-title`

  return (
    <section id={section.id} className="section" aria-labelledby={headingId}>
      <p className="section-number">{String(section.number).padStart(2, '0')}</p>
      <h2 id={headingId} className="section-title">
        {section.title}
        {section.status === 'draft' && <span className="badge">초안</span>}
      </h2>
      <p className="section-tagline">{section.tagline}</p>

      <Callout kind="analogy" title="비유로 이해하기">
        <p>{section.analogy}</p>
      </Callout>

      {section.executive.map((paragraph, i) => (
        <p key={i}>{paragraph}</p>
      ))}

      {Interactive && <Interactive />}

      {audience === 'engineer' && (
        <Callout kind="engineer" title={ENGINEER_CALLOUT_TITLE}>
          {section.engineer.map((paragraph, i) => (
            <p key={i}>{paragraph}</p>
          ))}
          {section.keyTerms && section.keyTerms.length > 0 && (
            <dl className="key-terms">
              {section.keyTerms.map((item) => (
                <div key={item.term} className="key-term">
                  <dt>{item.term}</dt>
                  <dd>{item.meaning}</dd>
                </div>
              ))}
            </dl>
          )}
        </Callout>
      )}
    </section>
  )
}
