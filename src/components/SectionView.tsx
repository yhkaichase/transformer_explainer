import type { ComponentType } from 'react'
import type { InteractiveKind, Section } from '../content/types'
import { useAudience } from '../state/audience-context'
import { useLocale } from '../state/locale-context'
import { AttentionDemo } from './AttentionDemo'
import { Callout } from './Callout'
import { FeedForwardDemo } from './FeedForwardDemo'
import { FlowDiagram } from './FlowDiagram'
import { LayerStackDemo } from './LayerStackDemo'
import { PositionalEncodingDemo } from './PositionalEncodingDemo'
import { TemperatureDemo } from './TemperatureDemo'
import { TokenizerDemo } from './TokenizerDemo'
import { TrainingDemo } from './TrainingDemo'

const INTERACTIVES: Record<InteractiveKind, ComponentType> = {
  tokenizer: TokenizerDemo,
  flow: FlowDiagram,
  temperature: TemperatureDemo,
  positional: PositionalEncodingDemo,
  attention: AttentionDemo,
  ffn: FeedForwardDemo,
  stack: LayerStackDemo,
  training: TrainingDemo,
}

export function SectionView({ section }: { section: Section }) {
  const { audience } = useAudience()
  const { content } = useLocale()
  const Interactive = section.interactive ? INTERACTIVES[section.interactive] : null
  const headingId = `${section.id}-title`

  return (
    <section id={section.id} className="section" aria-labelledby={headingId}>
      <p className="section-number">{String(section.number).padStart(2, '0')}</p>
      <h2 id={headingId} className="section-title">
        {section.title}
        {section.status === 'draft' && <span className="badge">{content.ui.draftBadge}</span>}
      </h2>
      <p className="section-tagline">{section.tagline}</p>

      <Callout kind="analogy" title={content.ui.analogyTitle}>
        <p>{section.analogy}</p>
      </Callout>

      {section.executive.map((paragraph, i) => (
        <p key={i}>{paragraph}</p>
      ))}

      {Interactive && <Interactive />}

      {audience === 'engineer' && (
        <Callout kind="engineer" title={content.ui.engineerTitle}>
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
