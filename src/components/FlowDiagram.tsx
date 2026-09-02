import { useState } from 'react'
import { useLocale } from '../state/locale-context'

export function FlowDiagram() {
  const { content } = useLocale()
  const strings = content.ui.flow
  const [index, setIndex] = useState(0)
  const total = strings.stages.length
  const stage = strings.stages[index]

  return (
    <div className="demo" data-testid="flow-diagram">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <ol className="flow-stages" aria-label={strings.heading}>
        {strings.stages.map((item, i) => (
          <li key={item.section} className={i === index ? 'is-active' : undefined}>
            <button
              type="button"
              aria-current={i === index ? 'step' : undefined}
              onClick={() => setIndex(i)}
            >
              <span className="flow-number">{i + 1}</span>
              <span className="flow-label">{item.label}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="flow-detail" aria-live="polite">
        <p className="flow-step">{strings.stepLabel(index + 1, total)}</p>
        <p className="flow-text">{stage.detail}</p>
        <a href={`#${stage.section}`}>{strings.goToSection}</a>
      </div>

      <div className="demo-actions">
        <button
          type="button"
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={index === 0}
        >
          {strings.prev}
        </button>
        <button
          type="button"
          className="button-primary"
          onClick={() => setIndex((i) => Math.min(total - 1, i + 1))}
          disabled={index === total - 1}
        >
          {strings.next}
        </button>
      </div>
    </div>
  )
}
