import { useAudience } from '../state/audience-context'
import { useLocale } from '../state/locale-context'

export function SummaryCard() {
  const { content } = useLocale()
  const { audience } = useAudience()
  const strings = content.ui.summary

  return (
    <div className="demo summary-card" data-testid="summary-card">
      <section className="summary-panel" aria-labelledby="summary-card-title">
        <h3 id="summary-card-title">{strings.cardTitle}</h3>
        <ol className="summary-steps">
          {strings.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <h4>{strings.takeawaysTitle}</h4>
        <ul className="summary-takeaways">
          {strings.takeaways.map((item) => (
            <li key={item.lead}>
              <strong>{item.lead}</strong> {item.text}
            </li>
          ))}
        </ul>
      </section>

      {audience === 'engineer' && (
        <section className="summary-panel summary-engineer" aria-labelledby="summary-next-title">
          <h3 id="summary-next-title">{strings.nextStepsTitle}</h3>
          <ol className="summary-next">
            {strings.nextSteps.map((item) => (
              <li key={item.url}>
                <a href={item.url} target="_blank" rel="noreferrer">
                  {item.label}
                </a>
                <span className="summary-detail">{item.detail}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  )
}
