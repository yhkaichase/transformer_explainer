import { useAudience } from '../state/audience-context'
import { useLocale } from '../state/locale-context'

export function Hero() {
  const { audience } = useAudience()
  const { content } = useLocale()
  const { hero } = content.ui

  return (
    <header className="hero">
      <p className="hero-kicker">{hero.kicker}</p>
      <h1>{hero.title}</h1>
      <p className="hero-lead">{hero.lead}</p>
      <p className="hero-mode" aria-live="polite">
        {hero.modeBefore}
        <strong>{content.ui.audience[audience].label}</strong>
        {hero.modeAfter[audience]}
      </p>
    </header>
  )
}
