import { useLocale } from '../state/locale-context'

export function Toc() {
  const { content } = useLocale()

  return (
    <nav className="toc" aria-label={content.ui.tocTitle}>
      <p className="toc-title">{content.ui.tocTitle}</p>
      <ol className="toc-list">
        {content.sections.map((section) => (
          <li key={section.id}>
            <a href={`#${section.id}`}>
              <span className="toc-number">{String(section.number).padStart(2, '0')}</span>
              <span>{section.shortTitle}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  )
}
