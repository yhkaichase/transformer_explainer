import { useLocale } from '../state/locale-context'

export function Sources() {
  const { content } = useLocale()

  return (
    <footer className="sources" aria-labelledby="sources-title">
      <h2 id="sources-title">{content.ui.sourcesTitle}</h2>
      <ul>
        {content.sources.map((source) => (
          <li key={source.url}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.label}
            </a>
            <span className="source-detail"> {source.detail}</span>
          </li>
        ))}
      </ul>
    </footer>
  )
}
