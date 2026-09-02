import { sources } from '../content/sources'

export function Sources() {
  return (
    <footer className="sources" aria-labelledby="sources-title">
      <h2 id="sources-title">참고 자료</h2>
      <ul>
        {sources.map((source) => (
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
