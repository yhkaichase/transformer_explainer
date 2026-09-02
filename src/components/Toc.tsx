import type { Section } from '../content/types'

export function Toc({ sections }: { sections: Section[] }) {
  return (
    <nav className="toc" aria-label="목차">
      <p className="toc-title">목차</p>
      <ol className="toc-list">
        {sections.map((section) => (
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
