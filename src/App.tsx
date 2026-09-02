import { AudienceToggle } from './components/AudienceToggle'
import { Hero } from './components/Hero'
import { SectionView } from './components/SectionView'
import { Sources } from './components/Sources'
import { Toc } from './components/Toc'
import { sections } from './content/sections'
import { AudienceProvider } from './state/AudienceProvider'

export default function App() {
  return (
    <AudienceProvider>
      <a className="skip-link" href="#main">
        본문으로 건너뛰기
      </a>
      <div className="site-header">
        <a className="brand" href="#top">
          <span className="brand-mark" aria-hidden="true">
            T
          </span>
          트랜스포머 쉽게 이해하기
        </a>
        <AudienceToggle />
      </div>
      <div className="layout" id="top">
        <Toc sections={sections} />
        <main id="main" className="content">
          <Hero />
          {sections.map((section) => (
            <SectionView key={section.id} section={section} />
          ))}
          <Sources />
        </main>
      </div>
    </AudienceProvider>
  )
}
