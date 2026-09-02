import { AudienceToggle } from './components/AudienceToggle'
import { Hero } from './components/Hero'
import { LocaleToggle } from './components/LocaleToggle'
import { SectionView } from './components/SectionView'
import { Sources } from './components/Sources'
import { Toc } from './components/Toc'
import { AudienceProvider } from './state/AudienceProvider'
import { useLocale } from './state/locale-context'
import { LocaleProvider } from './state/LocaleProvider'

function Page() {
  const { content } = useLocale()

  return (
    <>
      <a className="skip-link" href="#main">
        {content.ui.skipToContent}
      </a>
      <div className="site-header">
        <a className="brand" href="#top">
          <span className="brand-mark" aria-hidden="true">
            T
          </span>
          {content.ui.brand}
        </a>
        <div className="header-controls">
          <LocaleToggle />
          <AudienceToggle />
        </div>
      </div>
      <div className="layout" id="top">
        <Toc />
        <main id="main" className="content">
          <Hero />
          {content.sections.map((section) => (
            <SectionView key={section.id} section={section} />
          ))}
          <Sources />
        </main>
      </div>
    </>
  )
}

export default function App() {
  return (
    <LocaleProvider>
      <AudienceProvider>
        <Page />
      </AudienceProvider>
    </LocaleProvider>
  )
}
