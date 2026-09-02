import { LOCALES } from '../content'
import { useLocale } from '../state/locale-context'
import { SegmentedControl } from './SegmentedControl'

export function LocaleToggle() {
  const { locale, setLocale, content } = useLocale()

  return (
    <SegmentedControl
      legend={content.ui.localeLegend}
      name="locale"
      options={LOCALES}
      value={locale}
      onChange={setLocale}
    />
  )
}
