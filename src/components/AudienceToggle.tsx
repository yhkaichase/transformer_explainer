import type { Audience } from '../content/types'
import { useAudience } from '../state/audience-context'
import { useLocale } from '../state/locale-context'
import { SegmentedControl } from './SegmentedControl'

const AUDIENCES: Audience[] = ['executive', 'engineer']

export function AudienceToggle() {
  const { audience, setAudience } = useAudience()
  const { content } = useLocale()
  const options = AUDIENCES.map((value) => ({ value, ...content.ui.audience[value] }))

  return (
    <SegmentedControl
      legend={content.ui.audienceLegend}
      name="audience"
      options={options}
      value={audience}
      onChange={setAudience}
    />
  )
}
