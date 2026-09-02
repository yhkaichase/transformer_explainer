import type { Audience } from '../content/types'
import { useAudience } from '../state/audience-context'

const OPTIONS: { value: Audience; label: string; hint: string }[] = [
  { value: 'executive', label: '임원용', hint: '핵심과 비유만' },
  { value: 'engineer', label: '엔지니어용', hint: '수식과 구조까지' },
]

export function AudienceToggle() {
  const { audience, setAudience } = useAudience()

  return (
    <fieldset className="audience-toggle">
      <legend className="visually-hidden">설명 수준 선택</legend>
      {OPTIONS.map((option) => (
        <label
          key={option.value}
          className={`audience-option${audience === option.value ? ' is-active' : ''}`}
        >
          <input
            type="radio"
            name="audience"
            value={option.value}
            checked={audience === option.value}
            onChange={() => setAudience(option.value)}
          />
          <span className="audience-label">{option.label}</span>
          <span className="audience-hint">{option.hint}</span>
        </label>
      ))}
    </fieldset>
  )
}
