export interface SegmentedOption<T extends string> {
  value: T
  label: string
  hint?: string
}

interface SegmentedControlProps<T extends string> {
  /** 스크린 리더용 그룹 이름 */
  legend: string
  /** 라디오 그룹 이름 (페이지 안에서 고유해야 한다) */
  name: string
  options: SegmentedOption<T>[]
  value: T
  onChange: (value: T) => void
}

/** 라디오 버튼으로 만든 세그먼트 컨트롤. 키보드와 스크린 리더로도 쓸 수 있다. */
export function SegmentedControl<T extends string>({
  legend,
  name,
  options,
  value,
  onChange,
}: SegmentedControlProps<T>) {
  return (
    <fieldset className="segmented">
      <legend className="visually-hidden">{legend}</legend>
      {options.map((option) => (
        <label
          key={option.value}
          className={`segmented-option${value === option.value ? ' is-active' : ''}`}
        >
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span className="segmented-label">{option.label}</span>
          {option.hint && <span className="segmented-hint">{option.hint}</span>}
        </label>
      ))}
    </fieldset>
  )
}
