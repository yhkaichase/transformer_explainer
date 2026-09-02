import { useId, useState } from 'react'
import { feedForward } from '../lib/math'
import { useLocale } from '../state/locale-context'

const fmt = (value: number) => value.toFixed(2)
const vec = (values: number[]) => `[${values.map(fmt).join(', ')}]`

export function FeedForwardDemo() {
  const { content } = useLocale()
  const strings = content.ui.ffn
  const [selected, setSelected] = useState(0)
  const groupName = useId()
  const result = feedForward(
    strings.inputs[selected],
    strings.w1,
    strings.b1,
    strings.w2,
    strings.b2,
  )
  const zeroCount = result.hidden.filter((value) => value === 0).length

  return (
    <div className="demo" data-testid="ffn-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <fieldset className="choice-group">
        <legend>{strings.selectLegend}</legend>
        {strings.tokens.map((token, index) => (
          <label key={token} className={`choice-chip${index === selected ? ' is-active' : ''}`}>
            <input
              type="radio"
              name={groupName}
              value={index}
              checked={index === selected}
              onChange={() => setSelected(index)}
            />
            {token}
          </label>
        ))}
      </fieldset>

      <div className="ffn-lanes" aria-hidden="true">
        {strings.tokens.map((token, index) => (
          <div key={token} className={`ffn-lane${index === selected ? ' is-active' : ''}`}>
            <span className="ffn-token">{token}</span>
            <span className="ffn-arrow">↓</span>
            <span className="ffn-box">FFN</span>
            <span className="ffn-arrow">↓</span>
            <span className="ffn-token ffn-token-out">{token}′</span>
          </div>
        ))}
      </div>
      <p className="demo-regime">{strings.sharedLabel}</p>

      <dl className="ffn-numbers" aria-live="polite">
        <div>
          <dt>{strings.stageLabels.input}</dt>
          <dd className="vector">{vec(strings.inputs[selected])}</dd>
        </div>
        <div>
          <dt>{strings.stageLabels.preActivation}</dt>
          <dd className="vector">{vec(result.preActivation)}</dd>
        </div>
        <div>
          <dt>{strings.stageLabels.hidden}</dt>
          <dd className="vector">{vec(result.hidden)}</dd>
        </div>
        <div>
          <dt>{strings.stageLabels.output}</dt>
          <dd className="vector">{vec(result.output)}</dd>
        </div>
      </dl>
      {zeroCount > 0 && <p className="demo-regime">{strings.zeroNote}</p>}

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
