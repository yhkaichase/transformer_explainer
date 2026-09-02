import { useId, useState } from 'react'
import { dot, scaledDotProductAttention } from '../lib/math'
import { useAudience } from '../state/audience-context'
import { useLocale } from '../state/locale-context'

const fmt = (value: number) => value.toFixed(2)
const vec = (values: number[]) => `[${values.map(fmt).join(', ')}]`
/** 곱셈 항 표기. 음수는 괄호로 감싸 "+ -1.00" 같은 표기를 피한다. */
const factor = (value: number) => (value < 0 ? `(${fmt(value)})` : fmt(value))

export function AttentionDemo() {
  const { content } = useLocale()
  const { audience } = useAudience()
  const strings = content.ui.attention
  const [query, setQuery] = useState(0)
  const [causal, setCausal] = useState(false)
  const groupName = useId()

  const dK = strings.k[0].length
  const scale = Math.sqrt(dK)
  const { weights, output } = scaledDotProductAttention(strings.q, strings.k, strings.v, { causal })
  const row = weights[query]
  const rawScores = strings.k.map((key) => dot(strings.q[query], key))
  const isMasked = (index: number) => causal && index > query
  const anyMasked = strings.tokens.some((_, index) => isMasked(index))
  const queryToken = strings.tokens[query]
  const newVector = output[query]

  return (
    <div className="demo" data-testid="attention-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <p className="demo-context">{strings.sentence}</p>

      <div className="attention-controls">
        <fieldset className="choice-group">
          <legend>{strings.queryLegend}</legend>
          {strings.tokens.map((token, index) => (
            <label key={token} className={`choice-chip${index === query ? ' is-active' : ''}`}>
              <input
                type="radio"
                name={groupName}
                value={index}
                checked={index === query}
                onChange={() => setQuery(index)}
              />
              {token}
            </label>
          ))}
        </fieldset>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={causal}
            onChange={(event) => setCausal(event.target.checked)}
          />
          {strings.causalLabel}
        </label>
      </div>

      <h4 className="demo-subheading">{strings.weightsHeading(queryToken)}</h4>
      <table className="prob-table">
        <caption className="visually-hidden">{strings.weightsHeading(queryToken)}</caption>
        <thead>
          <tr>
            <th scope="col">{strings.columns.token}</th>
            <th scope="col">{strings.columns.weight}</th>
          </tr>
        </thead>
        <tbody>
          {strings.tokens.map((token, index) => (
            <tr key={token} className={index === query ? 'is-self' : undefined}>
              <th scope="row">{token}</th>
              <td>
                {isMasked(index) ? (
                  <span className="masked">{strings.masked}</span>
                ) : (
                  <div className="prob-cell">
                    <div className="prob-track" aria-hidden="true">
                      <div className="prob-bar" style={{ width: `${row[index] * 100}%` }} />
                    </div>
                    <span className="prob-value">{(row[index] * 100).toFixed(1)}%</span>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {anyMasked && <p className="demo-regime">{strings.maskedNote(queryToken)}</p>}

      <p className="attention-output" aria-live="polite">
        <strong>{strings.outputHeading(queryToken)}</strong>{' '}
        <span className="vector">{vec(newVector)}</span>
        <span className="attention-output-dims">
          {' '}
          ({strings.valueDims[0]} {fmt(newVector[0])}, {strings.valueDims[1]} {fmt(newVector[1])})
        </span>
      </p>

      <details className="calc-details" open={audience === 'engineer'}>
        <summary>{strings.detailsSummary}</summary>

        <h5>{strings.steps.vectors}</h5>
        <div className="table-scroll">
          <table className="calc-table">
            <thead>
              <tr>
                <th scope="col">{strings.columns.token}</th>
                <th scope="col">{strings.columns.q}</th>
                <th scope="col">{strings.columns.k}</th>
                <th scope="col">{strings.columns.v}</th>
              </tr>
            </thead>
            <tbody>
              {strings.tokens.map((token, index) => (
                <tr key={token}>
                  <th scope="row">{token}</th>
                  <td className="vector">{vec(strings.q[index])}</td>
                  <td className="vector">{vec(strings.k[index])}</td>
                  <td className="vector">{vec(strings.v[index])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h5>{strings.steps.scores}</h5>
        <div className="table-scroll">
          <table className="calc-table">
            <thead>
              <tr>
                <th scope="col">{strings.columns.token}</th>
                <th scope="col">Q({queryToken}) · K</th>
                <th scope="col">{strings.columns.score}</th>
              </tr>
            </thead>
            <tbody>
              {strings.tokens.map((token, index) => (
                <tr key={token}>
                  <th scope="row">{token}</th>
                  <td className="vector">
                    {strings.q[query]
                      .map((qValue, d) => `${factor(qValue)}×${factor(strings.k[index][d])}`)
                      .join(' + ')}
                  </td>
                  <td className="vector">{fmt(rawScores[index])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h5>{strings.steps.scaled}</h5>
        <p className="vector">
          {strings.tokens
            .map((token, index) =>
              isMasked(index)
                ? `${token}: −∞`
                : `${token}: ${fmt(rawScores[index])} ÷ ${fmt(scale)} = ${fmt(rawScores[index] / scale)}`,
            )
            .join('   ')}
        </p>

        <h5>{strings.steps.softmax}</h5>
        <p className="vector">
          {strings.tokens.map((token, index) => `${token}: ${fmt(row[index])}`).join('   ')}
        </p>

        <h5>{strings.steps.output}</h5>
        <p className="vector">
          {strings.tokens
            .map((_, index) => `${fmt(row[index])} × ${vec(strings.v[index])}`)
            .join(' + ')}{' '}
          = {vec(newVector)}
        </p>
      </details>

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
