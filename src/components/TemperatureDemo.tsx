import { useId, useState } from 'react'
import { sampleIndex, softmax } from '../lib/math'
import { useLocale } from '../state/locale-context'

const MIN_TEMPERATURE = 0.1
const MAX_TEMPERATURE = 3
const TEMPERATURE_STEP = 0.1

function formatPercent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}

function regimeOf(temperature: number): 'low' | 'neutral' | 'high' {
  if (temperature < 0.95) return 'low'
  if (temperature > 1.05) return 'high'
  return 'neutral'
}

interface TemperatureDemoProps {
  /** 뽑기에 쓰는 난수 함수. 테스트에서 고정 값을 넣기 위해 바꿀 수 있다. */
  random?: () => number
}

export function TemperatureDemo({ random = Math.random }: TemperatureDemoProps) {
  const { content } = useLocale()
  const strings = content.ui.temperature
  const [temperature, setTemperature] = useState(1)
  const [draws, setDraws] = useState<string[]>([])
  const sliderId = useId()

  const probabilities = softmax(
    strings.candidates.map((candidate) => candidate.score),
    temperature,
  )
  const lastDraw = draws.length > 0 ? draws[draws.length - 1] : null

  const draw = (count: number) => {
    const picked: string[] = []
    for (let i = 0; i < count; i++) {
      picked.push(strings.candidates[sampleIndex(probabilities, random)].token)
    }
    setDraws(picked)
  }

  return (
    <div className="demo" data-testid="temperature-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <p className="demo-context">{strings.context}</p>

      <div className="slider-row">
        <label htmlFor={sliderId}>{strings.sliderLabel}</label>
        <input
          id={sliderId}
          type="range"
          min={MIN_TEMPERATURE}
          max={MAX_TEMPERATURE}
          step={TEMPERATURE_STEP}
          value={temperature}
          aria-valuetext={strings.valueLabel(temperature)}
          onChange={(event) => {
            setTemperature(Number(event.target.value))
            setDraws([])
          }}
        />
        <output htmlFor={sliderId} className="slider-value">
          {temperature.toFixed(1)}
        </output>
      </div>
      <p className="demo-regime" aria-live="polite">
        {strings.regimes[regimeOf(temperature)]}
      </p>

      <table className="prob-table">
        <caption className="visually-hidden">{strings.tableCaption}</caption>
        <thead>
          <tr>
            <th scope="col" className="prob-rank">
              {strings.columns.rank}
            </th>
            <th scope="col">{strings.columns.token}</th>
            <th scope="col">{strings.columns.probability}</th>
          </tr>
        </thead>
        <tbody>
          {strings.candidates.map((candidate, index) => {
            const probability = probabilities[index]
            const isDrawn = candidate.token === lastDraw
            return (
              <tr key={candidate.token} className={isDrawn ? 'is-drawn' : undefined}>
                <td className="prob-rank">{index + 1}</td>
                <th scope="row">
                  {candidate.token}
                  {isDrawn && (
                    <span className="drawn-mark" aria-hidden="true">
                      {' '}
                      ✓
                    </span>
                  )}
                </th>
                <td>
                  <div className="prob-cell">
                    <div className="prob-track" aria-hidden="true">
                      <div className="prob-bar" style={{ width: `${probability * 100}%` }} />
                    </div>
                    <span className="prob-value">{formatPercent(probability)}</span>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <div className="demo-actions">
        <button type="button" className="button-primary" onClick={() => draw(1)}>
          {strings.draw}
        </button>
        <button type="button" onClick={() => draw(10)}>
          {strings.drawMany}
        </button>
      </div>

      <div className="draw-results" data-testid="draw-results" aria-live="polite">
        {draws.length > 0 && (
          <>
            <span className="draw-results-title">{strings.resultHeading}: </span>
            {draws.map((token, i) => (
              <span key={`${i}-${token}`} className="draw-chip">
                {token}
              </span>
            ))}
          </>
        )}
      </div>

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
