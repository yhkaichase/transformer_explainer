import { useState } from 'react'
import { crossEntropy, gradientStep, softmax } from '../lib/math'
import { useLocale } from '../state/locale-context'

export const LEARNING_RATE = 0.5

export function TrainingDemo() {
  const { content } = useLocale()
  const strings = content.ui.training
  const initialScores = strings.candidates.map((candidate) => candidate.score)
  const answerIndex = strings.candidates.findIndex(
    (candidate) => candidate.token === strings.answer,
  )
  const [scores, setScores] = useState(initialScores)
  const [steps, setSteps] = useState(0)

  const probabilities = softmax(scores)
  const loss = crossEntropy(probabilities, answerIndex)

  const train = (count: number) => {
    let next = scores
    for (let i = 0; i < count; i++) {
      next = gradientStep(next, answerIndex, LEARNING_RATE)
    }
    setScores(next)
    setSteps(steps + count)
  }

  const reset = () => {
    setScores(initialScores)
    setSteps(0)
  }

  return (
    <div className="demo" data-testid="training-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <p className="demo-context">
        {strings.context}{' '}
        <span className="training-answer">
          ({strings.answerLabel}: {strings.answer})
        </span>
      </p>

      <dl className="stat-row" aria-live="polite">
        <div className="stat">
          <dt>{strings.stepsLabel}</dt>
          <dd>{steps}</dd>
        </div>
        <div className="stat">
          <dt>{strings.lossLabel}</dt>
          <dd>{loss.toFixed(2)}</dd>
        </div>
        <div className="stat">
          <dt>{strings.answerProbabilityLabel}</dt>
          <dd>{(probabilities[answerIndex] * 100).toFixed(1)}%</dd>
        </div>
      </dl>

      <table className="prob-table">
        <caption className="visually-hidden">{strings.tableCaption}</caption>
        <thead>
          <tr>
            <th scope="col">{strings.columns.token}</th>
            <th scope="col">{strings.columns.probability}</th>
          </tr>
        </thead>
        <tbody>
          {strings.candidates.map((candidate, index) => (
            <tr key={candidate.token} className={index === answerIndex ? 'is-drawn' : undefined}>
              <th scope="row">
                {candidate.token}
                {index === answerIndex && (
                  <span className="answer-mark"> {strings.correctMark}</span>
                )}
              </th>
              <td>
                <div className="prob-cell">
                  <div className="prob-track" aria-hidden="true">
                    <div className="prob-bar" style={{ width: `${probabilities[index] * 100}%` }} />
                  </div>
                  <span className="prob-value">{(probabilities[index] * 100).toFixed(1)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="demo-actions">
        <button type="button" className="button-primary" onClick={() => train(1)}>
          {strings.stepOnce}
        </button>
        <button type="button" onClick={() => train(10)}>
          {strings.stepMany}
        </button>
        <button type="button" onClick={reset} disabled={steps === 0}>
          {strings.reset}
        </button>
      </div>

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
