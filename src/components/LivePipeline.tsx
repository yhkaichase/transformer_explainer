import { useId, useMemo, useState } from 'react'
import { loadTinyModel } from '../lib/tinyTransformer'
import { useLocale } from '../state/locale-context'
import { HeatmapTable } from './HeatmapTable'

const TOP_K = 10
const MIN_TEMPERATURE = 0.1
const MAX_TEMPERATURE = 3

/** 글자를 화면용으로 바꾼다: 공백은 ␣, 줄바꿈은 ⏎. */
function displayChar(char: string): string {
  if (char === ' ') return '␣'
  if (char === '\n') return '⏎'
  return char
}

/** -1..1 로 정규화된 값을 일곱 단계 색 클래스로 (파랑 음수, 회색 0, 빨강 양수). */
function bucketClass(value: number): string {
  const bucket = Math.round(Math.max(-1, Math.min(1, value)) * 3)
  if (bucket === 0) return 'cell-0'
  return bucket < 0 ? `cell-n${-bucket}` : `cell-p${bucket}`
}

interface LivePipelineProps {
  /** 뽑기에 쓰는 난수 함수. 테스트에서 고정 값을 넣기 위해 바꿀 수 있다. */
  random?: () => number
}

export function LivePipeline({ random = Math.random }: LivePipelineProps) {
  const { content } = useLocale()
  const strings = content.ui.pipeline
  const model = useMemo(() => loadTinyModel(), [])
  const [edited, setEdited] = useState<string | null>(null)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [temperature, setTemperature] = useState(1)
  const [selected, setSelected] = useState<number | null>(null)
  const inputId = useId()
  const layerId = useId()
  const headId = useId()
  const sliderId = useId()

  const text = edited ?? strings.defaultText
  const ids = useMemo(() => model.encode(text), [model, text])
  const result = useMemo(() => (ids.length > 0 ? model.forward(ids) : null), [model, ids])
  const distribution = useMemo(
    () => (result ? model.nextTokenDistribution(result.logits, temperature) : []),
    [model, result, temperature],
  )

  const { dModel, nLayers, nHeads, context } = model.config
  const charCount = Array.from(text).length
  const layerIndex = Math.min(layer, nLayers - 1)
  const headIndex = Math.min(head, nHeads - 1)

  const append = (steps: number) => {
    setEdited(model.generate(text, steps, temperature, random))
    setSelected(null)
  }

  const displayTokens = result ? result.tokens.map(displayChar) : []
  const selectedIndex = result
    ? Math.min(selected ?? result.tokens.length - 1, result.tokens.length - 1)
    : 0
  const maxAbs = result
    ? Math.max(1e-6, ...result.embeddings.map((row) => Math.max(...Array.from(row, Math.abs))))
    : 1
  const attentionRow = result ? result.attentions[layerIndex][headIndex][selectedIndex] : []
  const strongest = attentionRow.indexOf(Math.max(...attentionRow))

  return (
    <div className="demo live-pipeline" data-testid="live-pipeline">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <label htmlFor={inputId} className="pipeline-input-label">
        {strings.inputLabel}
      </label>
      <textarea
        id={inputId}
        className="demo-input"
        value={text}
        maxLength={context}
        rows={2}
        spellCheck={false}
        onChange={(event) => {
          setEdited(event.target.value)
          setSelected(null)
        }}
      />
      <p className="demo-stat" aria-live="polite">
        {strings.charCount(charCount, context)}
      </p>

      {!result ? (
        <p className="demo-regime">{strings.emptyInput}</p>
      ) : (
        <>
          <section className="pipeline-stage" aria-label={strings.stageTokens}>
            <h4>{strings.stageTokens}</h4>
            <div className="token-strip" role="group" aria-label={strings.tokensLegend}>
              {result.tokens.map((token, index) => (
                <button
                  key={index}
                  type="button"
                  className={`token-chip${index === selectedIndex ? ' is-active' : ''}${
                    result.ids[index] === 0 ? ' is-unknown' : ''
                  }`}
                  aria-pressed={index === selectedIndex}
                  onClick={() => setSelected(index)}
                >
                  <span className="token-chip-char">{displayChar(token)}</span>
                  <span className="token-chip-id">{result.ids[index]}</span>
                </button>
              ))}
            </div>
            <p className="demo-note">{strings.unknownNote}</p>
          </section>

          <section className="pipeline-stage" aria-label={strings.stageEmbeddings}>
            <h4>{strings.stageEmbeddings}</h4>
            <div
              className="emb-grid"
              aria-hidden="true"
              style={{ gridTemplateColumns: `repeat(${dModel}, 1fr)` }}
            >
              {result.embeddings.flatMap((row, i) =>
                Array.from(row, (value, j) => (
                  <span
                    key={`${i}-${j}`}
                    className={`emb-cell ${bucketClass(value / maxAbs)}${i === selectedIndex ? ' is-selected' : ''}`}
                  />
                )),
              )}
            </div>
            <p className="demo-note">{strings.embeddingNote(dModel)}</p>
            <p className="vector-print">
              <strong>{strings.vectorHeading(displayTokens[selectedIndex])}</strong> [
              {Array.from(result.embeddings[selectedIndex], (value) => value.toFixed(2)).join(', ')}
              ]
            </p>
          </section>

          <section className="pipeline-stage" aria-label={strings.stageAttention}>
            <h4>{strings.stageAttention}</h4>
            <div className="selects-row">
              <label htmlFor={layerId}>{strings.layerLabel}</label>
              <select
                id={layerId}
                value={layerIndex}
                onChange={(event) => setLayer(Number(event.target.value))}
              >
                {Array.from({ length: nLayers }, (_, i) => (
                  <option key={i} value={i}>
                    {strings.layerOption(i)}
                  </option>
                ))}
              </select>
              <label htmlFor={headId}>{strings.headLabel}</label>
              <select
                id={headId}
                value={headIndex}
                onChange={(event) => setHead(Number(event.target.value))}
              >
                {Array.from({ length: nHeads }, (_, i) => (
                  <option key={i} value={i}>
                    {strings.headOption(i)}
                  </option>
                ))}
              </select>
            </div>
            <HeatmapTable
              tokens={displayTokens}
              matrix={result.attentions[layerIndex][headIndex]}
              caption={strings.attentionCaption(layerIndex, headIndex)}
              selectedRow={selectedIndex}
              onSelectRow={setSelected}
              cellTitle={strings.cellTitle}
            />
            <p className="demo-regime" aria-live="polite">
              {strings.strongest(
                displayTokens[selectedIndex],
                displayTokens[strongest],
                attentionRow[strongest].toFixed(2),
              )}
            </p>
          </section>

          <section className="pipeline-stage" aria-label={strings.stageOutput}>
            <h4>{strings.stageOutput}</h4>
            <div className="slider-row">
              <label htmlFor={sliderId}>{strings.temperatureLabel}</label>
              <input
                id={sliderId}
                type="range"
                min={MIN_TEMPERATURE}
                max={MAX_TEMPERATURE}
                step={0.1}
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
              <output htmlFor={sliderId} className="slider-value">
                {temperature.toFixed(1)}
              </output>
            </div>
            <table className="prob-table">
              <caption className="visually-hidden">{strings.probabilityCaption}</caption>
              <thead>
                <tr>
                  <th scope="col">{strings.columns.token}</th>
                  <th scope="col">{strings.columns.probability}</th>
                </tr>
              </thead>
              <tbody>
                {distribution.slice(0, TOP_K).map((candidate) => (
                  <tr key={candidate.id}>
                    <th scope="row">
                      <span className="token token-word">{displayChar(candidate.token)}</span>
                    </th>
                    <td>
                      <div className="prob-cell">
                        <div className="prob-track" aria-hidden="true">
                          <div
                            className="prob-bar"
                            style={{ width: `${candidate.probability * 100}%` }}
                          />
                        </div>
                        <span className="prob-value">
                          {(candidate.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="demo-actions">
              <button type="button" className="button-primary" onClick={() => append(1)}>
                {strings.appendOne}
              </button>
              <button type="button" onClick={() => append(10)}>
                {strings.appendMany}
              </button>
              <button type="button" onClick={() => setEdited(null)} disabled={edited === null}>
                {strings.reset}
              </button>
            </div>
          </section>
        </>
      )}

      <p className="demo-note">
        {strings.modelNote(model.training.paramCount, nLayers, nHeads, model.training.corpusChars)}
      </p>
    </div>
  )
}
