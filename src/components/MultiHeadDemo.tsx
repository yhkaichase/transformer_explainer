import { useId, useState } from 'react'
import {
  attentionData,
  getAttentionDataset,
  weightBin,
  type AttentionDataset,
} from '../lib/attentionData'
import { useLocale } from '../state/locale-context'
import { DataEmptyState } from './DataEmptyState'
import { HeatmapTable } from './HeatmapTable'

interface MultiHeadDemoProps {
  /** 테스트용. 생략하면 현재 언어의 데이터 파일을 쓴다. */
  dataset?: AttentionDataset | null
}

export function MultiHeadDemo({ dataset }: MultiHeadDemoProps) {
  const { locale, content } = useLocale()
  const strings = content.ui.multihead
  const shared = content.ui.heatmap
  const data = dataset === undefined ? getAttentionDataset(locale) : dataset
  const [exampleIndex, setExampleIndex] = useState(0)
  const [layer, setLayer] = useState<number | null>(null)
  const [head, setHead] = useState(0)
  const exampleGroup = useId()
  const layerId = useId()

  if (!data || data.examples.length === 0) {
    return (
      <DataEmptyState
        heading={strings.heading}
        title={shared.emptyTitle}
        body={shared.emptyBody}
        errorsTitle={shared.errorsTitle}
        errors={attentionData.errors}
        testId="multihead-empty"
      />
    )
  }

  const example = data.examples[Math.min(exampleIndex, data.examples.length - 1)]
  const layerIndex = Math.min(layer ?? Math.floor(example.layers / 2), example.layers - 1)
  const headIndex = Math.min(head, example.heads - 1)
  const heads = example.attention[layerIndex]
  const n = example.tokens.length

  return (
    <div className="demo" data-testid="multihead-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <fieldset className="choice-group">
        <legend>{shared.exampleLegend}</legend>
        {data.examples.map((item, index) => (
          <label
            key={item.id}
            className={`choice-chip${index === exampleIndex ? ' is-active' : ''}`}
          >
            <input
              type="radio"
              name={exampleGroup}
              value={index}
              checked={index === exampleIndex}
              onChange={() => setExampleIndex(index)}
            />
            {item.text}
          </label>
        ))}
      </fieldset>

      <div className="selects-row">
        <label htmlFor={layerId}>{shared.layerLabel}</label>
        <select
          id={layerId}
          value={layerIndex}
          onChange={(event) => setLayer(Number(event.target.value))}
        >
          {Array.from({ length: example.layers }, (_, i) => (
            <option key={i} value={i}>
              {shared.layerOption(i)}
            </option>
          ))}
        </select>
      </div>

      <div className="head-thumbs" role="group" aria-label={strings.thumbsLabel}>
        {heads.map((matrix, h) => (
          <button
            key={h}
            type="button"
            className={`head-thumb${h === headIndex ? ' is-active' : ''}`}
            aria-pressed={h === headIndex}
            onClick={() => setHead(h)}
          >
            <span className="head-thumb-title">{shared.headOption(h)}</span>
            <span
              className="mini-heatmap"
              aria-hidden="true"
              style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}
            >
              {matrix.flatMap((row, q) =>
                row.map((weight, k) => (
                  <span
                    key={`${q}-${k}`}
                    className={k > q && weight === 0 ? 'attn-masked' : `attn-w${weightBin(weight)}`}
                  />
                )),
              )}
            </span>
          </button>
        ))}
      </div>

      <h4 className="demo-subheading">{shared.headOption(headIndex)}</h4>
      <HeatmapTable
        tokens={example.tokens}
        matrix={heads[headIndex]}
        caption={shared.tableCaption(layerIndex, headIndex)}
        cellTitle={shared.cellTitle}
      />

      <p className="demo-note">{shared.modelNote(data.model, data.generatedAt)}</p>
    </div>
  )
}
