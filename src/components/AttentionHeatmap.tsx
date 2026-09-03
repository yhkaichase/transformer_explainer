import { useId, useState } from 'react'
import {
  attentionData,
  displayToken,
  getAttentionDataset,
  type AttentionDataset,
} from '../lib/attentionData'
import { useLocale } from '../state/locale-context'
import { DataEmptyState } from './DataEmptyState'
import { HeatmapTable } from './HeatmapTable'

interface AttentionHeatmapProps {
  /** 테스트용. 생략하면 현재 언어의 데이터 파일을 쓴다. */
  dataset?: AttentionDataset | null
}

export function AttentionHeatmap({ dataset }: AttentionHeatmapProps) {
  const { locale, content } = useLocale()
  const strings = content.ui.heatmap
  const data = dataset === undefined ? getAttentionDataset(locale) : dataset
  const [exampleIndex, setExampleIndex] = useState(0)
  const [layer, setLayer] = useState<number | null>(null)
  const [head, setHead] = useState(0)
  const [row, setRow] = useState<number | null>(null)
  const exampleGroup = useId()
  const layerId = useId()
  const headId = useId()

  if (!data || data.examples.length === 0) {
    return (
      <DataEmptyState
        heading={strings.heading}
        title={strings.emptyTitle}
        body={strings.emptyBody}
        errorsTitle={strings.errorsTitle}
        errors={attentionData.errors}
        testId="attention-heatmap-empty"
      />
    )
  }

  const example = data.examples[Math.min(exampleIndex, data.examples.length - 1)]
  const layerIndex = Math.min(layer ?? Math.floor(example.layers / 2), example.layers - 1)
  const headIndex = Math.min(head, example.heads - 1)
  const matrix = example.attention[layerIndex][headIndex]
  const selectedRow = Math.min(row ?? example.tokens.length - 1, example.tokens.length - 1)
  const weights = matrix[selectedRow]
  const strongest = weights.indexOf(Math.max(...weights))

  return (
    <div className="demo" data-testid="attention-heatmap">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <fieldset className="choice-group">
        <legend>{strings.exampleLegend}</legend>
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
              onChange={() => {
                setExampleIndex(index)
                setRow(null)
              }}
            />
            {item.text}
          </label>
        ))}
      </fieldset>

      <div className="selects-row">
        <label htmlFor={layerId}>{strings.layerLabel}</label>
        <select
          id={layerId}
          value={layerIndex}
          onChange={(event) => setLayer(Number(event.target.value))}
        >
          {Array.from({ length: example.layers }, (_, i) => (
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
          {Array.from({ length: example.heads }, (_, i) => (
            <option key={i} value={i}>
              {strings.headOption(i)}
            </option>
          ))}
        </select>
      </div>

      <HeatmapTable
        tokens={example.tokens}
        matrix={matrix}
        caption={strings.tableCaption(layerIndex, headIndex)}
        selectedRow={selectedRow}
        onSelectRow={setRow}
        cellTitle={strings.cellTitle}
      />

      <p className="heatmap-legend">
        <span className="heatmap-legend-swatches" aria-hidden="true">
          {[0, 1, 2, 3, 4].map((bin) => (
            <span key={bin} className={`swatch attn-w${bin}`} />
          ))}
        </span>
        <span>{strings.legendLabel}</span>
      </p>

      <p className="demo-regime" aria-live="polite">
        {strings.strongest(
          displayToken(example.tokens[selectedRow]),
          displayToken(example.tokens[strongest]),
          weights[strongest].toFixed(2),
        )}
      </p>

      <p className="demo-note">{strings.modelNote(data.model, data.generatedAt)}</p>
    </div>
  )
}
