import { useId, useState } from 'react'
import { useLocale } from '../state/locale-context'

export const MIN_LAYERS = 1
export const MAX_LAYERS = 12
export const DEFAULT_LAYERS = 6

/** 서브층 하나: 입력 → [계산] → ⊕ → 정규화. 위로 도는 선이 잔차 연결이다. */
function Sublayer({ label, norm }: { label: string; norm: string }) {
  return (
    <svg className="sublayer" viewBox="0 0 190 46" role="img" aria-label={`${label}, ${norm}`}>
      <path className="sublayer-skip" d="M8 32 C 8 4, 122 4, 122 32" />
      <line className="sublayer-line" x1="8" y1="32" x2="40" y2="32" />
      <rect className="sublayer-box" x="40" y="20" width="70" height="24" rx="5" />
      <text className="sublayer-text" x="75" y="36" textAnchor="middle">
        {label}
      </text>
      <line className="sublayer-line" x1="110" y1="32" x2="114" y2="32" />
      <circle className="sublayer-add" cx="122" cy="32" r="8" />
      <text className="sublayer-plus" x="122" y="36" textAnchor="middle">
        +
      </text>
      <line className="sublayer-line" x1="130" y1="32" x2="140" y2="32" />
      <rect className="sublayer-box sublayer-norm" x="140" y="20" width="44" height="24" rx="5" />
      <text className="sublayer-text" x="162" y="36" textAnchor="middle">
        {norm}
      </text>
    </svg>
  )
}

export function LayerStackDemo() {
  const { content } = useLocale()
  const strings = content.ui.stack
  const [layers, setLayers] = useState(DEFAULT_LAYERS)
  const sliderId = useId()

  return (
    <div className="demo" data-testid="stack-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <div className="slider-row">
        <label htmlFor={sliderId}>{strings.sliderLabel}</label>
        <input
          id={sliderId}
          type="range"
          min={MIN_LAYERS}
          max={MAX_LAYERS}
          step={1}
          value={layers}
          aria-valuetext={strings.layerLabel(layers)}
          onChange={(event) => setLayers(Number(event.target.value))}
        />
        <output htmlFor={sliderId} className="slider-value">
          {strings.layerLabel(layers)}
        </output>
      </div>
      <p className="demo-regime" aria-live="polite">
        {strings.layersCaption(layers)}
      </p>

      <div className="stack-diagram">
        <p className="stack-io">{strings.outputLabel}</p>
        <ol className="stack-layers" aria-label={strings.sliderLabel}>
          {Array.from({ length: layers }, (_, i) => layers - i).map((layerNumber) => (
            <li key={layerNumber} className="stack-layer">
              <span className="stack-layer-title">{strings.layerTitle(layerNumber)}</span>
              <div className="stack-sublayers">
                <Sublayer label={strings.attention} norm={strings.norm} />
                <Sublayer label={strings.ffn} norm={strings.norm} />
              </div>
            </li>
          ))}
        </ol>
        <p className="stack-io">{strings.inputLabel}</p>
      </div>
      <p className="demo-regime">{strings.residual}</p>

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
