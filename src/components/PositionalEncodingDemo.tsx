import { useMemo, useState } from 'react'
import { positionalEncoding } from '../lib/math'
import { useLocale } from '../state/locale-context'

/** 설명용으로 줄인 크기. 원 논문은 d_model = 512. */
export const DEMO_D_MODEL = 16
export const DEMO_POSITIONS = 16

/** -1..1 값을 -3..3 의 일곱 단계로 나눈다 (파랑 3단계, 회색, 빨강 3단계). */
function bucketOf(value: number): number {
  return Math.round(value * 3)
}

function bucketClass(value: number): string {
  const bucket = bucketOf(value)
  if (bucket === 0) return 'cell-0'
  return bucket < 0 ? `cell-n${-bucket}` : `cell-p${bucket}`
}

const LEGEND_BUCKETS = [-3, -2, -1, 0, 1, 2, 3]

export function PositionalEncodingDemo() {
  const { content } = useLocale()
  const strings = content.ui.positional
  const [selected, setSelected] = useState(0)
  const rows = useMemo(
    () =>
      Array.from({ length: DEMO_POSITIONS }, (_, position) =>
        positionalEncoding(position, DEMO_D_MODEL),
      ),
    [],
  )
  const dimensions = Array.from({ length: DEMO_D_MODEL }, (_, d) => d)
  const selectedVector = rows[selected]

  return (
    <div className="demo" data-testid="positional-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>

      <div className="heatmap-wrap">
        <table className="heatmap">
          <caption className="visually-hidden">{strings.tableCaption}</caption>
          <thead>
            <tr>
              <td />
              {dimensions.map((d) => (
                <th key={d} scope="col" aria-label={strings.columnLabel(d)}>
                  {d}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((vector, position) => (
              <tr key={position} className={position === selected ? 'is-selected' : undefined}>
                <th scope="row">
                  <button
                    type="button"
                    aria-pressed={position === selected}
                    onClick={() => setSelected(position)}
                  >
                    {strings.rowLabel(position)}
                  </button>
                </th>
                {vector.map((value, d) => (
                  <td
                    key={d}
                    className={`cell ${bucketClass(value)}`}
                    title={`${strings.rowLabel(position)}, ${strings.columnLabel(d)}: ${value.toFixed(2)}`}
                  >
                    <span className="visually-hidden">{value.toFixed(2)}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="heatmap-legend">
        <span className="heatmap-legend-swatches" aria-hidden="true">
          {LEGEND_BUCKETS.map((bucket) => (
            <span key={bucket} className={`swatch ${bucketClass(bucket / 3)}`} />
          ))}
        </span>
        <span>{strings.legendLabel}</span>
      </p>

      <div className="selected-vector">
        <h4>{strings.selectedHeading(selected)}</h4>
        <div className="vector-bars" aria-hidden="true">
          {selectedVector.map((value, d) => (
            <div key={d} className="vector-slot">
              <div
                className={`vbar ${value >= 0 ? 'vbar-pos' : 'vbar-neg'}`}
                style={{ height: `${Math.abs(value) * 50}%` }}
              />
            </div>
          ))}
        </div>
        <p className="vector-print">
          [{selectedVector.map((value) => value.toFixed(2)).join(', ')}]
        </p>
      </div>

      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
