import type { ReactNode } from 'react'
import { CELL, HEAT_CELL, maxAbs, ROW } from './format'
import { MatrixCanvas } from './MatrixCanvas'

/** 두 시뮬레이터 페이지가 함께 쓰는 작은 표시 부품. 단계 id 는 페이지마다 다르므로 문자열로 받는다. */

export interface StripProps {
  title: string
  data: Float32Array
  rows: number
  cols: number
  cell?: number
  highlightRow?: number | null
  dimRowsBefore?: number | null
  separatorEvery?: number
  markZeros?: boolean
  stages: string[]
  active: string
  onSelectRow?: (row: number) => void
  scale?: number
  /** 재사용(물려받은) 값임을 점선 테두리로 표시 */
  inherited?: boolean
}

/** 활성값 행렬(T × 열)을 띠 모양 히트맵으로 그린다. */
export function Strip({
  title,
  data,
  rows,
  cols,
  cell = CELL,
  highlightRow = null,
  dimRowsBefore = null,
  separatorEvery,
  markZeros,
  stages,
  active,
  onSelectRow,
  scale,
  inherited = false,
}: StripProps) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-strip${isActive ? ' is-active' : ''}${inherited ? ' is-inherited' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>{title}</figcaption>
      <MatrixCanvas
        rows={rows}
        cols={cols}
        value={(r, c) => data[r * cols + c]}
        scale={scale ?? maxAbs(data)}
        mode="diverging"
        cellWidth={cell}
        cellHeight={ROW}
        highlightRow={highlightRow}
        dimRowsBefore={dimRowsBefore}
        separatorEvery={separatorEvery}
        markZeros={markZeros}
        label={`${title} ${rows}×${cols}`}
        tooltip={(r, c, v) => `${title}[${r}, ${c}] = ${v.toFixed(3)}`}
        onSelectRow={onSelectRow}
      />
    </figure>
  )
}

export interface HeatProps {
  title: string
  data: Float32Array
  rows: number
  cols: number
  cell?: number
  stages: string[]
  active: string
  highlightRow?: number | null
}

/** 가중치 행렬 히트맵. */
export function Heat({
  title,
  data,
  rows,
  cols,
  cell = HEAT_CELL,
  stages,
  active,
  highlightRow = null,
}: HeatProps) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-heat${isActive ? ' is-active' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>
        {title}{' '}
        <span className="sim-shape">
          {rows}×{cols}
        </span>
      </figcaption>
      <MatrixCanvas
        rows={rows}
        cols={cols}
        value={(r, c) => data[r * cols + c]}
        scale={maxAbs(data)}
        mode="diverging"
        cellWidth={cell}
        cellHeight={cell}
        highlightRow={highlightRow}
        label={`${title} ${rows}×${cols}`}
        tooltip={(r, c, v) => `${title}[${r}, ${c}] = ${v.toFixed(4)}`}
      />
    </figure>
  )
}

export function Op({ children }: { children: ReactNode }) {
  return (
    <span className="sim-op" aria-hidden="true">
      {children}
    </span>
  )
}

export interface BarItem {
  key: string
  label: string
  value: number
  display: string
}

export function Bars({
  title,
  items,
  max,
  stages,
  active,
  highlightKey,
}: {
  title: string
  items: BarItem[]
  max: number
  stages: string[]
  active: string
  highlightKey?: string
}) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-bars${isActive ? ' is-active' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>{title}</figcaption>
      <ol>
        {items.map((item) => (
          <li key={item.key} className={item.key === highlightKey ? 'is-picked' : undefined}>
            <span className="sim-bar-label">{item.label}</span>
            <span className="sim-bar-track" aria-hidden="true">
              <span
                className="sim-bar-fill"
                style={{ width: `${(Math.max(0, item.value) / max) * 100}%` }}
              />
            </span>
            <span className="sim-bar-value">{item.display}</span>
          </li>
        ))}
      </ol>
    </figure>
  )
}
