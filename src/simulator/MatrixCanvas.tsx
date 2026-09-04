import { useEffect, useRef, useState, type MouseEvent } from 'react'

type ColorMode = 'diverging' | 'sequential'

interface MatrixCanvasProps {
  rows: number
  cols: number
  value: (row: number, col: number) => number
  /** diverging: 값 ÷ scale 을 -1..1 로 본다. sequential: 0..scale */
  scale: number
  mode: ColorMode
  cellWidth: number
  cellHeight: number
  gap?: number
  /** 테두리로 강조할 행 */
  highlightRow?: number | null
  /** 이 번호보다 작은 행을 흐리게 (decode 단계의 K·V 캐시 표시) */
  dimRowsBefore?: number | null
  /** 이 간격마다 세로 구분선 (헤드 경계) */
  separatorEvery?: number
  /** 0 인 칸을 회색으로 (ReLU 로 잘린 자리) */
  markZeros?: boolean
  label: string
  tooltip?: (row: number, col: number, value: number) => string
  onSelectRow?: (row: number) => void
  className?: string
}

type Rgb = [number, number, number]

function readColor(styles: CSSStyleDeclaration, name: string, fallback: string): Rgb {
  const raw = styles.getPropertyValue(name).trim() || fallback
  const hex = raw.startsWith('#') ? raw.slice(1) : fallback.slice(1)
  const full =
    hex.length === 3
      ? hex
          .split('')
          .map((c) => c + c)
          .join('')
      : hex
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ]
}

function mix(a: Rgb, b: Rgb, t: number): string {
  const k = Math.max(0, Math.min(1, t))
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * k)}, ${Math.round(a[1] + (b[1] - a[1]) * k)}, ${Math.round(a[2] + (b[2] - a[2]) * k)})`
}

/** 행렬을 캔버스 히트맵으로 그린다. 값은 마우스를 올리면 툴팁으로 읽을 수 있다. */
export function MatrixCanvas({
  rows,
  cols,
  value,
  scale,
  mode,
  cellWidth,
  cellHeight,
  gap = 0,
  highlightRow = null,
  dimRowsBefore = null,
  separatorEvery,
  markZeros = false,
  label,
  tooltip,
  onSelectRow,
  className,
}: MatrixCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ x: number; y: number; row: number; col: number } | null>(
    null,
  )
  const width = cols * (cellWidth + gap)
  const height = rows * (cellHeight + gap)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const draw = () => {
      const context = canvas.getContext('2d')
      if (!context) return
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.max(1, Math.round(width * dpr))
      canvas.height = Math.max(1, Math.round(height * dpr))
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      context.clearRect(0, 0, width, height)
      const styles = getComputedStyle(document.documentElement)
      const negative = readColor(styles, '--div-n3', '#1c5cab')
      const middle = readColor(styles, '--div-0', '#f0efec')
      const positive = readColor(styles, '--div-p3', '#c62f2e')
      const seqLow = readColor(styles, '--seq-0', '#edf3fc')
      const seqHigh = readColor(styles, '--seq-4', '#104281')
      const zero = readColor(styles, '--border', '#e2e2dc')
      const surface = readColor(styles, '--surface', '#ffffff')
      const accent = styles.getPropertyValue('--accent').trim() || '#1d5fc2'
      const safeScale = scale > 0 ? scale : 1

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = value(r, c)
          if (markZeros && v === 0) {
            context.fillStyle = mix(zero, zero, 0)
          } else if (mode === 'diverging') {
            const t = Math.max(-1, Math.min(1, v / safeScale))
            context.fillStyle = t < 0 ? mix(middle, negative, -t) : mix(middle, positive, t)
          } else {
            context.fillStyle = mix(seqLow, seqHigh, v / safeScale)
          }
          context.fillRect(c * (cellWidth + gap), r * (cellHeight + gap), cellWidth, cellHeight)
        }
      }
      if (dimRowsBefore !== null && dimRowsBefore > 0) {
        context.fillStyle = `rgba(${surface[0]}, ${surface[1]}, ${surface[2]}, 0.62)`
        context.fillRect(0, 0, width, Math.min(rows, dimRowsBefore) * (cellHeight + gap))
      }
      if (separatorEvery && separatorEvery > 0 && separatorEvery < cols) {
        context.strokeStyle = `rgb(${surface[0]}, ${surface[1]}, ${surface[2]})`
        context.lineWidth = 1.5
        for (let c = separatorEvery; c < cols; c += separatorEvery) {
          const x = c * (cellWidth + gap) - gap / 2
          context.beginPath()
          context.moveTo(x, 0)
          context.lineTo(x, height)
          context.stroke()
        }
      }
      if (highlightRow !== null && highlightRow >= 0 && highlightRow < rows) {
        context.strokeStyle = accent
        context.lineWidth = 2
        context.strokeRect(
          1,
          highlightRow * (cellHeight + gap) + 1,
          width - 2,
          cellHeight - 2 + gap,
        )
      }
    }
    draw()
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', draw)
    return () => media.removeEventListener('change', draw)
  }, [
    rows,
    cols,
    value,
    scale,
    mode,
    cellWidth,
    cellHeight,
    gap,
    highlightRow,
    dimRowsBefore,
    separatorEvery,
    markZeros,
    width,
    height,
  ])

  const locate = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const col = Math.floor(x / (cellWidth + gap))
    const row = Math.floor(y / (cellHeight + gap))
    if (row < 0 || row >= rows || col < 0 || col >= cols) return null
    return { x, y, row, col }
  }

  return (
    <div className={`mc-wrap${className ? ` ${className}` : ''}`}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        style={{ width, height, cursor: onSelectRow ? 'pointer' : 'default' }}
        onMouseMove={(event) => setHover(locate(event))}
        onMouseLeave={() => setHover(null)}
        onClick={(event) => {
          const at = locate(event)
          if (at && onSelectRow) onSelectRow(at.row)
        }}
      />
      {hover && tooltip && (
        <div className="mc-tip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          {tooltip(hover.row, hover.col, value(hover.row, hover.col))}
        </div>
      )}
    </div>
  )
}
