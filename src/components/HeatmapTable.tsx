import { displayToken, weightBin } from '../lib/attentionData'

interface HeatmapTableProps {
  tokens: string[]
  /** matrix[보는 토큰][보이는 토큰] */
  matrix: number[][]
  caption: string
  selectedRow?: number
  onSelectRow?: (index: number) => void
  /** 셀 안내문: (보는 토큰, 보이는 토큰, 가중치) */
  cellTitle: (from: string, to: string, weight: string) => string
}

/** 실제 어텐션 값을 표로 그린다. 값은 셀 색과 숨긴 텍스트 양쪽으로 제공한다. */
export function HeatmapTable({
  tokens,
  matrix,
  caption,
  selectedRow,
  onSelectRow,
  cellTitle,
}: HeatmapTableProps) {
  return (
    <div className="heatmap-wrap">
      <table className="attn-heatmap">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            <td />
            {tokens.map((token, k) => (
              <th key={k} scope="col">
                <span className="attn-col-label">{displayToken(token)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, q) => (
            <tr key={q} className={q === selectedRow ? 'is-selected' : undefined}>
              <th scope="row">
                {onSelectRow ? (
                  <button
                    type="button"
                    aria-pressed={q === selectedRow}
                    onClick={() => onSelectRow(q)}
                  >
                    {displayToken(tokens[q])}
                  </button>
                ) : (
                  displayToken(tokens[q])
                )}
              </th>
              {row.map((weight, k) => {
                const masked = k > q && weight === 0
                return (
                  <td
                    key={k}
                    className={`attn-cell ${masked ? 'attn-masked' : `attn-w${weightBin(weight)}`}`}
                    title={
                      masked
                        ? undefined
                        : cellTitle(
                            displayToken(tokens[q]),
                            displayToken(tokens[k]),
                            weight.toFixed(2),
                          )
                    }
                  >
                    <span className="visually-hidden">{masked ? '' : weight.toFixed(2)}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
