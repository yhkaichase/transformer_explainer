import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { scaledDotProductAttention } from '../lib/math'
import { AudienceProvider } from '../state/AudienceProvider'
import { AUDIENCE_STORAGE_KEY } from '../state/AudienceProvider'
import { LocaleProvider } from '../state/LocaleProvider'
import { AttentionDemo } from './AttentionDemo'

const ko = getContent('ko').ui.attention

function renderDemo() {
  return render(
    <LocaleProvider>
      <AudienceProvider>
        <AttentionDemo />
      </AudienceProvider>
    </LocaleProvider>,
  )
}

function weightRows() {
  const table = screen.getByRole('table', { name: /가중치/ })
  return within(table).getAllByRole('row').slice(1)
}

describe('AttentionDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('기본 질의 토큰의 가중치를 실제 계산값으로 보여 준다', () => {
    renderDemo()
    const { weights, output } = scaledDotProductAttention(ko.q, ko.k, ko.v)
    const rows = weightRows()
    expect(rows).toHaveLength(ko.tokens.length)
    rows.forEach((row, index) => {
      expect(within(row).getByText(`${(weights[0][index] * 100).toFixed(1)}%`)).toBeInTheDocument()
    })
    // 예시 벡터는 "은행" 이 "돈" 을 가장 많이 보도록 정해져 있다.
    const best = weights[0].indexOf(Math.max(...weights[0]))
    expect(ko.tokens[best]).toBe('돈')
    const vector = `[${output[0].map((v) => v.toFixed(2)).join(', ')}]`
    const outputLine = screen.getByText(ko.outputHeading('은행')).closest('p')
    expect(outputLine).toHaveTextContent(vector)
  })

  it('질의 토큰을 바꾸면 가중치가 그 토큰 기준으로 바뀐다', async () => {
    const user = userEvent.setup()
    renderDemo()
    await user.click(screen.getByRole('radio', { name: '돈' }))
    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(ko.weightsHeading('돈'))
    const { weights } = scaledDotProductAttention(ko.q, ko.k, ko.v)
    const rows = weightRows()
    expect(within(rows[0]).getByText(`${(weights[2][0] * 100).toFixed(1)}%`)).toBeInTheDocument()
  })

  it('인과 마스크를 켜면 뒤에 오는 토큰이 가려지고 안내문이 나온다', async () => {
    const user = userEvent.setup()
    renderDemo()
    await user.click(screen.getByRole('checkbox'))
    const rows = weightRows()
    expect(within(rows[0]).getByText('100.0%')).toBeInTheDocument()
    expect(within(rows[1]).getByText(ko.masked)).toBeInTheDocument()
    expect(within(rows[3]).getByText(ko.masked)).toBeInTheDocument()
    expect(screen.getByText(ko.maskedNote('은행'))).toBeInTheDocument()

    // 마지막 토큰은 모두 볼 수 있으므로 가려지는 것이 없다.
    await user.click(screen.getByRole('radio', { name: '찾았다' }))
    expect(screen.queryByText(ko.masked)).not.toBeInTheDocument()
    expect(screen.queryByText(ko.maskedNote('찾았다'))).not.toBeInTheDocument()
  })

  it('계산 과정은 임원용에서는 접혀 있고 엔지니어용에서는 펼쳐져 있다', () => {
    const first = renderDemo()
    const details = () => screen.getByText(ko.detailsSummary).closest('details')
    expect(details()).not.toHaveAttribute('open')
    first.unmount()

    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, 'engineer')
    renderDemo()
    expect(details()).toHaveAttribute('open')
    expect(screen.getByText(ko.steps.softmax)).toBeInTheDocument()
  })
})
