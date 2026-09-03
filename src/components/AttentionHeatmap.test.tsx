import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { LocaleProvider } from '../state/LocaleProvider'
import { attentionFixture } from '../test/fixtures/attention-fixture'
import { AttentionHeatmap } from './AttentionHeatmap'

const ko = getContent('ko').ui.heatmap

function renderWith(dataset: Parameters<typeof AttentionHeatmap>[0]['dataset']) {
  return render(
    <LocaleProvider>
      <AttentionHeatmap dataset={dataset} />
    </LocaleProvider>,
  )
}

/** 표 본문에서 (행, 열) 셀의 숨긴 값 텍스트를 읽는다. */
function cellText(row: number, column: number): string {
  const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
  return within(rows[row]).getAllByRole('cell')[column].textContent ?? ''
}

describe('AttentionHeatmap', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('데이터가 없으면 만드는 방법을 안내한다', () => {
    renderWith(null)
    expect(screen.getByTestId('attention-heatmap-empty')).toBeInTheDocument()
    expect(screen.getByText(ko.emptyTitle)).toBeInTheDocument()
    expect(screen.getByText(/precompute_attention\.py/)).toBeInTheDocument()
  })

  it('가운데 층과 첫 헤드의 값을 표로 보여 주고, 마지막 토큰 행이 선택되어 있다', () => {
    renderWith(attentionFixture())
    // 2층 중 가운데 = 인덱스 1, 헤드 0: 마지막 행 [0.1, 0.1, 0.8]
    expect(screen.getByRole('combobox', { name: ko.layerLabel })).toHaveValue('1')
    expect(cellText(2, 2)).toBe('0.80')
    expect(cellText(0, 1)).toBe('') // 인과 마스크로 가려진 자리
    expect(screen.getByText(ko.strongest('c', 'c', '0.80'))).toBeInTheDocument()
    expect(screen.getByText(/test-fixture/)).toBeInTheDocument()
  })

  it('층과 헤드를 바꾸고 행을 고르면 값과 안내가 바뀐다', async () => {
    const user = userEvent.setup()
    renderWith(attentionFixture())
    fireEvent.change(screen.getByRole('combobox', { name: ko.layerLabel }), {
      target: { value: '0' },
    })
    fireEvent.change(screen.getByRole('combobox', { name: ko.headLabel }), {
      target: { value: '1' },
    })
    // 0층 헤드 1: 행 2 = [0.6, 0.2, 0.2]
    expect(cellText(2, 0)).toBe('0.60')
    expect(screen.getByText(ko.strongest('c', 'a', '0.60'))).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'b' }))
    // 행 1 = [0.1, 0.9, 0]
    expect(screen.getByText(ko.strongest('b', 'b', '0.90'))).toBeInTheDocument()
  })
})
