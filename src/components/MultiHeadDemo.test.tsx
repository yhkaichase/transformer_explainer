import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { LocaleProvider } from '../state/LocaleProvider'
import { attentionFixture } from '../test/fixtures/attention-fixture'
import { MultiHeadDemo } from './MultiHeadDemo'

const ko = getContent('ko').ui

function renderWith(dataset: Parameters<typeof MultiHeadDemo>[0]['dataset']) {
  return render(
    <LocaleProvider>
      <MultiHeadDemo dataset={dataset} />
    </LocaleProvider>,
  )
}

describe('MultiHeadDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('데이터가 없으면 안내 상태를 보여 준다', () => {
    renderWith(null)
    expect(screen.getByTestId('multihead-empty')).toBeInTheDocument()
    expect(screen.getByText(ko.heatmap.emptyTitle)).toBeInTheDocument()
  })

  it('헤드 수만큼 작은 그림이 있고, 고른 헤드의 표를 크게 보여 준다', async () => {
    const user = userEvent.setup()
    renderWith(attentionFixture())
    const thumbs = within(
      screen.getByRole('group', { name: ko.multihead.thumbsLabel }),
    ).getAllByRole('button')
    expect(thumbs).toHaveLength(2)
    expect(thumbs[0]).toHaveAttribute('aria-pressed', 'true')

    await user.click(thumbs[1])
    expect(thumbs[1]).toHaveAttribute('aria-pressed', 'true')
    // 가운데 층(인덱스 1) 헤드 1 의 마지막 행 = [0.4, 0.4, 0.2]
    const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
    expect(
      within(rows[2])
        .getAllByRole('cell')
        .map((c) => c.textContent),
    ).toEqual(['0.40', '0.40', '0.20'])
    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(ko.heatmap.headOption(1))
  })
})
