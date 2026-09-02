import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { positionalEncoding } from '../lib/math'
import { LocaleProvider } from '../state/LocaleProvider'
import { DEMO_D_MODEL, DEMO_POSITIONS, PositionalEncodingDemo } from './PositionalEncodingDemo'

const ko = getContent('ko').ui.positional

function renderDemo() {
  return render(
    <LocaleProvider>
      <PositionalEncodingDemo />
    </LocaleProvider>,
  )
}

describe('PositionalEncodingDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('위치 수만큼 행이 있고 각 행에 차원 수만큼 값이 있다', () => {
    renderDemo()
    const table = screen.getByRole('table', { name: ko.tableCaption })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(DEMO_POSITIONS)
    expect(within(rows[0]).getAllByRole('cell')).toHaveLength(DEMO_D_MODEL)
    // 위치 0 은 [0, 1, 0, 1, ...]
    const values = within(rows[0])
      .getAllByRole('cell')
      .map((cell) => cell.textContent)
    expect(values.slice(0, 4)).toEqual(['0.00', '1.00', '0.00', '1.00'])
  })

  it('행을 누르면 그 위치의 숫자들을 보여 준다', async () => {
    const user = userEvent.setup()
    renderDemo()
    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(ko.selectedHeading(0))

    await user.click(screen.getByRole('button', { name: ko.rowLabel(3) }))

    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(ko.selectedHeading(3))
    expect(screen.getByRole('button', { name: ko.rowLabel(3) })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    const expected = `[${positionalEncoding(3, DEMO_D_MODEL)
      .map((v) => v.toFixed(2))
      .join(', ')}]`
    expect(screen.getByText(expected)).toBeInTheDocument()
  })
})
