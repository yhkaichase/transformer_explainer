import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { LocaleProvider } from '../state/LocaleProvider'
import { TemperatureDemo } from './TemperatureDemo'

const ko = getContent('ko').ui.temperature

function renderDemo(random?: () => number) {
  return render(
    <LocaleProvider>
      <TemperatureDemo random={random} />
    </LocaleProvider>,
  )
}

/** 표에서 확률 열의 숫자를 순서대로 읽는다. */
function readProbabilities(): number[] {
  const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
  return rows.map((row) => {
    const text = within(row).getByText(/%$/).textContent ?? '0'
    return Number(text.replace('%', ''))
  })
}

describe('TemperatureDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('후보 단어를 점수 순서로 보여 주고 확률의 합은 100% 에 가깝다', () => {
    renderDemo()
    const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(ko.candidates.length)
    expect(within(rows[0]).getByRole('rowheader')).toHaveTextContent(ko.candidates[0].token)
    const probabilities = readProbabilities()
    const total = probabilities.reduce((a, b) => a + b, 0)
    expect(total).toBeGreaterThan(99)
    expect(total).toBeLessThan(101)
    expect(probabilities[0]).toBeGreaterThan(probabilities[1])
  })

  it('온도를 낮추면 1등 확률이 커지고 높이면 작아진다', () => {
    renderDemo()
    const slider = screen.getByRole('slider', { name: ko.sliderLabel })
    const atOne = readProbabilities()[0]

    fireEvent.change(slider, { target: { value: '0.2' } })
    expect(readProbabilities()[0]).toBeGreaterThan(atOne)
    expect(screen.getByText(ko.regimes.low)).toBeInTheDocument()

    fireEvent.change(slider, { target: { value: '3' } })
    expect(readProbabilities()[0]).toBeLessThan(atOne)
    expect(screen.getByText(ko.regimes.high)).toBeInTheDocument()
  })

  it('뽑기 버튼은 확률에 따라 단어를 뽑고 결과를 표시한다', async () => {
    const user = userEvent.setup()
    renderDemo(() => 0)
    await user.click(screen.getByRole('button', { name: ko.draw }))
    const results = screen.getByTestId('draw-results')
    expect(within(results).getAllByText(ko.candidates[0].token)).toHaveLength(1)

    await user.click(screen.getByRole('button', { name: ko.drawMany }))
    expect(within(results).getAllByText(ko.candidates[0].token)).toHaveLength(10)
  })

  it('난수가 1 에 가까우면 확률이 가장 낮은 단어가 뽑힌다', async () => {
    const user = userEvent.setup()
    renderDemo(() => 0.9999)
    await user.click(screen.getByRole('button', { name: ko.draw }))
    const last = ko.candidates[ko.candidates.length - 1].token
    expect(within(screen.getByTestId('draw-results')).getByText(last)).toBeInTheDocument()
  })

  it('온도를 바꾸면 이전 뽑기 결과가 지워진다', async () => {
    const user = userEvent.setup()
    renderDemo(() => 0)
    await user.click(screen.getByRole('button', { name: ko.draw }))
    fireEvent.change(screen.getByRole('slider', { name: ko.sliderLabel }), {
      target: { value: '0.5' },
    })
    expect(screen.getByTestId('draw-results')).toBeEmptyDOMElement()
  })
})
