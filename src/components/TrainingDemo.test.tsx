import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { crossEntropy, softmax } from '../lib/math'
import { LocaleProvider } from '../state/LocaleProvider'
import { TrainingDemo } from './TrainingDemo'

const ko = getContent('ko').ui.training

function renderDemo() {
  return render(
    <LocaleProvider>
      <TrainingDemo />
    </LocaleProvider>,
  )
}

function readLoss(): number {
  const stat = screen.getByText(ko.lossLabel).closest('.stat')
  if (!stat) throw new Error('손실 타일이 없습니다')
  return Number(within(stat as HTMLElement).getByRole('definition').textContent)
}

describe('TrainingDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('처음에는 정답이 1등이 아니고 손실은 실제 교차 엔트로피 값이다', () => {
    renderDemo()
    const answerIndex = ko.candidates.findIndex((c) => c.token === ko.answer)
    const probabilities = softmax(ko.candidates.map((c) => c.score))
    expect(answerIndex).toBeGreaterThan(0)
    expect(readLoss()).toBeCloseTo(crossEntropy(probabilities, answerIndex), 2)
    const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
    expect(within(rows[answerIndex]).getByText(ko.correctMark)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: ko.reset })).toBeDisabled()
  })

  it('학습할수록 손실이 내려가고 정답 확률이 오르며, 처음으로 되돌릴 수 있다', async () => {
    const user = userEvent.setup()
    renderDemo()
    const initialLoss = readLoss()

    await user.click(screen.getByRole('button', { name: ko.stepOnce }))
    const afterOne = readLoss()
    expect(afterOne).toBeLessThan(initialLoss)

    await user.click(screen.getByRole('button', { name: ko.stepMany }))
    expect(readLoss()).toBeLessThan(afterOne)
    const stepsStat = screen.getByText(ko.stepsLabel).closest('.stat') as HTMLElement
    expect(within(stepsStat).getByRole('definition')).toHaveTextContent('11')

    await user.click(screen.getByRole('button', { name: ko.reset }))
    expect(readLoss()).toBeCloseTo(initialLoss, 5)
    expect(within(stepsStat).getByRole('definition')).toHaveTextContent('0')
  })
})
