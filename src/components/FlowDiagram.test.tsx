import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { LocaleProvider } from '../state/LocaleProvider'
import { FlowDiagram } from './FlowDiagram'

const ko = getContent('ko').ui.flow

function renderDiagram() {
  return render(
    <LocaleProvider>
      <FlowDiagram />
    </LocaleProvider>,
  )
}

describe('FlowDiagram', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('모든 단계를 순서대로 보여 주고 첫 단계가 선택되어 있다', () => {
    renderDiagram()
    const list = screen.getByRole('list', { name: ko.heading })
    const buttons = within(list).getAllByRole('button')
    expect(buttons.map((b) => b.textContent)).toEqual(
      ko.stages.map((stage, i) => `${i + 1}${stage.label}`),
    )
    expect(buttons[0]).toHaveAttribute('aria-current', 'step')
    expect(screen.getByText(ko.stepLabel(1, ko.stages.length))).toBeInTheDocument()
    expect(screen.getByText(ko.stages[0].detail)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: ko.goToSection })).toHaveAttribute(
      'href',
      `#${ko.stages[0].section}`,
    )
  })

  it('다음/이전 버튼과 단계 버튼으로 이동한다', async () => {
    const user = userEvent.setup()
    renderDiagram()
    const prev = screen.getByRole('button', { name: ko.prev })
    const next = screen.getByRole('button', { name: ko.next })
    expect(prev).toBeDisabled()

    await user.click(next)
    expect(screen.getByText(ko.stepLabel(2, ko.stages.length))).toBeInTheDocument()
    expect(screen.getByText(ko.stages[1].detail)).toBeInTheDocument()
    expect(prev).toBeEnabled()

    const last = ko.stages.length - 1
    await user.click(screen.getByRole('button', { name: `${last + 1}${ko.stages[last].label}` }))
    expect(screen.getByText(ko.stages[last].detail)).toBeInTheDocument()
    expect(next).toBeDisabled()
    expect(screen.getByRole('link', { name: ko.goToSection })).toHaveAttribute(
      'href',
      `#${ko.stages[last].section}`,
    )
  })
})
