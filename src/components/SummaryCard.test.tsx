import { render, screen, within } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { AUDIENCE_STORAGE_KEY, AudienceProvider } from '../state/AudienceProvider'
import { LocaleProvider } from '../state/LocaleProvider'
import { SummaryCard } from './SummaryCard'

const ko = getContent('ko').ui.summary

function renderCard() {
  return render(
    <LocaleProvider>
      <AudienceProvider>
        <SummaryCard />
      </AudienceProvider>
    </LocaleProvider>,
  )
}

describe('SummaryCard', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('임원용에서는 요약 단계와 핵심만 보이고 다음 단계는 숨긴다', () => {
    renderCard()
    const panel = within(screen.getByRole('region', { name: ko.cardTitle }))
    expect(panel.getAllByRole('listitem')).toHaveLength(ko.steps.length + ko.takeaways.length)
    expect(screen.queryByRole('region', { name: ko.nextStepsTitle })).not.toBeInTheDocument()
    expect(screen.getByText(ko.steps[0])).toBeInTheDocument()
    expect(screen.getByText(ko.takeaways[0].lead)).toBeInTheDocument()
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
  })

  it('엔지니어용에서는 확인된 출처 링크가 있는 다음 단계를 보여 준다', () => {
    window.localStorage.setItem(AUDIENCE_STORAGE_KEY, 'engineer')
    renderCard()
    const links = screen.getAllByRole('link')
    expect(links.map((a) => a.getAttribute('href'))).toEqual(ko.nextSteps.map((n) => n.url))
    expect(links[0]).toHaveTextContent(ko.nextSteps[0].label)
  })
})
