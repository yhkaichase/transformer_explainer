import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import App from './App'
import { getContent } from './content'
import { AUDIENCE_STORAGE_KEY } from './state/AudienceProvider'
import { LOCALE_STORAGE_KEY } from './state/LocaleProvider'

const ko = getContent('ko')
const en = getContent('en')

/** 제목을 정규식으로 쓸 때 특수문자를 이스케이프한다. */
const titlePattern = (title: string) => new RegExp(title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))

describe('App', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  afterEach(() => {
    window.history.replaceState({}, '', '/')
  })

  it('기본은 한국어이며 제목, 목차, 모든 섹션을 보여 준다', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(ko.ui.hero.title)
    const toc = screen.getByRole('navigation', { name: ko.ui.tocTitle })
    expect(within(toc).getAllByRole('link')).toHaveLength(ko.sections.length)
    for (const section of ko.sections) {
      expect(
        screen.getByRole('heading', { level: 2, name: titlePattern(section.title) }),
      ).toBeVisible()
    }
    expect(document.documentElement.lang).toBe('ko')
    expect(document.title).toBe(ko.ui.documentTitle)
  })

  it('기본은 임원용이며, 엔지니어용으로 바꾸면 심화 설명이 나타난다', async () => {
    const user = userEvent.setup()
    render(<App />)
    expect(screen.getByRole('radio', { name: /임원용/ })).toBeChecked()
    expect(screen.queryByText(ko.ui.engineerTitle)).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: /엔지니어용/ }))

    expect(screen.getByRole('radio', { name: /엔지니어용/ })).toBeChecked()
    expect(screen.getAllByText(ko.ui.engineerTitle)).toHaveLength(ko.sections.length)
  })

  it('선택한 설명 수준을 localStorage 에 기억하고 다시 불러온다', async () => {
    const user = userEvent.setup()
    const first = render(<App />)
    await user.click(screen.getByRole('radio', { name: /엔지니어용/ }))
    expect(window.localStorage.getItem(AUDIENCE_STORAGE_KEY)).toBe('engineer')
    first.unmount()

    render(<App />)
    expect(screen.getByRole('radio', { name: /엔지니어용/ })).toBeChecked()
  })

  it('English 로 바꾸면 본문, 토글, 문서 제목이 영어가 되고 URL 과 localStorage 에 반영된다', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('radio', { name: 'English' }))

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(en.ui.hero.title)
    expect(screen.getByRole('radio', { name: /Executive/ })).toBeChecked()
    for (const section of en.sections) {
      expect(
        screen.getByRole('heading', { level: 2, name: titlePattern(section.title) }),
      ).toBeVisible()
    }
    expect(document.documentElement.lang).toBe('en')
    expect(document.title).toBe(en.ui.documentTitle)
    expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('en')
    expect(new URLSearchParams(window.location.search).get('lang')).toBe('en')
  })

  it('?lang=en 링크로 열면 처음부터 영어로 보인다', () => {
    window.history.replaceState({}, '', '/?lang=en')
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(en.ui.hero.title)
    expect(screen.getByRole('radio', { name: 'English' })).toBeChecked()
  })

  it('저장된 언어를 다시 불러오고, URL 파라미터가 있으면 그것을 우선한다', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en')
    const first = render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(en.ui.hero.title)
    first.unmount()

    window.history.replaceState({}, '', '/?lang=ko')
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(ko.ui.hero.title)
  })
})
