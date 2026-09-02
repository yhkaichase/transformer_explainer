import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import App from './App'
import { ENGINEER_CALLOUT_TITLE } from './components/SectionView'
import { sections } from './content/sections'
import { AUDIENCE_STORAGE_KEY } from './state/AudienceProvider'

describe('App', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('제목과 목차, 모든 섹션을 보여 준다', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('트랜스포머')
    const toc = screen.getByRole('navigation', { name: '목차' })
    expect(within(toc).getAllByRole('link')).toHaveLength(sections.length)
    for (const section of sections) {
      expect(
        screen.getByRole('heading', { level: 2, name: new RegExp(section.title) }),
      ).toBeVisible()
    }
  })

  it('기본은 임원용이며, 엔지니어용으로 바꾸면 심화 설명이 나타난다', async () => {
    const user = userEvent.setup()
    render(<App />)
    expect(screen.getByRole('radio', { name: /임원용/ })).toBeChecked()
    expect(screen.queryByText(ENGINEER_CALLOUT_TITLE)).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: /엔지니어용/ }))

    expect(screen.getByRole('radio', { name: /엔지니어용/ })).toBeChecked()
    expect(screen.getAllByText(ENGINEER_CALLOUT_TITLE)).toHaveLength(sections.length)
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
})
