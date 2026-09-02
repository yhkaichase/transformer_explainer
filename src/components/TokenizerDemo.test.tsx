import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { LOCALE_STORAGE_KEY, LocaleProvider } from '../state/LocaleProvider'
import { TokenizerDemo } from './TokenizerDemo'

function renderDemo() {
  return render(
    <LocaleProvider>
      <TokenizerDemo />
    </LocaleProvider>,
  )
}

describe('TokenizerDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('기본 문장을 토큰으로 보여 준다', () => {
    renderDemo()
    const list = screen.getByRole('list', { name: '토큰 목록' })
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((li) => li.textContent),
    ).toEqual(['0은행에', '1가서', '2돈을', '3찾았다', '4.'])
    expect(screen.getByText('토큰 5개')).toBeInTheDocument()
  })

  it('문장을 바꾸면 토큰이 다시 계산된다', async () => {
    const user = userEvent.setup()
    renderDemo()
    const input = screen.getByLabelText('토큰으로 나눌 문장')
    await user.clear(input)
    await user.type(input, 'Hello, world')
    expect(screen.getByText('토큰 3개')).toBeInTheDocument()
  })

  it('영어에서는 영어 기본 문장과 영어 문구를 쓴다', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en')
    renderDemo()
    const list = screen.getByRole('list', { name: 'Token list' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(9)
    expect(screen.getByText('9 tokens')).toBeInTheDocument()
  })
})
