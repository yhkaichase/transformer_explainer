import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { feedForward } from '../lib/math'
import { LocaleProvider } from '../state/LocaleProvider'
import { FeedForwardDemo } from './FeedForwardDemo'

const ko = getContent('ko').ui.ffn
const vec = (values: number[]) => `[${values.map((v) => v.toFixed(2)).join(', ')}]`

function renderDemo() {
  return render(
    <LocaleProvider>
      <FeedForwardDemo />
    </LocaleProvider>,
  )
}

describe('FeedForwardDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('선택한 토큰의 입력, 중간값, ReLU 결과, 출력을 실제 계산값으로 보여 준다', () => {
    renderDemo()
    const result = feedForward(ko.inputs[0], ko.w1, ko.b1, ko.w2, ko.b2)
    expect(screen.getByText(vec(ko.inputs[0]))).toBeInTheDocument()
    expect(screen.getByText(vec(result.preActivation))).toBeInTheDocument()
    expect(screen.getByText(vec(result.hidden))).toBeInTheDocument()
    expect(screen.getByText(vec(result.output))).toBeInTheDocument()
    // 예시 가중치는 첫 토큰에서 숨은 값 두 개가 0 이 되도록 정해져 있다.
    expect(result.hidden.filter((v) => v === 0)).toHaveLength(2)
    expect(screen.getByText(ko.zeroNote)).toBeInTheDocument()
  })

  it('토큰을 바꾸면 그 토큰의 계산으로 바뀐다', async () => {
    const user = userEvent.setup()
    renderDemo()
    await user.click(screen.getByRole('radio', { name: ko.tokens[2] }))
    const result = feedForward(ko.inputs[2], ko.w1, ko.b1, ko.w2, ko.b2)
    expect(screen.getByText(vec(ko.inputs[2]))).toBeInTheDocument()
    expect(screen.getByText(vec(result.output))).toBeInTheDocument()
  })
})
