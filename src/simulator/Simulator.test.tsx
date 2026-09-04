import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { loadTinyModel } from '../lib/tinyTransformer'
import { LABELS } from './labels'
import { Simulator } from './Simulator'
import { STAGES } from './stages'

const L = LABELS.ko
const model = loadTinyModel()

beforeAll(() => {
  // jsdom 에는 캔버스 그리기와 matchMedia 가 없으므로 최소한의 대체를 넣는다.
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => null,
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia
})

describe('Simulator', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/simulator.html')
  })

  it('프롬프트 토큰과 모든 단계를 보여 주고 prefill 상태로 시작한다', () => {
    render(<Simulator initialLang="ko" />)
    const tokens = within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')
    expect(tokens).toHaveLength(model.encode('은행에 가서 돈을').length)
    expect(screen.getByText(L.prefill)).toBeInTheDocument()
    const stageButtons = screen.getAllByRole('button', { current: 'step' })
    expect(stageButtons).toHaveLength(1)
    expect(stageButtons[0]).toHaveTextContent(L.stageNames[STAGES[0]])
    expect(screen.getByText(L.stageNames.qkv, { selector: 'ol button' })).toBeInTheDocument()
  })

  it('다음 단계 버튼과 단계 칩으로 수식이 바뀐다', async () => {
    const user = userEvent.setup()
    render(<Simulator initialLang="ko" />)
    await user.click(screen.getByRole('button', { name: L.next }))
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(L.stageNames.embed)
    await user.click(screen.getByRole('button', { name: new RegExp(L.stageNames.scores) }))
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(L.stageNames.scores)
    // 초점 토큰의 점수 표: 마지막 토큰은 앞의 모든 토큰을 본다
    const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(model.encode('은행에 가서 돈을').length)
  })

  it('Decode 를 누르면 토큰이 하나 늘고 캐시 표시와 함께 decode 상태가 된다', async () => {
    const user = userEvent.setup()
    render(<Simulator initialLang="ko" random={() => 0} />)
    const before = model.encode('은행에 가서 돈을').length
    await user.click(screen.getByRole('button', { name: L.decodeOne }))
    const tokens = within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')
    expect(tokens).toHaveLength(before + 1)
    expect(screen.getByText(L.decode(1, before))).toBeInTheDocument()
    expect(screen.getByText(L.newToken)).toBeInTheDocument()
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(L.stageNames.sample)

    await user.click(screen.getByRole('button', { name: L.reset }))
    expect(screen.getByText(L.prefill)).toBeInTheDocument()
  })

  it('프롬프트를 바꾸면 토큰이 다시 계산되고, 빈 입력에는 안내가 나온다', async () => {
    const user = userEvent.setup()
    render(<Simulator initialLang="ko" />)
    const input = screen.getByRole('textbox', { name: L.prompt })
    await user.clear(input)
    await user.type(input, 'abc')
    expect(within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')).toHaveLength(
      3,
    )
    await user.clear(input)
    expect(screen.getByText(L.empty)).toBeInTheDocument()
  })

  it('층과 헤드를 바꿀 수 있고, 온도 슬라이더가 확률 막대를 바꾼다', () => {
    render(<Simulator initialLang="ko" />)
    fireEvent.change(screen.getByRole('combobox', { name: L.layer }), { target: { value: '2' } })
    expect(screen.getByText(L.panels.block(2))).toBeInTheDocument()
    fireEvent.change(screen.getByRole('slider', { name: L.temperature }), {
      target: { value: '0.2' },
    })
    expect(screen.getByText('0.2')).toBeInTheDocument()
  })

  it('영어 라벨로 전환된다', async () => {
    const user = userEvent.setup()
    render(<Simulator initialLang="ko" />)
    await user.click(screen.getByRole('radio', { name: 'English' }))
    expect(screen.getByText(LABELS.en.prefill)).toBeInTheDocument()
  })
})
