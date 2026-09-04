import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
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
    expect(screen.getByText('제작자: 최영하')).toBeInTheDocument()
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

    // 결과 카드: 파란 부분에 모델이 이어 쓴 글자가 바로 보인다
    const result = screen.getByTestId('sim-result')
    const generated = model
      .generate('은행에 가서 돈을', 1, 1, () => 0)
      .slice('은행에 가서 돈을'.length)
    expect(result.querySelector('.sim-gen')?.textContent).toBe(generated)
    expect(within(result).getByText(L.nextTop)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: L.reset }))
    expect(screen.getByText(L.prefill)).toBeInTheDocument()
  })

  it('결과 카드는 처음부터 다음 토큰 후보 5개를 보여 준다', () => {
    render(<Simulator initialLang="ko" />)
    const result = screen.getByTestId('sim-result')
    const top = model.nextTokenDistribution(
      model.forward(model.encode('은행에 가서 돈을')).logits,
      1,
    )
    expect(within(result).getAllByRole('listitem')).toHaveLength(5)
    expect(
      within(result).getByText(`${(top[0].probability * 100).toFixed(1)}%`, {
        selector: '.sim-bar-value',
      }),
    ).toBeInTheDocument()
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

describe('자동 재생 배속과 이어 쓰기', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('배속을 올리면 단계가 더 빨리 넘어간다', () => {
    render(<Simulator initialLang="ko" />)
    fireEvent.change(screen.getByRole('combobox', { name: L.speed }), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: L.autoplay }))
    act(() => {
      vi.advanceTimersByTime(1600 / 4 + 5)
    })
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(
      L.stageNames[STAGES[1]],
    )
    act(() => {
      vi.advanceTimersByTime(1600 / 4 + 5)
    })
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(
      L.stageNames[STAGES[2]],
    )
  })

  it('이어 쓰기 옵션을 켜면 마지막 단계 뒤에 토큰이 하나 늘고 첫 단계로 돌아간다', () => {
    render(<Simulator initialLang="ko" random={() => 0} />)
    const before = within(screen.getByRole('list', { name: L.focus })).getAllByRole('button').length
    fireEvent.click(screen.getByRole('checkbox', { name: L.loopDecode }))
    fireEvent.click(screen.getByRole('button', { name: `${STAGES.length}${L.stageNames.sample}` }))
    fireEvent.click(screen.getByRole('button', { name: L.autoplay }))
    act(() => {
      vi.advanceTimersByTime(1600 + 5)
    })
    expect(within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')).toHaveLength(
      before + 1,
    )
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(
      L.stageNames[STAGES[0]],
    )
    expect(screen.getByText(L.decode(1, before))).toBeInTheDocument()
  })
})
