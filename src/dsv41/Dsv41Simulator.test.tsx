import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { loadTinyDsv41 } from '../lib/tinyDsv41'
import { Dsv41Simulator } from './Dsv41Simulator'
import { LABELS } from './labels'
import { STAGES } from './stages'

const L = LABELS.ko
const model = loadTinyDsv41()
const PROMPT = 'DeepSeek-V4.1-Flash의 KV cache는'
const T = model.encode(PROMPT).length
const DS = model.decoderStartFor(T)

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

const layerMap = () => screen.getByRole('region', { name: L.panels.layerMap })

describe('Dsv41Simulator', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/DCv4.1-simulator.html')
  })

  it('프롬프트 토큰, 층 배치, 단계, prefill 상태를 보여 준다', () => {
    render(<Dsv41Simulator initialLang="ko" />)
    const tokens = within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')
    expect(tokens).toHaveLength(T)
    expect(screen.getByText(L.prefill(T, T - DS), { selector: 'p.sim-mode' })).toBeInTheDocument()
    expect(screen.getByText('제작자: 최영하')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { current: 'step' })).toHaveLength(1)
    expect(screen.getByText(L.stageNames.merge, { selector: 'ol button' })).toBeInTheDocument()
    // 층 배치: 칩 10개, 기본 선택은 L1 (Full)
    const chips = within(layerMap()).getAllByRole('button')
    expect(chips).toHaveLength(model.config.layers.length)
    expect(chips[1]).toHaveAttribute('aria-pressed', 'true')
    expect(within(chips[1]).getByText('KV 1')).toBeInTheDocument()
    // decoder 를 지나지 않는 프롬프트 앞부분 토큰에는 enc 표시가 붙는다
    expect(
      within(screen.getByRole('list', { name: L.focus })).getAllByText(L.encOnly),
    ).toHaveLength(DS)
  })

  it('층을 바꾸면 모드에 따라 물려받는 단계와 없는 단계가 표시된다', async () => {
    const user = userEvent.setup()
    render(<Dsv41Simulator initialLang="ko" />)
    await user.click(
      within(layerMap()).getByRole('button', { name: L.layerOption(2, L.modeTitles.reuse) }),
    )
    expect(screen.getAllByText(L.inherited(1)).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: new RegExp(L.stageNames.topk) }))
    expect(screen.getByText(L.stateNote.inherit(1))).toBeInTheDocument()

    await user.click(
      within(layerMap()).getByRole('button', { name: L.layerOption(0, L.modeTitles.swa) }),
    )
    await user.click(screen.getByRole('button', { name: new RegExp(L.stageNames.compress) }))
    expect(screen.getAllByText(L.stateNote.none).length).toBeGreaterThan(0)

    // Reindex 층은 후보 풀을 L5 에서 물려받는다
    await user.click(
      within(layerMap()).getByRole('button', { name: L.layerOption(7, L.modeTitles.reindex) }),
    )
    await user.click(screen.getByRole('button', { name: new RegExp(L.stageNames.pool) }))
    expect(screen.getByText(L.stateNote.inherit(5))).toBeInTheDocument()
    expect(screen.getAllByText(L.panels.block(7, L.modeTitles.reindex)).length).toBeGreaterThan(0)
  })

  it('점수 단계는 초점 토큰이 보는 엔트리마다 점수 표를 보여 준다', async () => {
    const user = userEvent.setup()
    render(<Dsv41Simulator initialLang="ko" />)
    await user.click(screen.getByRole('button', { name: new RegExp(L.stageNames.scores) }))
    const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(Math.ceil(T / 2)) // encoder 는 2토큰당 엔트리 하나
    expect(rows.filter((r) => r.classList.contains('is-picked'))).toHaveLength(model.config.topK)
  })

  it('Decode 를 누르면 토큰이 하나 늘고 decode 상태가 된다', async () => {
    const user = userEvent.setup()
    render(<Dsv41Simulator initialLang="ko" random={() => 0} />)
    await user.click(screen.getByRole('button', { name: L.decodeOne }))
    const tokens = within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')
    expect(tokens).toHaveLength(T + 1)
    expect(screen.getByText(L.decode(1, T), { selector: 'p.sim-mode' })).toBeInTheDocument()
    expect(screen.getByText(L.newToken)).toBeInTheDocument()
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(L.stageNames.sample)
    const result = screen.getByTestId('sim-result')
    const generated = model.generate(PROMPT, 1, 1, () => 0).slice(PROMPT.length)
    expect(result.querySelector('.sim-gen')?.textContent).toBe(generated)
    await user.click(screen.getByRole('button', { name: L.reset }))
    expect(screen.getByText(L.prefill(T, T - DS), { selector: 'p.sim-mode' })).toBeInTheDocument()
  })

  it('KV cache 카드는 엔트리 수, 토큰당 바이트, 실제 모델 값을 보여 준다', () => {
    render(<Dsv41Simulator initialLang="ko" />)
    const card = screen.getByTestId('sim-cache')
    expect(card).toHaveTextContent(L.cacheCard.perToken(30))
    expect(card).toHaveTextContent('890 B/token')
    expect(card).toHaveTextContent(L.cacheCard.entries(1, 2, Math.ceil(T / 2), 15))
  })

  it('프롬프트를 바꾸면 토큰이 다시 계산되고, 빈 입력에는 안내가 나온다', async () => {
    const user = userEvent.setup()
    render(<Dsv41Simulator initialLang="ko" />)
    const input = screen.getByRole('textbox', { name: L.prompt })
    await user.clear(input)
    await user.type(input, 'abc')
    expect(within(screen.getByRole('list', { name: L.focus })).getAllByRole('button')).toHaveLength(
      3,
    )
    await user.clear(input)
    expect(screen.getByText(L.empty)).toBeInTheDocument()
  })

  it('영어 라벨로 전환된다', async () => {
    const user = userEvent.setup()
    render(<Dsv41Simulator initialLang="ko" />)
    await user.click(screen.getByRole('radio', { name: 'English' }))
    expect(
      screen.getByText(LABELS.en.stageNames.tokens, { selector: 'ol button' }),
    ).toBeInTheDocument()
    expect(screen.getByText(LABELS.en.cacheCard.title)).toBeInTheDocument()
  })
})

describe('자동 재생', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('배속에 따라 단계가 넘어간다', () => {
    render(<Dsv41Simulator initialLang="ko" />)
    fireEvent.change(screen.getByRole('combobox', { name: L.speed }), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: L.autoplay }))
    act(() => {
      vi.advanceTimersByTime(1600 / 4 + 5)
    })
    expect(screen.getByRole('button', { current: 'step' })).toHaveTextContent(
      L.stageNames[STAGES[1]],
    )
    fireEvent.click(screen.getByRole('button', { name: L.autoplayStop }))
  })
})
