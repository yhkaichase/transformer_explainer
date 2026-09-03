import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { loadTinyModel } from '../lib/tinyTransformer'
import { LocaleProvider } from '../state/LocaleProvider'
import { LivePipeline } from './LivePipeline'

const ko = getContent('ko').ui.pipeline
const model = loadTinyModel()

function renderPipeline(random?: () => number) {
  return render(
    <LocaleProvider>
      <LivePipeline random={random} />
    </LocaleProvider>,
  )
}

function tokenChips() {
  return within(screen.getByRole('group', { name: ko.tokensLegend })).getAllByRole('button')
}

describe('LivePipeline', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('기본 문장을 토큰으로 나누고 확률 상위 10개를 실제 모델 계산으로 보여 준다', () => {
    renderPipeline()
    const expectedIds = model.encode(ko.defaultText)
    const chips = tokenChips()
    expect(chips).toHaveLength(expectedIds.length)
    expect(chips.map((chip) => chip.textContent)).toEqual(
      Array.from(ko.defaultText).map((char, i) => `${char === ' ' ? '␣' : char}${expectedIds[i]}`),
    )
    const distribution = model.nextTokenDistribution(model.forward(expectedIds).logits, 1)
    const rows = within(screen.getByRole('table', { name: ko.probabilityCaption }))
      .getAllByRole('row')
      .slice(1)
    expect(rows).toHaveLength(10)
    expect(rows[0]).toHaveTextContent(`${(distribution[0].probability * 100).toFixed(1)}%`)
  })

  it('글을 바꾸면 토큰과 확률이 다시 계산된다', async () => {
    const user = userEvent.setup()
    renderPipeline()
    const input = screen.getByLabelText(ko.inputLabel)
    await user.clear(input)
    await user.type(input, '트랜스포머는')
    expect(tokenChips()).toHaveLength(6)
    expect(screen.getByText(ko.charCount(6, model.config.context))).toBeInTheDocument()
    const distribution = model.nextTokenDistribution(
      model.forward(model.encode('트랜스포머는')).logits,
      1,
    )
    const rows = within(screen.getByRole('table', { name: ko.probabilityCaption }))
      .getAllByRole('row')
      .slice(1)
    expect(rows[0]).toHaveTextContent(`${(distribution[0].probability * 100).toFixed(1)}%`)
  })

  it('토큰을 고르면 그 토큰의 숫자 목록과 어텐션 행이 보인다', async () => {
    const user = userEvent.setup()
    renderPipeline()
    await user.click(tokenChips()[0])
    expect(screen.getByText(ko.vectorHeading('은'))).toBeInTheDocument()
    // 첫 토큰은 자기 자신만 볼 수 있으므로 가중치 1.00
    expect(screen.getByText(ko.strongest('은', '은', '1.00'))).toBeInTheDocument()
  })

  it('층과 헤드를 바꾸면 어텐션 표가 바뀐다', () => {
    renderPipeline()
    fireEvent.change(screen.getByRole('combobox', { name: ko.layerLabel }), {
      target: { value: '2' },
    })
    fireEvent.change(screen.getByRole('combobox', { name: ko.headLabel }), {
      target: { value: '1' },
    })
    expect(screen.getByRole('table', { name: ko.attentionCaption(2, 1) })).toBeInTheDocument()
  })

  it('한 글자 뽑아 붙이기는 모델이 뽑은 글자를 입력 끝에 붙이고, 처음으로 가 되돌린다', async () => {
    const user = userEvent.setup()
    renderPipeline(() => 0)
    const expected = model.generate(ko.defaultText, 1, 1, () => 0)
    await user.click(screen.getByRole('button', { name: ko.appendOne }))
    expect(screen.getByLabelText(ko.inputLabel)).toHaveValue(expected)
    await user.click(screen.getByRole('button', { name: ko.reset }))
    expect(screen.getByLabelText(ko.inputLabel)).toHaveValue(ko.defaultText)
  })

  it('빈 입력이면 안내문만 보인다', async () => {
    const user = userEvent.setup()
    renderPipeline()
    await user.clear(screen.getByLabelText(ko.inputLabel))
    expect(screen.getByText(ko.emptyInput)).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
