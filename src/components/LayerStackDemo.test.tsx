import { fireEvent, render, screen, within } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { getContent } from '../content'
import { LocaleProvider } from '../state/LocaleProvider'
import { DEFAULT_LAYERS, LayerStackDemo } from './LayerStackDemo'

const ko = getContent('ko').ui.stack

function renderDemo() {
  return render(
    <LocaleProvider>
      <LayerStackDemo />
    </LocaleProvider>,
  )
}

describe('LayerStackDemo', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState({}, '', '/')
  })

  it('기본은 6층이고 각 층에 어텐션과 피드포워드 서브층이 있다', () => {
    renderDemo()
    const list = screen.getByRole('list', { name: ko.sliderLabel })
    const layers = within(list).getAllByRole('listitem')
    expect(layers).toHaveLength(DEFAULT_LAYERS)
    // 위에서부터 마지막 층, 아래가 1번째 층
    expect(layers[0]).toHaveTextContent(ko.layerTitle(DEFAULT_LAYERS))
    expect(layers[DEFAULT_LAYERS - 1]).toHaveTextContent(ko.layerTitle(1))
    expect(within(layers[0]).getAllByRole('img')).toHaveLength(2)
    expect(screen.getByText(ko.layersCaption(DEFAULT_LAYERS))).toBeInTheDocument()
  })

  it('슬라이더로 층 수를 바꾼다', () => {
    renderDemo()
    fireEvent.change(screen.getByRole('slider', { name: ko.sliderLabel }), {
      target: { value: '3' },
    })
    const list = screen.getByRole('list', { name: ko.sliderLabel })
    expect(within(list).getAllByRole('listitem')).toHaveLength(3)
    expect(screen.getByText(ko.layersCaption(3))).toBeInTheDocument()
  })
})
