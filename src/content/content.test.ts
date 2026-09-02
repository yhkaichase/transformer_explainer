import { describe, expect, it } from 'vitest'
import { getContent, LOCALES } from './index'
import { SECTION_META } from './structure'

const ko = getContent('ko')
const en = getContent('en')

/** 문자열 값들을 재귀적으로 모두 모은다 (빈 문자열 검사용). */
function collectStrings(value: unknown, path: string, out: [string, string][]): void {
  if (typeof value === 'string') out.push([path, value])
  else if (Array.isArray(value)) value.forEach((v, i) => collectStrings(v, `${path}[${i}]`, out))
  else if (value && typeof value === 'object') {
    for (const [key, v] of Object.entries(value)) collectStrings(v, `${path}.${key}`, out)
  }
}

describe('content structure', () => {
  it('섹션 id 가 중복되지 않고 번호가 1부터 순서대로 매겨진다', () => {
    const ids = SECTION_META.map((m) => m.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(ko.sections.map((s) => s.number)).toEqual(ids.map((_, i) => i + 1))
  })

  it('토큰 섹션에 토크나이저 데모가 붙어 있다', () => {
    expect(ko.sections.find((s) => s.id === 'tokens')?.interactive).toBe('tokenizer')
  })
})

describe('content parity between locales', () => {
  it('모든 언어가 같은 섹션을 같은 순서로 가진다', () => {
    for (const { value } of LOCALES) {
      expect(getContent(value).sections.map((s) => s.id)).toEqual(ko.sections.map((s) => s.id))
    }
  })

  it('문단 수와 핵심 용어 수가 한국어와 영어에서 같다', () => {
    ko.sections.forEach((section, i) => {
      const other = en.sections[i]
      expect(other.executive.length, `${section.id}.executive`).toBe(section.executive.length)
      expect(other.engineer.length, `${section.id}.engineer`).toBe(section.engineer.length)
      expect((other.keyTerms ?? []).length, `${section.id}.keyTerms`).toBe(
        (section.keyTerms ?? []).length,
      )
    })
  })

  it('참고 자료 URL 목록이 언어와 무관하게 같다', () => {
    expect(en.sources.map((s) => s.url)).toEqual(ko.sources.map((s) => s.url))
  })

  it('빈 문자열이 없다', () => {
    for (const { value } of LOCALES) {
      const { ui, sections, sources } = getContent(value)
      const strings: [string, string][] = []
      collectStrings({ ui, sections, sources }, value, strings)
      const empty = strings.filter(([, s]) => s.trim() === '').map(([path]) => path)
      expect(empty).toEqual([])
    }
  })

  it('토큰 수 문구가 언어별로 만들어진다', () => {
    expect(ko.ui.tokenizer.count(5)).toBe('토큰 5개')
    expect(en.ui.tokenizer.count(1)).toBe('1 token')
    expect(en.ui.tokenizer.count(5)).toBe('5 tokens')
  })
})

describe('demo data', () => {
  it('흐름 단계와 온도 데모 후보가 언어 간에 같은 구조를 가진다', () => {
    expect(en.ui.flow.stages.map((s) => s.section)).toEqual(ko.ui.flow.stages.map((s) => s.section))
    expect(en.ui.temperature.candidates.map((c) => c.score)).toEqual(
      ko.ui.temperature.candidates.map((c) => c.score),
    )
  })

  it('흐름 단계가 가리키는 섹션이 실제로 존재한다', () => {
    const ids = new Set(SECTION_META.map((m) => m.id))
    for (const stage of ko.ui.flow.stages) expect(ids.has(stage.section)).toBe(true)
  })

  it('온도 데모 후보는 점수 내림차순이고 단어가 겹치지 않는다', () => {
    for (const { value } of LOCALES) {
      const candidates = getContent(value).ui.temperature.candidates
      const scores = candidates.map((c) => c.score)
      expect(scores).toEqual([...scores].sort((a, b) => b - a))
      expect(new Set(candidates.map((c) => c.token)).size).toBe(candidates.length)
    }
  })

  it('언어별 문구 함수가 동작한다', () => {
    expect(ko.ui.flow.stepLabel(2, 6)).toBe('2단계 / 6단계')
    expect(en.ui.flow.stepLabel(2, 6)).toBe('Step 2 of 6')
    expect(ko.ui.positional.selectedHeading(3)).toBe('위치 3의 패턴')
    expect(en.ui.temperature.valueLabel(0.5)).toBe('Temperature 0.5')
  })
})
