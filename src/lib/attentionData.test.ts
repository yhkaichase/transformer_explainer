import { describe, expect, it } from 'vitest'
import { attentionFixture } from '../test/fixtures/attention-fixture'
import {
  AttentionDataError,
  displayToken,
  loadDatasets,
  validateDataset,
  weightBin,
} from './attentionData'

describe('validateDataset', () => {
  it('올바른 데이터를 받아들인다', () => {
    const dataset = validateDataset(attentionFixture('en'))
    expect(dataset.locale).toBe('en')
    expect(dataset.examples[0].tokens).toEqual(['a', ' b', ' c'])
  })

  it('schemaVersion, locale, 모양이 틀리면 이유를 담은 오류를 던진다', () => {
    const base = attentionFixture()
    expect(() => validateDataset({ ...base, schemaVersion: 2 })).toThrow(AttentionDataError)
    expect(() => validateDataset({ ...base, locale: 'fr' })).toThrow(/locale/)
    expect(() => validateDataset({ ...base, examples: 'x' })).toThrow(/examples/)

    const badLayers = structuredClone(base)
    badLayers.examples[0].layers = 3
    expect(() => validateDataset(badLayers)).toThrow(/층 수/)

    const badRow = structuredClone(base)
    badRow.examples[0].attention[0][0][1] = [0.5, 0.5]
    expect(() => validateDataset(badRow)).toThrow(/열 수/)

    const badWeight = structuredClone(base)
    badWeight.examples[0].attention[1][1][2][0] = 1.5
    expect(() => validateDataset(badWeight)).toThrow(/0~1/)
  })
})

describe('loadDatasets', () => {
  it('언어별로 정리하고, 깨진 파일은 오류 목록에 남긴다', () => {
    const { datasets, errors } = loadDatasets({
      '../data/attention/ko.json': attentionFixture('ko'),
      '../data/attention/en.json': attentionFixture('en'),
      '../data/attention/broken.json': { schemaVersion: 1 },
    })
    expect(datasets.ko?.locale).toBe('ko')
    expect(datasets.en?.locale).toBe('en')
    expect(errors).toHaveLength(1)
    expect(errors[0]).toContain('broken.json')
  })
})

describe('weightBin / displayToken', () => {
  it('가중치를 다섯 단계로 나눈다', () => {
    expect([0, 0.049, 0.05, 0.149, 0.15, 0.299, 0.3, 0.499, 0.5, 1].map(weightBin)).toEqual([
      0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
    ])
  })

  it('앞 공백을 지우고 공백만 있는 조각은 ␣ 로 보여 준다', () => {
    expect(displayToken(' bank')).toBe('bank')
    expect(displayToken(' ')).toBe('␣')
  })
})
