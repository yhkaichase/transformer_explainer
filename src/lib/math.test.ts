import { describe, expect, it } from 'vitest'
import { dot, layerNorm, matMul, scaledDotProductAttention, softmax, transpose } from './math'

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)

describe('softmax', () => {
  it('합이 1이고, 큰 점수가 큰 확률을 받는다', () => {
    const p = softmax([1, 2, 3])
    expect(sum(p)).toBeCloseTo(1, 10)
    expect(p[2]).toBeGreaterThan(p[1])
    expect(p[1]).toBeGreaterThan(p[0])
  })

  it('모든 점수에 같은 값을 더해도 결과가 같다', () => {
    const a = softmax([1, 2, 3])
    const b = softmax([101, 102, 103])
    a.forEach((value, i) => expect(value).toBeCloseTo(b[i], 10))
  })

  it('온도가 낮을수록 분포가 뾰족해진다', () => {
    const cold = softmax([1, 2, 3], 0.5)
    const hot = softmax([1, 2, 3], 2)
    expect(Math.max(...cold)).toBeGreaterThan(Math.max(...hot))
  })

  it('-Infinity 점수는 확률 0이 된다', () => {
    const p = softmax([0, -Infinity, 0])
    expect(p[1]).toBe(0)
    expect(p[0]).toBeCloseTo(0.5, 10)
  })

  it('빈 입력은 빈 배열을 돌려준다', () => {
    expect(softmax([])).toEqual([])
  })

  it('온도가 0 이하이면 오류', () => {
    expect(() => softmax([1], 0)).toThrow(RangeError)
  })

  it('모든 점수가 -Infinity 이면 오류', () => {
    expect(() => softmax([-Infinity, -Infinity])).toThrow(RangeError)
  })
})

describe('dot / transpose / matMul', () => {
  it('내적을 계산한다', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32)
  })

  it('길이가 다르면 오류', () => {
    expect(() => dot([1], [1, 2])).toThrow(RangeError)
  })

  it('전치한다', () => {
    expect(
      transpose([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ])
    expect(transpose([])).toEqual([])
  })

  it('행렬을 곱한다', () => {
    expect(
      matMul(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ),
    ).toEqual([
      [19, 22],
      [43, 50],
    ])
  })
})

describe('scaledDotProductAttention', () => {
  const identity = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]

  it('각 행의 가중치 합은 1', () => {
    const { weights } = scaledDotProductAttention(identity, identity, identity)
    weights.forEach((row) => expect(sum(row)).toBeCloseTo(1, 10))
  })

  it('질의와 가장 잘 맞는 키가 가장 큰 가중치를 받는다', () => {
    const { weights } = scaledDotProductAttention(identity, identity, identity)
    weights.forEach((row, i) => expect(Math.max(...row)).toBe(row[i]))
  })

  it('점수가 모두 같으면 가중치는 균등하고 출력은 V 의 평균이다', () => {
    const q = [[0, 0]]
    const k = [
      [1, 1],
      [2, 2],
    ]
    const v = [
      [1, 0],
      [0, 1],
    ]
    const { weights, output } = scaledDotProductAttention(q, k, v)
    expect(weights[0][0]).toBeCloseTo(0.5, 10)
    expect(weights[0][1]).toBeCloseTo(0.5, 10)
    expect(output[0][0]).toBeCloseTo(0.5, 10)
    expect(output[0][1]).toBeCloseTo(0.5, 10)
  })

  it('인과 마스크는 미래 토큰의 가중치를 0으로 만든다', () => {
    const { weights } = scaledDotProductAttention(identity, identity, identity, { causal: true })
    expect(weights[0][0]).toBe(1)
    expect(weights[0][1]).toBe(0)
    expect(weights[0][2]).toBe(0)
    expect(weights[1][2]).toBe(0)
    weights.forEach((row) => expect(sum(row)).toBeCloseTo(1, 10))
  })

  it('점수를 √d_k 로 나눈다', () => {
    // q·k₀ = 4, d_k = 4 → 점수 2. q·k₁ = 0 → 점수 0. 따라서 softmax([2, 0]).
    const q = [[2, 0, 0, 0]]
    const k = [
      [2, 0, 0, 0],
      [0, 0, 0, 0],
    ]
    const v = [[1], [0]]
    const { weights } = scaledDotProductAttention(q, k, v)
    const expected = softmax([2, 0])
    expect(weights[0][0]).toBeCloseTo(expected[0], 10)
    expect(weights[0][1]).toBeCloseTo(expected[1], 10)
  })

  it('K 가 비어 있으면 오류', () => {
    expect(() => scaledDotProductAttention([[1]], [], [])).toThrow(RangeError)
  })
})

describe('layerNorm', () => {
  it('평균 0, 분산 약 1 로 맞춘다', () => {
    const y = layerNorm([1, 2, 3, 4])
    expect(sum(y) / y.length).toBeCloseTo(0, 10)
    const variance = sum(y.map((value) => value * value)) / y.length
    expect(variance).toBeCloseTo(1, 3) // epsilon 때문에 정확히 1은 아니다
  })

  it('빈 입력은 빈 배열', () => {
    expect(layerNorm([])).toEqual([])
  })
})
