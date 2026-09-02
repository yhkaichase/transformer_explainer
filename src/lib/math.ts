/**
 * 트랜스포머의 핵심 연산을 작은 크기로 구현한 순수 함수 모음.
 *
 * 학습된 가중치는 없습니다. 계산 과정을 눈으로 확인하고 시각화에 쓰기 위한 것입니다.
 * 수식 출처: Vaswani et al., "Attention Is All You Need" (2017), 3.2.1절
 *   Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
 */

export type Vector = number[]
export type Matrix = number[][]

function assertSameLength(a: Vector, b: Vector): void {
  if (a.length !== b.length) {
    throw new RangeError(`벡터 길이가 다릅니다: ${a.length} vs ${b.length}`)
  }
}

/**
 * 점수 목록을 합이 1인 확률로 바꾼다.
 * temperature 가 1보다 작으면 분포가 뾰족해지고, 크면 평평해진다.
 * -Infinity 점수는 확률 0이 된다 (인과 마스크에 사용).
 */
export function softmax(scores: Vector, temperature = 1): Vector {
  if (scores.length === 0) return []
  if (!(temperature > 0)) {
    throw new RangeError('temperature는 0보다 커야 합니다.')
  }
  const scaled = scores.map((s) => s / temperature)
  // 수치 안정성을 위해 최댓값을 빼고 지수를 취한다 (결과는 변하지 않음).
  const max = scaled.reduce((m, s) => (Number.isFinite(s) && s > m ? s : m), -Infinity)
  if (!Number.isFinite(max)) {
    throw new RangeError('모든 점수가 -Infinity 이면 확률을 정할 수 없습니다.')
  }
  const exps = scaled.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

/** 두 벡터의 내적 (같은 위치의 값을 곱해 모두 더한 것). */
export function dot(a: Vector, b: Vector): number {
  assertSameLength(a, b)
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return sum
}

/** 행과 열을 바꾼다. */
export function transpose(m: Matrix): Matrix {
  if (m.length === 0) return []
  return m[0].map((_, column) => m.map((row) => row[column]))
}

/** 행렬 곱 A(n×k) · B(k×m) = C(n×m). */
export function matMul(a: Matrix, b: Matrix): Matrix {
  const bT = transpose(b)
  return a.map((row) => bT.map((column) => dot(row, column)))
}

export interface AttentionOptions {
  /** true 이면 각 토큰이 자기보다 뒤에 있는 토큰을 보지 못하게 막는다 (디코더 전용 모델). */
  causal?: boolean
}

export interface AttentionResult {
  /** weights[i][j]: i번째 토큰이 j번째 토큰에 두는 가중치. 각 행의 합은 1. */
  weights: Matrix
  /** 가중치로 V 를 섞은 결과. 토큰 수 × d_v. */
  output: Matrix
}

/**
 * 스케일드 닷프로덕트 어텐션.
 * q, k: 토큰 수 × d_k, v: 토큰 수 × d_v.
 */
export function scaledDotProductAttention(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  options: AttentionOptions = {},
): AttentionResult {
  if (k.length === 0 || k[0].length === 0) {
    throw new RangeError('K 가 비어 있습니다.')
  }
  const scale = Math.sqrt(k[0].length)
  const weights = q.map((qi, i) =>
    softmax(k.map((kj, j) => (options.causal && j > i ? -Infinity : dot(qi, kj) / scale))),
  )
  return { weights, output: matMul(weights, v) }
}

/** 벡터의 평균을 0, 분산을 1로 맞춘다 (학습 가능한 scale/shift 는 생략). */
export function layerNorm(x: Vector, epsilon = 1e-5): Vector {
  if (x.length === 0) return []
  const mean = x.reduce((a, b) => a + b, 0) / x.length
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length
  const denominator = Math.sqrt(variance + epsilon)
  return x.map((value) => (value - mean) / denominator)
}

/**
 * 원 논문 3.5절의 사인·코사인 위치 인코딩.
 *   PE(pos, 2i)   = sin(pos / base^(2i / d_model))
 *   PE(pos, 2i+1) = cos(pos / base^(2i / d_model))
 * 짝수 차원은 sin, 홀수 차원은 cos 이며, 앞쪽 차원일수록 빠르게 변한다.
 */
export function positionalEncoding(position: number, dModel: number, base = 10000): Vector {
  if (!Number.isInteger(dModel) || dModel <= 0 || dModel % 2 !== 0) {
    throw new RangeError('d_model 은 양의 짝수여야 합니다.')
  }
  const encoding: Vector = []
  for (let i = 0; i < dModel / 2; i++) {
    const angle = position / base ** ((2 * i) / dModel)
    encoding.push(Math.sin(angle), Math.cos(angle))
  }
  return encoding
}

/**
 * 확률 벡터에서 인덱스 하나를 뽑는다 (누적 확률 방식).
 * random 은 [0, 1) 범위의 난수를 돌려주는 함수. 테스트에서는 고정 함수를 넣는다.
 */
export function sampleIndex(probabilities: Vector, random: () => number = Math.random): number {
  if (probabilities.length === 0) {
    throw new RangeError('확률 벡터가 비어 있습니다.')
  }
  const r = random()
  let cumulative = 0
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i]
    if (r < cumulative) return i
  }
  // 부동소수점 오차로 합이 1 에 조금 못 미칠 때는 마지막 항목을 돌려준다.
  return probabilities.length - 1
}

/** ReLU: 음수를 0 으로 자른다. */
export function relu(x: Vector): Vector {
  return x.map((value) => Math.max(0, value))
}

/** 벡터 x 에 행렬 w 를 곱하고 편향 b 를 더한다: xW + b. */
export function affine(x: Vector, w: Matrix, b: Vector): Vector {
  if (w.length !== x.length) {
    throw new RangeError(`입력 길이 ${x.length} 와 행렬의 행 수 ${w.length} 가 다릅니다.`)
  }
  const result = matMul([x], w)[0]
  assertSameLength(result, b)
  return result.map((value, i) => value + b[i])
}

export interface FeedForwardResult {
  /** 첫 선형 변환 결과 xW₁ + b₁ (ReLU 적용 전) */
  preActivation: Vector
  /** ReLU 적용 후의 숨은 층 */
  hidden: Vector
  /** 두 번째 선형 변환 결과 */
  output: Vector
}

/**
 * 원 논문 3.3절의 위치별 피드포워드 네트워크.
 *   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
 */
export function feedForward(
  x: Vector,
  w1: Matrix,
  b1: Vector,
  w2: Matrix,
  b2: Vector,
): FeedForwardResult {
  const preActivation = affine(x, w1, b1)
  const hidden = relu(preActivation)
  const output = affine(hidden, w2, b2)
  return { preActivation, hidden, output }
}

/** 교차 엔트로피 손실: 정답 항목 확률의 -ln. 정답 확률이 1 이면 0, 작을수록 커진다. */
export function crossEntropy(probabilities: Vector, targetIndex: number): number {
  if (targetIndex < 0 || targetIndex >= probabilities.length) {
    throw new RangeError('정답 인덱스가 범위를 벗어났습니다.')
  }
  // 0 - ln(1) 로 써서 -0 이 아니라 +0 을 돌려준다.
  return 0 - Math.log(probabilities[targetIndex])
}

/**
 * softmax + 교차 엔트로피 손실을 점수(로짓)에 대해 경사하강 한 걸음 내려간다.
 * 손실의 점수에 대한 기울기는 (softmax(scores) - onehot(target)) 이다.
 */
export function gradientStep(scores: Vector, targetIndex: number, learningRate: number): Vector {
  if (!(learningRate > 0)) {
    throw new RangeError('learningRate 는 0 보다 커야 합니다.')
  }
  const probabilities = softmax(scores)
  if (targetIndex < 0 || targetIndex >= scores.length) {
    throw new RangeError('정답 인덱스가 범위를 벗어났습니다.')
  }
  return scores.map((score, i) => {
    const gradient = probabilities[i] - (i === targetIndex ? 1 : 0)
    return score - learningRate * gradient
  })
}
