/** 학습 스크립트가 내보낸 fp16(base64) 텐서를 읽는 공통 함수. tinyTransformer.ts 와 tinyDsv41.ts 가 함께 쓴다. */

export interface TensorRecord {
  shape: number[]
  dtype: 'f16'
  data: string
}

function decodeBase64(data: string): Uint8Array {
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** IEEE 754 반정밀도(16비트) → 32비트 실수 */
export function halfToFloat(half: number): number {
  const sign = half & 0x8000 ? -1 : 1
  const exponent = (half >> 10) & 0x1f
  const fraction = half & 0x3ff
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024)
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024)
}

export function decodeTensor(record: TensorRecord): Float32Array {
  const bytes = decodeBase64(record.data)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const count = bytes.byteLength / 2
  const out = new Float32Array(count)
  for (let i = 0; i < count; i++) out[i] = halfToFloat(view.getUint16(i * 2, true))
  const expected = record.shape.reduce((a, b) => a * b, 1)
  if (expected !== count) throw new Error(`텐서 크기가 맞지 않습니다: ${expected} vs ${count}`)
  return out
}

/** rows×k 행렬 a 에 k×n 행렬 w 를 곱하고 편향 b 를 더한다 (행 우선). */
export function matmulAdd(
  a: Float32Array,
  rows: number,
  k: number,
  w: Float32Array,
  n: number,
  b: Float32Array,
): Float32Array {
  const out = new Float32Array(rows * n)
  for (let r = 0; r < rows; r++) {
    for (let j = 0; j < n; j++) {
      let sum = b[j]
      for (let i = 0; i < k; i++) sum += a[r * k + i] * w[i * n + j]
      out[r * n + j] = sum
    }
  }
  return out
}

export function layerNormRows(
  x: Float32Array,
  rows: number,
  d: number,
  g: Float32Array,
  b: Float32Array,
  eps: number,
): Float32Array {
  const out = new Float32Array(rows * d)
  for (let r = 0; r < rows; r++) {
    let mean = 0
    for (let i = 0; i < d; i++) mean += x[r * d + i]
    mean /= d
    let variance = 0
    for (let i = 0; i < d; i++) variance += (x[r * d + i] - mean) ** 2
    variance /= d
    const inv = 1 / Math.sqrt(variance + eps)
    for (let i = 0; i < d; i++) out[r * d + i] = (x[r * d + i] - mean) * inv * g[i] + b[i]
  }
  return out
}
