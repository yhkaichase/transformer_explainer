import modelFile from '../data/model/tiny-transformer.json'
import { sampleIndex, softmax } from './math'

/**
 * scripts/train_tiny_model.py 가 학습해 내보낸 작은 문자 단위 트랜스포머를 브라우저에서 실행한다.
 * 구조(학습 코드와 동일): 토큰 임베딩 + 학습된 위치 임베딩 → [pre-LN 멀티헤드 인과 어텐션 → pre-LN ReLU 피드포워드] × L
 * → 최종 LayerNorm → 임베딩과 가중치를 공유한 출력층.
 */

export interface TensorRecord {
  shape: number[]
  dtype: 'f16'
  data: string
}

export interface LayerRecord {
  ln1_g: TensorRecord
  ln1_b: TensorRecord
  wq: TensorRecord
  bq: TensorRecord
  wk: TensorRecord
  bk: TensorRecord
  wv: TensorRecord
  bv: TensorRecord
  wo: TensorRecord
  bo: TensorRecord
  ln2_g: TensorRecord
  ln2_b: TensorRecord
  w1: TensorRecord
  b1: TensorRecord
  w2: TensorRecord
  b2: TensorRecord
}

export interface TinyModelConfig {
  dModel: number
  nHeads: number
  nLayers: number
  dFf: number
  context: number
  layerNormEps: number
}

export interface TinyModelFile {
  schemaVersion: number
  description: string
  config: TinyModelConfig
  vocab: string[]
  unkToken: string
  training: {
    steps: number
    batch: number
    learningRate: number
    seed: number
    corpusChars: number
    paramCount: number
    lossHistory: { step: number; loss: number }[]
    generatedAt: string
    jaxVersion: string
  }
  weights: {
    tok_emb: TensorRecord
    pos_emb: TensorRecord
    lnf_g: TensorRecord
    lnf_b: TensorRecord
    layers: LayerRecord[]
  }
  reference: {
    prompt: string
    ids: number[]
    top5: { token: string; probability: number }[]
    attention_l0_h0: number[][]
    attention_last_layer_h1_last_row: number[]
  }[]
}

export interface ForwardResult {
  ids: number[]
  tokens: string[]
  /** 토큰별 입력 벡터 (토큰 임베딩 + 위치 임베딩), 길이 d_model */
  embeddings: Float32Array[]
  /** [층][헤드][보는 토큰][보이는 토큰] */
  attentions: number[][][][]
  /** 마지막 토큰 위치의 로짓 (어휘 크기) */
  logits: Float32Array
}

export interface Candidate {
  id: number
  token: string
  probability: number
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

interface DecodedLayer {
  ln1G: Float32Array
  ln1B: Float32Array
  wq: Float32Array
  bq: Float32Array
  wk: Float32Array
  bk: Float32Array
  wv: Float32Array
  bv: Float32Array
  wo: Float32Array
  bo: Float32Array
  ln2G: Float32Array
  ln2B: Float32Array
  w1: Float32Array
  b1: Float32Array
  w2: Float32Array
  b2: Float32Array
}

/** rows×k 행렬 a 에 k×n 행렬 w 를 곱하고 편향 b 를 더한다. */
function matmulAdd(
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

function layerNormRows(
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

export class TinyTransformer {
  readonly config: TinyModelConfig
  readonly vocab: string[]
  readonly unkToken: string
  readonly training: TinyModelFile['training']
  private readonly stoi: Map<string, number>
  private readonly tokEmb: Float32Array
  private readonly posEmb: Float32Array
  private readonly lnfG: Float32Array
  private readonly lnfB: Float32Array
  private readonly layers: DecodedLayer[]

  constructor(file: TinyModelFile) {
    if (file.schemaVersion !== 1)
      throw new Error(`지원하지 않는 모델 파일 버전: ${file.schemaVersion}`)
    this.config = file.config
    this.vocab = file.vocab
    this.unkToken = file.unkToken
    this.training = file.training
    this.stoi = new Map(file.vocab.map((token, id) => [token, id]))
    this.tokEmb = decodeTensor(file.weights.tok_emb)
    this.posEmb = decodeTensor(file.weights.pos_emb)
    this.lnfG = decodeTensor(file.weights.lnf_g)
    this.lnfB = decodeTensor(file.weights.lnf_b)
    this.layers = file.weights.layers.map((layer) => ({
      ln1G: decodeTensor(layer.ln1_g),
      ln1B: decodeTensor(layer.ln1_b),
      wq: decodeTensor(layer.wq),
      bq: decodeTensor(layer.bq),
      wk: decodeTensor(layer.wk),
      bk: decodeTensor(layer.bk),
      wv: decodeTensor(layer.wv),
      bv: decodeTensor(layer.bv),
      wo: decodeTensor(layer.wo),
      bo: decodeTensor(layer.bo),
      ln2G: decodeTensor(layer.ln2_g),
      ln2B: decodeTensor(layer.ln2_b),
      w1: decodeTensor(layer.w1),
      b1: decodeTensor(layer.b1),
      w2: decodeTensor(layer.w2),
      b2: decodeTensor(layer.b2),
    }))
  }

  /** 글자 단위 토큰화. 어휘에 없는 글자는 id 0(␀). 문맥 길이를 넘으면 뒤쪽만 남긴다. */
  encode(text: string): number[] {
    return Array.from(text)
      .slice(-this.config.context)
      .map((char) => this.stoi.get(char) ?? 0)
  }

  decode(ids: number[]): string {
    return ids.map((id) => this.vocab[id] ?? this.unkToken).join('')
  }

  forward(ids: number[]): ForwardResult {
    const { dModel: d, nHeads: h, nLayers, dFf, context, layerNormEps: eps } = this.config
    const t = ids.length
    if (t === 0) throw new RangeError('토큰이 하나 이상 필요합니다.')
    if (t > context) throw new RangeError(`문맥 길이 ${context} 를 넘었습니다.`)
    const dh = d / h
    const scale = 1 / Math.sqrt(dh)

    const x = new Float32Array(t * d)
    for (let i = 0; i < t; i++) {
      for (let j = 0; j < d; j++)
        x[i * d + j] = this.tokEmb[ids[i] * d + j] + this.posEmb[i * d + j]
    }
    const embeddings = Array.from({ length: t }, (_, i) => x.slice(i * d, (i + 1) * d))
    const attentions: number[][][][] = []

    for (let l = 0; l < nLayers; l++) {
      const layer = this.layers[l]
      const hn = layerNormRows(x, t, d, layer.ln1G, layer.ln1B, eps)
      const q = matmulAdd(hn, t, d, layer.wq, d, layer.bq)
      const k = matmulAdd(hn, t, d, layer.wk, d, layer.bk)
      const v = matmulAdd(hn, t, d, layer.wv, d, layer.bv)
      const concat = new Float32Array(t * d)
      const layerAttention: number[][][] = []
      for (let head = 0; head < h; head++) {
        const offset = head * dh
        const rows: number[][] = []
        for (let i = 0; i < t; i++) {
          const scores: number[] = []
          for (let j = 0; j <= i; j++) {
            let dot = 0
            for (let e = 0; e < dh; e++) dot += q[i * d + offset + e] * k[j * d + offset + e]
            scores.push(dot * scale)
          }
          const weights = softmax(scores)
          const row = new Array<number>(t).fill(0)
          for (let j = 0; j <= i; j++) row[j] = weights[j]
          rows.push(row)
          for (let e = 0; e < dh; e++) {
            let sum = 0
            for (let j = 0; j <= i; j++) sum += weights[j] * v[j * d + offset + e]
            concat[i * d + offset + e] = sum
          }
        }
        layerAttention.push(rows)
      }
      attentions.push(layerAttention)
      const projected = matmulAdd(concat, t, d, layer.wo, d, layer.bo)
      for (let i = 0; i < t * d; i++) x[i] += projected[i]

      const hn2 = layerNormRows(x, t, d, layer.ln2G, layer.ln2B, eps)
      const hidden = matmulAdd(hn2, t, d, layer.w1, dFf, layer.b1)
      for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0
      const ff = matmulAdd(hidden, t, dFf, layer.w2, d, layer.b2)
      for (let i = 0; i < t * d; i++) x[i] += ff[i]
    }

    const final = layerNormRows(x, t, d, this.lnfG, this.lnfB, eps)
    const last = (t - 1) * d
    const vocabSize = this.vocab.length
    const logits = new Float32Array(vocabSize)
    for (let vIndex = 0; vIndex < vocabSize; vIndex++) {
      let sum = 0
      for (let j = 0; j < d; j++) sum += final[last + j] * this.tokEmb[vIndex * d + j]
      logits[vIndex] = sum
    }

    return { ids, tokens: ids.map((id) => this.vocab[id]), embeddings, attentions, logits }
  }

  /** 마지막 위치의 다음 글자 확률 (온도 적용, 확률 내림차순). ␀ 는 뽑히지 않게 0 으로 둔다. */
  nextTokenDistribution(logits: Float32Array, temperature = 1): Candidate[] {
    const probabilities = softmax(Array.from(logits), temperature)
    probabilities[0] = 0
    const total = probabilities.reduce((a, b) => a + b, 0)
    return probabilities
      .map((p, id) => ({ id, token: this.vocab[id], probability: p / total }))
      .filter((candidate) => candidate.id !== 0)
      .sort((a, b) => b.probability - a.probability)
  }

  /** 온도에 따라 다음 글자를 뽑아 이어 쓴다. */
  generate(
    text: string,
    steps: number,
    temperature = 1,
    random: () => number = Math.random,
  ): string {
    let current = text
    for (let step = 0; step < steps; step++) {
      const ids = this.encode(current)
      if (ids.length === 0) break
      const distribution = this.nextTokenDistribution(this.forward(ids).logits, temperature)
      const index = sampleIndex(
        distribution.map((c) => c.probability),
        random,
      )
      current += distribution[index].token
    }
    return current
  }
}

let cached: TinyTransformer | null = null

/** 번들에 포함된 학습 결과 파일로 모델을 만든다 (한 번만 디코딩). */
export function loadTinyModel(): TinyTransformer {
  if (!cached) cached = new TinyTransformer(modelFile as unknown as TinyModelFile)
  return cached
}
