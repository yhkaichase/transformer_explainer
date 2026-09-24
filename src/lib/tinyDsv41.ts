import modelFile from '../data/model/tiny-dsv41.json'
import { sampleIndex, softmax } from './math'
import { decodeTensor, layerNormRows, matmulAdd, type TensorRecord } from './tensor'

/**
 * scripts/train_tiny_dsv41.py 가 학습해 내보낸, DeepSeek-V4.1-Flash 구조를 아주 작게 흉내 낸 문자 단위 모델을
 * 브라우저에서 실행한다. 구조(학습 코드와 동일):
 *
 *   토큰 임베딩 + 위치 임베딩
 *   → 층마다 [LN₁ → Main Q · SWA K,V → (모드에 따라) global KV 엔트리 생성/재사용 → 인덱서 점수 → Top-K 선택
 *              → global 로짓 + SWA 로짓을 한 softmax 로 병합 → 가중합 → W_O → 잔차
 *              → LN₂ → MoE(shared 1 + routed Top-2) → 잔차]
 *   → 최종 LayerNorm → 임베딩과 가중치를 공유한 출력층
 *
 * 앞쪽 encoderLayers 개 층이 causal encoder, 나머지가 decoder 다 (CED). decoder 층은 decoderStart 이후 위치만
 * 처리하고, decoder 의 global KV 는 encoder 마지막 출력(=decoder 첫 층의 입력)에서 투영한다.
 */

export type LayerMode = 'swa' | 'full' | 'reuse' | 'reindex'

export interface LayerSpec {
  mode: LayerMode
  /** full 층의 시퀀스 압축비 (encoder 2, decoder 1) */
  m?: number
  /** reuse/reindex 층이 KV 와 indexer K 를 가져오는 full 층 */
  src?: number
  /** reuse 층이 Top-K 선택을 가져오는 층 */
  idx?: number
}

export interface Dsv41Config {
  dModel: number
  context: number
  nHeads: number
  dLatent: number
  window: number
  topK: number
  poolBlock: number
  poolBlocks: number
  indexerHeads: number
  dIndexer: number
  experts: number
  expertsTopK: number
  dExpert: number
  dShared: number
  fp4Group: number
  layerNormEps: number
  encoderLayers: number
  layers: LayerSpec[]
}

export interface Dsv41ReferenceLayer {
  experts: number[]
  gates: number[]
  local_h0: number[]
  selected?: number[]
  global_h0?: number[]
  scores?: number[]
  pool?: number[]
  entry_last?: number[]
}

export interface Dsv41Reference {
  prompt: string
  ids: number[]
  decoderStart: number
  top5: { token: string; probability: number }[]
  layers: Record<string, Dsv41ReferenceLayer>
  greedy: string
}

export interface Dsv41ModelFile {
  schemaVersion: number
  description: string
  config: Dsv41Config
  vocab: string[]
  unkToken: string
  training: {
    steps: number
    batch: number
    learningRate: number
    seed: number
    sparseFromStep: number
    quantFromStep: number
    indexerKlWeight: number
    corpusChars: number
    paramCount: number
    lossHistory: {
      step: number
      loss: number
      indexerKl: number
      sparse: boolean
      quant: boolean
    }[]
    generatedAt: string
    jaxVersion: string
  }
  weights: {
    tok_emb: TensorRecord
    pos_emb: TensorRecord
    lnf_g: TensorRecord
    lnf_b: TensorRecord
    layers: Record<string, TensorRecord>[]
  }
  reference: Dsv41Reference[]
}

/** 디코딩된 한 층의 가중치 (행 우선). 모드에 따라 없는 항목은 undefined. */
export interface Dsv41LayerWeights {
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
  wr: Float32Array
  br: Float32Array
  rbias: Float32Array
  ws1: Float32Array
  bs1: Float32Array
  ws2: Float32Array
  bs2: Float32Array
  we1: Float32Array
  be1: Float32Array
  we2: Float32Array
  be2: Float32Array
  wc?: Float32Array
  bc?: Float32Array
  wi?: Float32Array
  wqi?: Float32Array
  bqi?: Float32Array
  ww?: Float32Array
  bw?: Float32Array
}

/** full 층 하나가 만든 global KV 엔트리 묶음. reuse/reindex 층은 같은 객체를 가리킨다. */
export interface EntryTrace {
  /** 만든 층 */
  layer: number
  /** 시퀀스 압축비 */
  m: number
  /** 엔트리 수 = ceil(T / m) */
  n: number
  /** 엔트리 i 가 맡는 첫 토큰 위치 (m·i). 이 위치 이후의 토큰만 엔트리를 볼 수 있다. */
  entryPos: number[]
  /** 압축기 입력을 이어 붙인 것 (n × m·D) */
  merged: Float32Array
  /** 양자화 전 latent (n × dLatent). m = 2 면 두 토큰이 모두 온 상태 */
  cRaw: Float32Array
  /** 짝이 아직 안 온 상태 [h; 0] 의 latent (n × dLatent). m = 1 이면 cRaw 와 같다 */
  cPartRaw: Float32Array
  /** FP4 로 저장했다가 다시 읽은 값 (n × dLatent). 어텐션은 이 값을 쓴다 */
  c: Float32Array
  cPart: Float32Array
  /** E2M1 코드 (부호 포함, -7..7), 표시용 (n × dLatent) */
  codes: Int8Array
  /** 그룹별 E4M3 scale (n × dLatent/fp4Group) */
  scales: Float32Array
  /** 양자화 전 indexer K = c · W_I (n × dIndexer) */
  kiRaw: Float32Array
  /** MXFP4 로 저장했다가 다시 읽은 indexer K (n × dIndexer) */
  ki: Float32Array
  kiPart: Float32Array
  /** 엔트리별 MXFP4 scale (n) */
  kiScale: Float32Array
}

export interface Dsv41LayerTrace {
  mode: LayerMode
  /** 이 층이 처리를 시작하는 위치 (encoder 0, decoder 는 decoderStart) */
  start: number
  ln1: Float32Array
  /** Main Q (T × H·dLatent) */
  q: Float32Array
  kSwa: Float32Array
  vSwa: Float32Array
  /** 이 층이 쓰는 global KV (swa 층은 null) */
  entries: EntryTrace | null
  n: number
  /** Indexer Q (T × HI·dIndexer), full/reindex 층만 */
  qi: Float32Array | null
  /** 인덱서 헤드 가중치 w_{t,h} (T × HI) */
  wIdx: Float32Array | null
  /** 인덱서 점수 (T × n), 범위 밖은 0 */
  scores: Float32Array | null
  /** 점수를 매긴 범위 (T × n): full 은 보이는 엔트리 전체, reindex 는 후보 풀 */
  range: Uint8Array | null
  /** 후보 풀 (T × n), decoder 의 full 층만 만든다 */
  pool: Uint8Array | null
  /** Top-K 로 고른 엔트리 (T × n) */
  selected: Uint8Array | null
  /** reuse 층이 선택을 가져온 층 */
  selectedFrom: number | null
  /** global 어텐션 가중치 (H × T × n), 선택 안 된 곳 0 */
  attnGlobal: Float32Array
  /** SWA 어텐션 가중치 (H × T × T), 윈도우 밖 0 */
  attnLocal: Float32Array
  /** 마스크된 곳은 -Infinity */
  logitGlobal: Float32Array
  logitLocal: Float32Array
  /** 헤드별 가중합을 이어 붙인 것 (T × H·dLatent) */
  o: Float32Array
  /** W_O 뒤 (T × D) */
  y: Float32Array
  x1: Float32Array
  ln2: Float32Array
  /** 라우터 sigmoid 점수 (T × E) */
  router: Float32Array
  expertMask: Uint8Array
  gates: Float32Array
  shared: Float32Array
  /** 전문가별 출력 (T × E × D) */
  expertOut: Float32Array
  routed: Float32Array
  /** 층 출력 (T × D). start 이전 위치는 입력 그대로 */
  x2: Float32Array
}

export interface Dsv41Trace {
  ids: number[]
  tokens: string[]
  decoderStart: number
  tokenEmbeddings: Float32Array
  positionEmbeddings: Float32Array
  /** 입력 x = E[id] + P[pos] (T × D) */
  x0: Float32Array
  layers: Dsv41LayerTrace[]
  final: Float32Array
  /** 마지막 위치의 로짓 (어휘 크기) */
  logits: Float32Array
}

export interface Candidate {
  id: number
  token: string
  probability: number
}

// ----------------------------------------------------------------------------- 양자화 (학습 스크립트와 같은 규칙)
export const E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6] as const

/** v = mant · 2^exp, mant ∈ [0.5, 1) */
function frexp(v: number): { mant: number; exp: number } {
  if (v === 0 || !Number.isFinite(v)) return { mant: 0, exp: 0 }
  let exp = Math.floor(Math.log2(Math.abs(v))) + 1
  let mant = v / 2 ** exp
  while (Math.abs(mant) >= 1) {
    mant /= 2
    exp += 1
  }
  while (Math.abs(mant) < 0.5) {
    mant *= 2
    exp -= 1
  }
  return { mant, exp }
}

/** 0 이상 실수를 E4M3(지수 4비트, 가수 3비트) 격자로 반올림한다. FP4 scale 저장용. */
export function e4m3(v: number): number {
  if (!(v > 0)) return 0
  const { mant, exp: rawExp } = frexp(Math.max(v, 1e-30))
  let exp = rawExp - 1
  let frac = Math.floor((mant * 2 - 1) * 8 + 0.5) / 8
  if (frac >= 1) {
    exp += 1
    frac = 0
  }
  exp = Math.max(-6, exp)
  return Math.min((1 + frac) * 2 ** exp, 448) // 448 = E4M3 최댓값에서 포화
}

/** E2M1 격자에서 가장 가까운 값의 코드(0..7)와 부호 */
export function nearestE2M1(y: number): { code: number; value: number } {
  const a = Math.abs(y)
  let code = 0
  for (const threshold of [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]) if (a >= threshold) code += 1
  return { code, value: Math.sign(y) * E2M1_VALUES[code] }
}

export interface Fp4Stored {
  /** 다시 읽은 값 */
  dequantized: Float32Array
  /** 부호 포함 코드 (-7..7) */
  codes: Int8Array
  /** 그룹별 scale */
  scales: Float32Array
}

/** main KV 저장: FP4(E2M1) 값 + 채널 그룹마다 E4M3 scale 하나. */
export function fp4Store(vec: ArrayLike<number>, group: number): Fp4Stored {
  const d = vec.length
  const groups = Math.ceil(d / group)
  const dequantized = new Float32Array(d)
  const codes = new Int8Array(d)
  const scales = new Float32Array(groups)
  for (let g = 0; g < groups; g++) {
    let amax = 0
    for (let i = g * group; i < Math.min(d, (g + 1) * group); i++)
      amax = Math.max(amax, Math.abs(vec[i]))
    let scale = e4m3(amax / 6)
    if (!(scale > 0)) scale = 1
    scales[g] = scale
    for (let i = g * group; i < Math.min(d, (g + 1) * group); i++) {
      const { code, value } = nearestE2M1(vec[i] / scale)
      codes[i] = vec[i] < 0 ? -code : code
      dequantized[i] = value * scale
    }
  }
  return { dequantized, codes, scales }
}

export interface Mxfp4Stored {
  dequantized: Float32Array
  codes: Int8Array
  /** 2의 거듭제곱 scale (E8M0) */
  scale: number
}

/** indexer K 저장: E2M1 값 + 벡터당 2의 거듭제곱 scale (MXFP4). */
export function mxfp4Store(vec: ArrayLike<number>): Mxfp4Stored {
  const d = vec.length
  let amax = 0
  for (let i = 0; i < d; i++) amax = Math.max(amax, Math.abs(vec[i]))
  const scale = amax > 0 ? 2 ** Math.ceil(Math.log2(amax / 6)) : 1
  const dequantized = new Float32Array(d)
  const codes = new Int8Array(d)
  for (let i = 0; i < d; i++) {
    const { code, value } = nearestE2M1(vec[i] / scale)
    codes[i] = vec[i] < 0 ? -code : code
    dequantized[i] = value * scale
  }
  return { dequantized, codes, scale }
}

// ----------------------------------------------------------------------------- 캐시 용량 (구조에서 바로 계산)
export interface KvBytes {
  /** main KV 엔트리 하나: FP4 값 + E4M3 scale (바이트) */
  mainEntry: number
  /** indexer K 엔트리 하나: MXFP4 값 + E8M0 scale 1 B */
  indexerEntry: number
  /** KV 를 만드는 full 층들 */
  sources: { layer: number; m: number; perToken: number }[]
  /** 토큰당 global KV 바이트 = Σ (mainEntry + indexerEntry) / m */
  perToken: number
}

/** 매뉴얼의 890 B/token 유도와 같은 식을 이 작은 모델의 구조에 적용한다. */
export function kvBytesPerToken(config: Dsv41Config): KvBytes {
  const mainEntry = config.dLatent / 2 + config.dLatent / config.fp4Group
  const indexerEntry = config.dIndexer / 2 + 1
  const sources = config.layers.flatMap((spec, layer) =>
    spec.mode === 'full'
      ? [{ layer, m: spec.m ?? 1, perToken: (mainEntry + indexerEntry) / (spec.m ?? 1) }]
      : [],
  )
  return {
    mainEntry,
    indexerEntry,
    sources,
    perToken: sources.reduce((sum, s) => sum + s.perToken, 0),
  }
}

// ----------------------------------------------------------------------------- 모델
const NEG = Number.NEGATIVE_INFINITY

function sigmoid(v: number): number {
  return 1 / (1 + Math.exp(-v))
}

/** 값 내림차순, 같으면 앞 인덱스 우선 */
function topIndices(values: ArrayLike<number>, candidates: number[], k: number): number[] {
  return candidates
    .slice()
    .sort((a, b) => values[b] - values[a] || a - b)
    .slice(0, k)
}

/** 위치 t 의 질의가 보는 엔트리 i 의 latent (짝이 안 온 엔트리는 부분 상태) */
export function entryValue(entries: EntryTrace, t: number, i: number): Float32Array {
  const d = entries.c.length / entries.n
  const partial = entries.m === 2 && entries.entryPos[i] === t
  return (partial ? entries.cPart : entries.c).subarray(i * d, (i + 1) * d)
}

function entryIndexerKey(entries: EntryTrace, t: number, i: number): Float32Array {
  const d = entries.ki.length / entries.n
  const partial = entries.m === 2 && entries.entryPos[i] === t
  return (partial ? entries.kiPart : entries.ki).subarray(i * d, (i + 1) * d)
}

export class TinyDsv41 {
  readonly config: Dsv41Config
  readonly vocab: string[]
  readonly unkToken: string
  readonly training: Dsv41ModelFile['training']
  readonly reference: Dsv41Reference[]
  private readonly stoi: Map<string, number>
  private readonly tokEmb: Float32Array
  private readonly posEmb: Float32Array
  private readonly lnfG: Float32Array
  private readonly lnfB: Float32Array
  private readonly layers: Dsv41LayerWeights[]

  constructor(file: Dsv41ModelFile) {
    if (file.schemaVersion !== 1)
      throw new Error(`지원하지 않는 모델 파일 버전: ${file.schemaVersion}`)
    this.config = file.config
    this.vocab = file.vocab
    this.unkToken = file.unkToken
    this.training = file.training
    this.reference = file.reference
    this.stoi = new Map(file.vocab.map((token, id) => [token, id]))
    this.tokEmb = decodeTensor(file.weights.tok_emb)
    this.posEmb = decodeTensor(file.weights.pos_emb)
    this.lnfG = decodeTensor(file.weights.lnf_g)
    this.lnfB = decodeTensor(file.weights.lnf_b)
    const optional = (layer: Record<string, TensorRecord>, key: string) =>
      key in layer ? decodeTensor(layer[key]) : undefined
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
      wr: decodeTensor(layer.wr),
      br: decodeTensor(layer.br),
      rbias: decodeTensor(layer.rbias),
      ws1: decodeTensor(layer.ws1),
      bs1: decodeTensor(layer.bs1),
      ws2: decodeTensor(layer.ws2),
      bs2: decodeTensor(layer.bs2),
      we1: decodeTensor(layer.we1),
      be1: decodeTensor(layer.be1),
      we2: decodeTensor(layer.we2),
      be2: decodeTensor(layer.be2),
      wc: optional(layer, 'wc'),
      bc: optional(layer, 'bc'),
      wi: optional(layer, 'wi'),
      wqi: optional(layer, 'wqi'),
      bqi: optional(layer, 'bqi'),
      ww: optional(layer, 'ww'),
      bw: optional(layer, 'bw'),
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

  layerWeights(layer: number): Dsv41LayerWeights {
    return this.layers[layer]
  }

  layerSpec(layer: number): LayerSpec {
    return this.config.layers[layer]
  }

  get tokenEmbedding(): Float32Array {
    return this.tokEmb
  }

  get positionEmbedding(): Float32Array {
    return this.posEmb
  }

  get finalNorm(): { g: Float32Array; b: Float32Array } {
    return { g: this.lnfG, b: this.lnfB }
  }

  /** prefill 에서 decoder 가 처리를 시작하는 위치: 프롬프트의 마지막 window 개 토큰만 decoder 를 지난다. */
  decoderStartFor(promptTokens: number): number {
    return Math.max(0, promptTokens - this.config.window)
  }

  /** 모든 중간값을 함께 돌려주는 순전파. decoderStart 이전 위치는 decoder 층을 지나지 않는다. */
  forwardTrace(ids: number[], decoderStart = 0): Dsv41Trace {
    const cfg = this.config
    const {
      dModel: D,
      nHeads: H,
      dLatent: DC,
      window: W,
      topK: K,
      poolBlock: B,
      poolBlocks: NB,
      indexerHeads: HI,
      dIndexer: DI,
      experts: E,
      expertsTopK: ET,
      dExpert: FE,
      dShared: FS,
      layerNormEps: eps,
    } = cfg
    const T = ids.length
    if (T === 0) throw new RangeError('토큰이 하나 이상 필요합니다.')
    if (T > cfg.context) throw new RangeError(`문맥 길이 ${cfg.context} 를 넘었습니다.`)
    if (decoderStart < 0 || decoderStart >= T) throw new RangeError('decoderStart 범위 오류')
    const HD = H * DC
    const scale = 1 / Math.sqrt(DC)

    const x = new Float32Array(T * D)
    const tokenEmbeddings = new Float32Array(T * D)
    const positionEmbeddings = new Float32Array(T * D)
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < D; j++) {
        tokenEmbeddings[i * D + j] = this.tokEmb[ids[i] * D + j]
        positionEmbeddings[i * D + j] = this.posEmb[i * D + j]
        x[i * D + j] = tokenEmbeddings[i * D + j] + positionEmbeddings[i * D + j]
      }
    }
    const x0 = x.slice()

    const entryStore = new Map<number, EntryTrace>()
    const selectionStore = new Map<number, Uint8Array>()
    let pool: Uint8Array | null = null
    const layers: Dsv41LayerTrace[] = []

    for (let l = 0; l < cfg.layers.length; l++) {
      const spec = cfg.layers[l]
      const w = this.layers[l]
      const isDecoder = l >= cfg.encoderLayers
      const start = isDecoder ? decoderStart : 0
      const hn = layerNormRows(x, T, D, w.ln1G, w.ln1B, eps)
      const q = matmulAdd(hn, T, D, w.wq, HD, w.bq)
      const kSwa = matmulAdd(hn, T, D, w.wk, HD, w.bk)
      const vSwa = matmulAdd(hn, T, D, w.wv, HD, w.bv)

      // ---- global KV 엔트리: full 층은 만들고, reuse/reindex 층은 src 층의 것을 쓴다
      let entries: EntryTrace | null = null
      if (spec.mode === 'full') {
        const m = spec.m ?? 1
        const n = m === 2 ? Math.ceil(T / 2) : T
        const merged = new Float32Array(n * m * D)
        const partialInput = new Float32Array(n * m * D)
        for (let i = 0; i < n; i++) {
          for (let slot = 0; slot < m; slot++) {
            const tokenIndex = m * i + slot
            if (tokenIndex < T) {
              merged.set(hn.subarray(tokenIndex * D, (tokenIndex + 1) * D), i * m * D + slot * D)
              if (slot === 0)
                partialInput.set(hn.subarray(tokenIndex * D, (tokenIndex + 1) * D), i * m * D)
            }
          }
        }
        const cRaw = matmulAdd(merged, n, m * D, w.wc!, DC, w.bc!)
        const cPartRaw = m === 2 ? matmulAdd(partialInput, n, m * D, w.wc!, DC, w.bc!) : cRaw
        const c = new Float32Array(n * DC)
        const cPart = new Float32Array(n * DC)
        const codes = new Int8Array(n * DC)
        const scales = new Float32Array(n * (DC / cfg.fp4Group))
        for (let i = 0; i < n; i++) {
          const stored = fp4Store(cRaw.subarray(i * DC, (i + 1) * DC), cfg.fp4Group)
          c.set(stored.dequantized, i * DC)
          codes.set(stored.codes, i * DC)
          scales.set(stored.scales, i * stored.scales.length)
          cPart.set(
            m === 2
              ? fp4Store(cPartRaw.subarray(i * DC, (i + 1) * DC), cfg.fp4Group).dequantized
              : stored.dequantized,
            i * DC,
          )
        }
        const kiRaw = matmulAdd(c, n, DC, w.wi!, DI, new Float32Array(DI))
        const kiPartRaw = m === 2 ? matmulAdd(cPart, n, DC, w.wi!, DI, new Float32Array(DI)) : kiRaw
        const ki = new Float32Array(n * DI)
        const kiPart = new Float32Array(n * DI)
        const kiScale = new Float32Array(n)
        for (let i = 0; i < n; i++) {
          const stored = mxfp4Store(kiRaw.subarray(i * DI, (i + 1) * DI))
          ki.set(stored.dequantized, i * DI)
          kiScale[i] = stored.scale
          kiPart.set(
            m === 2
              ? mxfp4Store(kiPartRaw.subarray(i * DI, (i + 1) * DI)).dequantized
              : stored.dequantized,
            i * DI,
          )
        }
        entries = {
          layer: l,
          m,
          n,
          entryPos: Array.from({ length: n }, (_, i) => m * i),
          merged,
          cRaw,
          cPartRaw,
          c,
          cPart,
          codes,
          scales,
          kiRaw,
          ki,
          kiPart,
          kiScale,
        }
        entryStore.set(l, entries)
      } else if (spec.mode !== 'swa') {
        entries = entryStore.get(spec.src!) ?? null
        if (!entries) throw new Error(`층 ${l} 이 참조하는 KV 소스 층 ${spec.src} 이 없습니다.`)
      }
      const n = entries ? entries.n : 0

      // ---- 인덱서 점수, 후보 풀, Top-K
      let qi: Float32Array | null = null
      let wIdx: Float32Array | null = null
      let scores: Float32Array | null = null
      let range: Uint8Array | null = null
      let poolThis: Uint8Array | null = null
      let selected: Uint8Array | null = null
      let selectedFrom: number | null = null
      if (entries && (spec.mode === 'full' || spec.mode === 'reindex')) {
        qi = matmulAdd(hn, T, D, w.wqi!, HI * DI, w.bqi!)
        wIdx = matmulAdd(hn, T, D, w.ww!, HI, w.bw!)
        scores = new Float32Array(T * n)
        range = new Uint8Array(T * n)
        selected = new Uint8Array(T * n)
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < n; j++) {
            const visible = entries.entryPos[j] <= t
            const inRange = spec.mode === 'full' ? visible : visible && pool![t * n + j] === 1
            if (!inRange) continue
            range[t * n + j] = 1
            const key = entryIndexerKey(entries, t, j)
            let s = 0
            for (let h = 0; h < HI; h++) {
              let dot = 0
              for (let d = 0; d < DI; d++) dot += qi[t * HI * DI + h * DI + d] * key[d]
              s += wIdx[t * HI + h] * Math.max(0, dot)
            }
            scores[t * n + j] = s - 1e-6 * j // 동점은 앞 엔트리 우선
          }
        }
        if (spec.mode === 'full' && isDecoder) {
          // 계층적 인덱서: 블록 최대 점수로 후보 풀을 만든다. 이후 reindex 층은 이 풀 안에서만 고른다.
          poolThis = new Uint8Array(T * n)
          const nb = Math.ceil(n / B)
          for (let t = 0; t < T; t++) {
            const blockMax = new Array<number>(nb).fill(NEG)
            for (let j = 0; j < n; j++)
              if (range[t * n + j])
                blockMax[Math.floor(j / B)] = Math.max(
                  blockMax[Math.floor(j / B)],
                  scores[t * n + j],
                )
            const candidates = blockMax.flatMap((v, b) => (v > NEG ? [b] : []))
            for (const b of topIndices(blockMax, candidates, Math.min(NB, nb)))
              for (let j = b * B; j < Math.min(n, (b + 1) * B); j++)
                if (range[t * n + j]) poolThis[t * n + j] = 1
          }
          pool = poolThis
        }
        for (let t = 0; t < T; t++) {
          const candidates: number[] = []
          for (let j = 0; j < n; j++) if (range[t * n + j]) candidates.push(j)
          const rowScores = scores.subarray(t * n, (t + 1) * n)
          for (const j of topIndices(rowScores, candidates, Math.min(K, n))) selected[t * n + j] = 1
        }
        selectionStore.set(l, selected)
      } else if (spec.mode === 'reuse') {
        selected = selectionStore.get(spec.idx!) ?? null
        if (!selected) throw new Error(`층 ${l} 이 참조하는 Top-K 층 ${spec.idx} 이 없습니다.`)
        selectedFrom = spec.idx!
      }

      // ---- global 로짓과 SWA 로짓을 하나의 softmax 로 병합
      const attnGlobal = new Float32Array(H * T * n)
      const attnLocal = new Float32Array(H * T * T)
      const logitGlobal = new Float32Array(H * T * n).fill(NEG)
      const logitLocal = new Float32Array(H * T * T).fill(NEG)
      const o = new Float32Array(T * HD)
      for (let t = start; t < T; t++) {
        for (let h = 0; h < H; h++) {
          const qv = q.subarray(t * HD + h * DC, t * HD + (h + 1) * DC)
          const globalIndex: number[] = []
          const localIndex: number[] = []
          const logits: number[] = []
          if (entries && selected) {
            for (let j = 0; j < n; j++) {
              if (!selected[t * n + j]) continue
              const cv = entryValue(entries, t, j)
              let dot = 0
              for (let d = 0; d < DC; d++) dot += qv[d] * cv[d]
              logitGlobal[(h * T + t) * n + j] = dot * scale
              globalIndex.push(j)
              logits.push(dot * scale)
            }
          }
          for (let s = Math.max(start, t - W + 1); s <= t; s++) {
            let dot = 0
            for (let d = 0; d < DC; d++) dot += qv[d] * kSwa[s * HD + h * DC + d]
            logitLocal[(h * T + t) * T + s] = dot * scale
            localIndex.push(s)
            logits.push(dot * scale)
          }
          const weights = softmax(logits)
          const out = o.subarray(t * HD + h * DC, t * HD + (h + 1) * DC)
          globalIndex.forEach((j, index) => {
            const weight = weights[index]
            attnGlobal[(h * T + t) * n + j] = weight
            const cv = entryValue(entries!, t, j)
            for (let d = 0; d < DC; d++) out[d] += weight * cv[d]
          })
          localIndex.forEach((s, index) => {
            const weight = weights[globalIndex.length + index]
            attnLocal[(h * T + t) * T + s] = weight
            for (let d = 0; d < DC; d++) out[d] += weight * vSwa[s * HD + h * DC + d]
          })
        }
      }
      const y = matmulAdd(o, T, HD, w.wo, D, w.bo)
      const x1 = x.slice()
      for (let t = start; t < T; t++) for (let d = 0; d < D; d++) x1[t * D + d] += y[t * D + d]

      // ---- MoE: shared 전문가 + routed 전문가 Top-2 (sigmoid 게이트, 선택에만 편향 적용)
      const ln2 = layerNormRows(x1, T, D, w.ln2G, w.ln2B, eps)
      const router = new Float32Array(T * E)
      const expertMask = new Uint8Array(T * E)
      const gates = new Float32Array(T * E)
      const shared = new Float32Array(T * D)
      const expertOut = new Float32Array(T * E * D)
      const routed = new Float32Array(T * D)
      const x2 = x1.slice()
      for (let t = start; t < T; t++) {
        const h2 = ln2.subarray(t * D, (t + 1) * D)
        const biased: number[] = []
        for (let e = 0; e < E; e++) {
          let z = w.br[e]
          for (let d = 0; d < D; d++) z += h2[d] * w.wr[d * E + e]
          router[t * E + e] = sigmoid(z)
          biased.push(router[t * E + e] + w.rbias[e])
        }
        const chosen = topIndices(
          biased,
          Array.from({ length: E }, (_, e) => e),
          ET,
        )
        let total = 0
        for (const e of chosen) {
          expertMask[t * E + e] = 1
          total += router[t * E + e]
        }
        for (const e of chosen) gates[t * E + e] = router[t * E + e] / total
        const hiddenShared = matmulAdd(h2, 1, D, w.ws1, FS, w.bs1)
        for (let i = 0; i < FS; i++) if (hiddenShared[i] < 0) hiddenShared[i] = 0
        shared.set(matmulAdd(hiddenShared, 1, FS, w.ws2, D, w.bs2), t * D)
        for (let e = 0; e < E; e++) {
          const hidden = matmulAdd(
            h2,
            1,
            D,
            w.we1.subarray(e * D * FE, (e + 1) * D * FE),
            FE,
            w.be1.subarray(e * FE, (e + 1) * FE),
          )
          for (let i = 0; i < FE; i++) if (hidden[i] < 0) hidden[i] = 0
          const out = matmulAdd(
            hidden,
            1,
            FE,
            w.we2.subarray(e * FE * D, (e + 1) * FE * D),
            D,
            w.be2.subarray(e * D, (e + 1) * D),
          )
          expertOut.set(out, (t * E + e) * D)
          const gate = gates[t * E + e]
          if (gate > 0) for (let d = 0; d < D; d++) routed[t * D + d] += gate * out[d]
        }
        for (let d = 0; d < D; d++) x2[t * D + d] += shared[t * D + d] + routed[t * D + d]
      }
      x.set(x2)

      layers.push({
        mode: spec.mode,
        start,
        ln1: hn,
        q,
        kSwa,
        vSwa,
        entries,
        n,
        qi,
        wIdx,
        scores,
        range,
        pool: poolThis,
        selected,
        selectedFrom,
        attnGlobal,
        attnLocal,
        logitGlobal,
        logitLocal,
        o,
        y,
        x1,
        ln2,
        router,
        expertMask,
        gates,
        shared,
        expertOut,
        routed,
        x2,
      })
    }

    const final = layerNormRows(x, T, D, this.lnfG, this.lnfB, eps)
    const last = (T - 1) * D
    const logits = new Float32Array(this.vocab.length)
    for (let v = 0; v < this.vocab.length; v++) {
      let sum = 0
      for (let j = 0; j < D; j++) sum += final[last + j] * this.tokEmb[v * D + j]
      logits[v] = sum
    }
    return {
      ids,
      tokens: ids.map((id) => this.vocab[id]),
      decoderStart,
      tokenEmbeddings,
      positionEmbeddings,
      x0,
      layers,
      final,
      logits,
    }
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

  /** 프롬프트 뒤에 글자를 이어 쓴다. decoder 시작 위치는 프롬프트 길이로 정해 두고 유지한다. */
  generate(
    text: string,
    steps: number,
    temperature = 1,
    random: () => number = Math.random,
  ): string {
    const ids = this.encode(text)
    if (ids.length === 0) return text
    const decoderStart = this.decoderStartFor(ids.length)
    let current = text
    for (let step = 0; step < steps; step++) {
      if (ids.length >= this.config.context) break
      const distribution = this.nextTokenDistribution(
        this.forwardTrace(ids, decoderStart).logits,
        temperature,
      )
      const pick =
        distribution[
          sampleIndex(
            distribution.map((c) => c.probability),
            random,
          )
        ]
      ids.push(pick.id)
      current += pick.token
    }
    return current
  }
}

let cached: TinyDsv41 | null = null

/** 번들에 포함된 학습 결과 파일로 모델을 만든다 (한 번만 디코딩). */
export function loadTinyDsv41(): TinyDsv41 {
  if (!cached) cached = new TinyDsv41(modelFile as unknown as Dsv41ModelFile)
  return cached
}
