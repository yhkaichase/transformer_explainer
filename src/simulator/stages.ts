/** 시뮬레이터의 단계 순서. 화면의 요소는 data-stage 로 이 id 에 연결된다. */
export type StageId =
  | 'tokens'
  | 'embed'
  | 'ln1'
  | 'qkv'
  | 'heads'
  | 'scores'
  | 'weighted'
  | 'wo'
  | 'add1'
  | 'ln2'
  | 'ffn1'
  | 'relu'
  | 'ffn2'
  | 'add2'
  | 'lnf'
  | 'logits'
  | 'softmax'
  | 'sample'

export type StageGroup = 'input' | 'embed' | 'attention' | 'ffn' | 'output'

export const STAGES: StageId[] = [
  'tokens',
  'embed',
  'ln1',
  'qkv',
  'heads',
  'scores',
  'weighted',
  'wo',
  'add1',
  'ln2',
  'ffn1',
  'relu',
  'ffn2',
  'add2',
  'lnf',
  'logits',
  'softmax',
  'sample',
]

export const STAGE_GROUP: Record<StageId, StageGroup> = {
  tokens: 'input',
  embed: 'embed',
  ln1: 'attention',
  qkv: 'attention',
  heads: 'attention',
  scores: 'attention',
  weighted: 'attention',
  wo: 'attention',
  add1: 'attention',
  ln2: 'ffn',
  ffn1: 'ffn',
  relu: 'ffn',
  ffn2: 'ffn',
  add2: 'ffn',
  lnf: 'output',
  logits: 'output',
  softmax: 'output',
  sample: 'output',
}

/** 단계별 수식. {T} 는 토큰 수, {D} 는 d_model, {F} 는 d_ff, {H} 는 헤드 수, {dh} 는 헤드 차원, {V} 는 어휘 수로 바뀐다. */
export const STAGE_FORMULA: Record<StageId, string> = {
  tokens: 'text → ids   [{T}]',
  embed: 'x = E[id] + P[pos]   [{T}×{D}]',
  ln1: 'h = LN₁(x)   [{T}×{D}]',
  qkv: 'Q = hW_Q + b_Q,  K = hW_K + b_K,  V = hW_V + b_V   [{T}×{D}]·[{D}×{D}]',
  heads: 'Q, K, V → {H} heads × {dh}',
  scores: 'A_h = softmax(Q_h K_hᵀ / √{dh} + mask)   [{T}×{T}] per head',
  weighted: 'head_h = A_h V_h,  concat   [{T}×{D}]',
  wo: 'attn = concat·W_O + b_O   [{T}×{D}]·[{D}×{D}]',
  add1: "x' = x + attn   [{T}×{D}]",
  ln2: "h = LN₂(x')   [{T}×{D}]",
  ffn1: 'u = hW₁ + b₁   [{T}×{D}]·[{D}×{F}]',
  relu: 'a = max(0, u)   [{T}×{F}]',
  ffn2: 'ffn = aW₂ + b₂   [{T}×{F}]·[{F}×{D}]',
  add2: "x'' = x' + ffn   [{T}×{D}]  → next layer",
  lnf: "h_T = LN_f(x''_T)   [1×{D}]",
  logits: 'z = h_T · Eᵀ   [1×{D}]·[{D}×{V}]',
  softmax: 'p = softmax(z / temperature)   [{V}]',
  sample: 'next = sample(p) → append → decode',
}

export function formatFormula(
  id: StageId,
  values: { T: number; D: number; F: number; H: number; dh: number; V: number },
): string {
  return STAGE_FORMULA[id].replace(/\{(T|D|F|H|dh|V)\}/g, (_, key: keyof typeof values) =>
    String(values[key]),
  )
}
