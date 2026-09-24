import type { LayerMode } from '../lib/tinyDsv41'

/** DCv4.1 시뮬레이터의 단계 순서. 화면의 요소는 data-stage 로 이 id 에 연결된다. */
export type StageId =
  | 'tokens'
  | 'embed'
  | 'ln1'
  | 'q'
  | 'swa'
  | 'compress'
  | 'fp4'
  | 'indexerK'
  | 'indexerQ'
  | 'scores'
  | 'pool'
  | 'topk'
  | 'gather'
  | 'logits'
  | 'merge'
  | 'weighted'
  | 'wo'
  | 'add1'
  | 'ln2'
  | 'router'
  | 'experts'
  | 'add2'
  | 'lnf'
  | 'outLogits'
  | 'softmax'
  | 'sample'

export type StageGroup =
  'input' | 'embed' | 'query' | 'kv' | 'select' | 'attention' | 'moe' | 'output'

export const STAGES: StageId[] = [
  'tokens',
  'embed',
  'ln1',
  'q',
  'swa',
  'compress',
  'fp4',
  'indexerK',
  'indexerQ',
  'scores',
  'pool',
  'topk',
  'gather',
  'logits',
  'merge',
  'weighted',
  'wo',
  'add1',
  'ln2',
  'router',
  'experts',
  'add2',
  'lnf',
  'outLogits',
  'softmax',
  'sample',
]

export const STAGE_GROUP: Record<StageId, StageGroup> = {
  tokens: 'input',
  embed: 'embed',
  ln1: 'query',
  q: 'query',
  swa: 'query',
  compress: 'kv',
  fp4: 'kv',
  indexerK: 'kv',
  indexerQ: 'select',
  scores: 'select',
  pool: 'select',
  topk: 'select',
  gather: 'attention',
  logits: 'attention',
  merge: 'attention',
  weighted: 'attention',
  wo: 'attention',
  add1: 'attention',
  ln2: 'moe',
  router: 'moe',
  experts: 'moe',
  add2: 'moe',
  lnf: 'output',
  outLogits: 'output',
  softmax: 'output',
  sample: 'output',
}

/**
 * 단계별 수식. 중괄호 자리는 formatFormula 가 실제 값으로 바꾼다.
 * T 토큰 수, D d_model, H 헤드 수, dc latent 차원, HD H·dc, n 엔트리 수, k Top-K, w 윈도우, m 압축비,
 * G scale 그룹 수, mainB/idxB 엔트리 바이트, DI/HI 인덱서 차원·헤드, NB/B 후보 풀 블록 수·크기,
 * E/ET 전문가 수·활성 수, V 어휘 수.
 */
export const STAGE_FORMULA: Record<StageId, string> = {
  tokens: 'text → ids   [{T}]',
  embed: 'x = E[id] + P[pos]   [{T}×{D}]',
  ln1: 'h = LN₁(x)   [{T}×{D}]',
  q: 'q_t = W_Q·h_t → {H} heads × {dc}   [{T}×{HD}]  (Main Q, every layer)',
  swa: 'k_t = W_K·h_t,  v_t = W_V·h_t → window w = {w}   [{T}×{HD}]  (SWA, every layer)',
  compress: 'c_j = W_c·[h_{mj} ; … ; h_{mj+m−1}] + b_c   [{n}×{dc}],  m = {m}',
  fp4: 'C_j = FP4(c_j) + scale   {dc}×4 bit + {G}×E4M3 = {mainB} B / entry',
  indexerK: 'K^I_j = MXFP4(W_I·C_j)   [{n}×{DI}],  {idxB} B / entry',
  indexerQ: 'q^I_t = W^I_Q·h_t → {HI} heads × {DI},   w_t = W_w·h_t + b_w',
  scores: 'S_{t,j} = Σ_h w_{t,h} · ReLU(q^I_{t,h} · K^I_j),   j in range',
  pool: 'P_t = entries of the top-{NB} blocks (size {B}) by max_j S_{t,j}   |P_t| ≤ {poolSize}',
  topk: 'T_t = TopK_k(S_{t,·}),  k = {k}',
  gather: 'c_j = dequant(C_j),  j ∈ T_t   (≤ {k} × {mainB} B read)',
  logits: 'e_j = q_t·c_j / √{dc}  (j ∈ T_t),   e_s = q_t·k_s / √{dc}  (t−{w} < s ≤ t)',
  merge: 'a = softmax([e_j]_{j∈T_t} , [e_s]_{s∈W_t})   length ≤ {k} + {w}',
  weighted: 'o_t = Σ_j a_j c_j + Σ_s a_s v_s   per head, concat → [{HD}]',
  wo: 'y_t = W_O·o_t + b_O   [{HD}] → [{D}]',
  add1: "x' = x + y   [{T}×{D}]",
  ln2: "h = LN₂(x')   [{T}×{D}]",
  router: 'r = σ(W_r·h + b_r),  pick top-{ET} of {E} by r + bias,  g_e = r_e / Σ r',
  experts: 'ffn = shared(h) + Σ_e g_e · expert_e(h),   expert(h) = ReLU(hW₁ + b₁)W₂ + b₂',
  add2: "x'' = x' + ffn   [{T}×{D}]  → next layer",
  lnf: "h_T = LN_f(x''_T)   [1×{D}]",
  outLogits: 'z = h_T · Eᵀ   [1×{D}]·[{D}×{V}]',
  softmax: 'p = softmax(z / temperature)   [{V}]',
  sample: 'next = sample(p) → append → decode',
}

export type FormulaValues = Record<string, number | string>

export function formatFormula(id: StageId, values: FormulaValues): string {
  return STAGE_FORMULA[id].replace(/\{([A-Za-z]+)\}/g, (match, key: string) =>
    key in values ? String(values[key]) : match,
  )
}

/** 어떤 층에서 이 단계가 실제로 계산되는지(compute), 앞 층 것을 물려받는지(inherit), 해당 없음인지(none) */
export type StageState = 'compute' | 'inherit' | 'none'

export function stageState(id: StageId, mode: LayerMode, isDecoder: boolean): StageState {
  switch (id) {
    case 'compress':
    case 'fp4':
    case 'indexerK':
      return mode === 'full' ? 'compute' : mode === 'swa' ? 'none' : 'inherit'
    case 'indexerQ':
    case 'scores':
      return mode === 'full' || mode === 'reindex' ? 'compute' : 'none'
    case 'pool':
      if (mode === 'full') return isDecoder ? 'compute' : 'none'
      return mode === 'reindex' ? 'inherit' : 'none'
    case 'topk':
      return mode === 'full' || mode === 'reindex'
        ? 'compute'
        : mode === 'reuse'
          ? 'inherit'
          : 'none'
    case 'gather':
      return mode === 'swa' ? 'none' : 'compute'
    default:
      return 'compute'
  }
}
