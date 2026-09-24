import type { LayerMode } from '../lib/tinyDsv41'
import type { StageGroup, StageId } from './stages'

export type SimLang = 'ko' | 'en'

export interface Dsv41Labels {
  title: string
  author: string
  subtitle: string
  prompt: string
  promptHint: (max: number) => string
  prefill: (encoderTokens: number, decoderTokens: number) => string
  decode: (step: number, cached: number) => string
  decodeOne: string
  decodeTen: string
  reset: string
  temperature: string
  layer: string
  head: string
  layerOption: (i: number, mode: string) => string
  headOption: (i: number) => string
  focus: string
  autoplay: string
  autoplayStop: string
  speed: string
  loopDecode: string
  result: string
  generatedText: string
  nextToken: string
  nextTop: string
  prev: string
  next: string
  stageNames: Record<StageId, string>
  groups: Record<StageGroup, string>
  modeTitles: Record<LayerMode, string>
  modeNotes: Record<LayerMode, string>
  encoder: string
  decoder: string
  ced: (fromLayer: number) => string
  kvGroup: (index: number, layer: number, m: number) => string
  reindexNote: string
  panels: {
    tokens: string
    embedding: string
    layerMap: string
    block: (layer: number, mode: string) => string
    query: string
    kv: string
    select: string
    attention: string
    moe: string
    output: string
    cache: string
  }
  encOnly: string
  newToken: string
  cache: string
  inherited: (layer: number) => string
  stateNote: { inherit: (from: number) => string; none: string }
  window: string
  partial: string
  topK: string
  probability: string
  focusTitle: (position: number, token: string) => string
  detail: Record<StageId, string>
  cacheCard: {
    title: string
    entries: (layer: number, m: number, n: number, bytes: number) => string
    perToken: (bytes: number) => string
    formula: (main: number, idx: number, terms: string, total: number) => string
    real: string
    readsTitle: string
    reads: (indexer: number, main: number, swa: number) => string
    realReads: string
  }
  modelNote: (params: number, layers: number, vocab: number) => string
  simplificationsTitle: string
  simplifications: string[]
  sources: string
  empty: string
  contextFull: (max: number) => string
}

export const REAL_MODEL = {
  layers: 40,
  encoder: 20,
  full: [2, 8, 14, 20],
  reindex: [24, 28, 32, 36],
  heads: 64,
  latent: 512,
  indexerDim: 128,
  topK: 512,
  window: 128,
  poolBlocks: 2048,
  poolBlock: 8,
  mainEntryBytes: 288,
  indexerEntryBytes: 68,
  bytesPerToken: 890,
  experts: 384,
  expertsActive: 6,
}

export const LABELS: Record<SimLang, Dsv41Labels> = {
  ko: {
    title: 'DeepSeek-V4.1-Flash Simulator',
    author: '제작자: 최영하',
    subtitle:
      '매뉴얼(DeepseekV4.1_manual.docx)이 설명하는 V4.1-Flash 의 구조를 아주 작게 축소한 모델이 페이지 안에서 실제로 계산합니다. CED 인코더·디코더, CSA2 세 모드, 인덱서 Top-K, SWA 병합 softmax, FP4 KV, MoE 를 단계별로 따라가 보세요.',
    prompt: '프롬프트',
    promptHint: (max) => `최대 ${max}글자. 글자 하나가 토큰 하나입니다.`,
    prefill: (encoderTokens, decoderTokens) =>
      `Prefill · encoder 는 ${encoderTokens}개 토큰 전부, decoder 는 마지막 ${decoderTokens}개만 계산`,
    decode: (step, cached) =>
      `Decode ${step} · 새 토큰 1개가 모든 층을 지나고, 캐시 ${cached}개 토큰 몫은 재사용`,
    decodeOne: 'Decode: 다음 토큰',
    decodeTen: 'Decode ×10',
    reset: '처음으로',
    temperature: '온도',
    layer: '층',
    head: '헤드',
    layerOption: (i, mode) => `L${i} · ${mode}`,
    headOption: (i) => `헤드 ${i + 1}`,
    focus: '초점 토큰',
    autoplay: '자동 재생',
    autoplayStop: '정지',
    speed: '배속',
    loopDecode: '마지막 단계 뒤 다음 토큰 이어 쓰기',
    result: '결과',
    generatedText: '생성된 글 (파란 부분이 모델이 이어 쓴 것)',
    nextToken: '다음 토큰',
    nextTop: '다음 토큰 후보 상위 5개',
    prev: '이전 단계',
    next: '다음 단계',
    stageNames: {
      tokens: '토큰화',
      embed: '임베딩 + 위치',
      ln1: 'LayerNorm₁',
      q: 'Main Q',
      swa: 'SWA K·V 기록',
      compress: 'KV 엔트리 생성',
      fp4: 'FP4 저장',
      indexerK: 'Indexer K',
      indexerQ: 'Indexer Q',
      scores: '인덱서 점수',
      pool: '후보 풀',
      topk: 'Top-K 선택',
      gather: 'main KV gather',
      logits: 'global·local 로짓',
      merge: '병합 softmax',
      weighted: '가중합',
      wo: 'W_O',
      add1: '잔차 +',
      ln2: 'LayerNorm₂',
      router: 'MoE 라우터',
      experts: '전문가 계산',
      add2: '잔차 +',
      lnf: 'LayerNorm_f',
      outLogits: '로짓 · Eᵀ',
      softmax: 'softmax',
      sample: '다음 토큰',
    },
    groups: {
      input: '입력',
      embed: '임베딩',
      query: 'Query · SWA',
      kv: 'global KV',
      select: '주소 정하기',
      attention: '어텐션',
      moe: 'MoE',
      output: '출력',
    },
    modeTitles: { swa: 'SWA 전용', full: 'Full', reuse: 'Reuse', reindex: 'Reindex' },
    modeNotes: {
      swa: 'global KV 없이 최근 윈도우만 봅니다.',
      full: 'main KV 와 indexer K 를 만들고 Top-K 를 고릅니다.',
      reuse: 'KV 와 Top-K 를 모두 물려받고 query 와 SWA 만 새로 계산합니다.',
      reindex: 'KV 는 물려받고, 후보 풀 안에서 Top-K 만 자기 indexer Q 로 다시 고릅니다.',
    },
    encoder: 'Encoder (causal)',
    decoder: 'Decoder',
    ced: (fromLayer) => `L${fromLayer} 출력에서 투영 (CED)`,
    kvGroup: (index, layer, m) => `KV ${index} · L${layer} 생성 · 압축비 ${m}`,
    reindexNote: 'Reindex 층은 후보 풀 안에서 Top-K 만 재선택',
    panels: {
      tokens: '토큰',
      embedding: '임베딩',
      layerMap: '층 배치 (CSA2 모드)',
      block: (layer, mode) => `L${layer} · ${mode}`,
      query: 'Query 와 SWA K·V (모든 층이 새로 계산)',
      kv: 'global KV 엔트리',
      select: '인덱서와 Top-K (읽을 주소 정하기)',
      attention: '두 분기 어텐션 → 하나의 softmax',
      moe: 'MoE 피드포워드',
      output: '출력',
      cache: 'KV cache',
    },
    encOnly: 'enc만',
    newToken: '새 토큰',
    cache: '캐시',
    inherited: (layer) => `L${layer} 의 것을 재사용`,
    stateNote: {
      inherit: (from) => `이 층은 이 값을 새로 만들지 않고 L${from} 의 것을 그대로 씁니다.`,
      none: '이 층에서는 이 단계가 없습니다.',
    },
    window: '윈도우',
    partial: '짝이 안 온 엔트리 (부분 상태)',
    topK: '상위 10개',
    probability: '확률',
    focusTitle: (position, token) => `초점: 위치 ${position} "${token}"`,
    detail: {
      tokens:
        '글자를 어휘 사전에서 찾아 번호(id)로 바꿉니다. prefill 에서는 프롬프트 토큰 전부가 encoder 층을 지나고, decoder 층은 마지막 윈도우 안의 토큰만 지납니다.',
      embed: 'E 의 id 번째 행과 P 의 pos 번째 행을 더합니다.',
      ln1: '행마다 평균을 빼고 표준편차로 나눈 뒤, 학습된 g 와 b 를 곱하고 더합니다.',
      q: '모든 층이 자기 hidden 에서 Main Q 를 새로 만듭니다. 헤드마다 latent 차원의 질의가 나오고, 이 질의가 공유 latent c_j 와 직접 내적합니다 (K 를 따로 만들지 않음).',
      swa: '이 토큰의 SWA 키와 값을 만들어 윈도우에 기록합니다. 윈도우는 최근 w 개 토큰뿐이므로 컨텍스트 길이와 무관한 상수 크기입니다.',
      compress:
        'Full 층만 global KV 를 만듭니다. encoder 는 인접 두 토큰의 hidden 을 이어 붙여 W_c 로 하나의 latent 로 합치고(압축비 2), decoder 는 encoder 마지막 출력에서 토큰마다 하나씩 투영합니다(압축비 1, CED). 짝이 아직 안 온 토큰은 [h; 0] 으로 만든 부분 엔트리를 씁니다.',
      fp4: 'latent 를 FP4(E2M1) 코드와 채널 그룹별 E4M3 scale 로 저장합니다. 어텐션 직전에 dequantize 하므로 저장 용량만 줄고 계산 정밀도는 유지됩니다.',
      indexerK:
        '저장된 엔트리를 작은 차원으로 투영해 indexer K 를 만들고 MXFP4 로 저장합니다. 이 값은 어떤 엔트리를 읽을지 정하는 데만 쓰입니다.',
      indexerQ:
        'Full 층과 Reindex 층만 Indexer Q 와 헤드 가중치 w 를 만듭니다. Reuse 층은 인덱서에 접근하지 않습니다.',
      scores:
        '초점 토큰의 Indexer Q 와 엔트리의 indexer K 를 헤드마다 내적해 ReLU 하고 w 로 가중합합니다. Full 층은 보이는 엔트리 전체, Reindex 층은 후보 풀 안에서만 계산합니다. 이 점수는 주소를 정하는 데서 역할이 끝납니다.',
      pool: 'decoder 의 첫 Full 층이 블록마다 최대 점수를 보고 상위 블록만 후보 풀로 남깁니다. 뒤의 Reindex 층은 이 풀 안에서만 점수를 매기므로 인덱서 비용이 컨텍스트 길이와 무관해집니다.',
      topk: '점수 상위 k 개 엔트리를 고릅니다. 고르지 않은 엔트리는 로짓 자체가 계산되지 않아 확률이 0 입니다. Reuse 층은 앞 층의 선택을 그대로 씁니다.',
      gather:
        '고른 엔트리의 main KV 만 가져와 FP4 에서 되돌립니다. 컨텍스트가 길어도 층 하나가 읽는 엔트리 수는 k 로 고정입니다.',
      logits:
        '같은 q_t 로 global 로짓(선택된 c_j)과 local 로짓(윈도우의 k_s)을 만듭니다. c_j 하나가 key 와 value 를 겸합니다.',
      merge:
        '두 로짓을 이어 붙여 softmax 하나로 만듭니다. 그래서 한 층이 실제로 보는 위치는 k + w 개를 넘지 않습니다.',
      weighted:
        'global 쪽은 c_j 를 value 로, SWA 쪽은 자기 v_s 를 value 로 가중합해 헤드 출력을 만들고 헤드를 이어 붙입니다.',
      wo: '이어 붙인 벡터에 W_O 를 곱해 어텐션 출력을 만듭니다.',
      add1: '어텐션 출력을 원래 x 에 더합니다 (잔차 연결).',
      ln2: 'MoE 전에 다시 정규화합니다.',
      router:
        '라우터가 전문가마다 sigmoid 점수를 내고, 점수에 부하 균형용 편향을 더한 값으로 상위 전문가를 고릅니다. 게이트 값은 편향 없이 점수를 정규화한 것입니다.',
      experts:
        'shared 전문가는 항상 계산하고, 고른 routed 전문가만 계산해 게이트로 섞습니다. 나머지 전문가는 계산하지 않습니다.',
      add2: 'MoE 출력을 더합니다. 이것이 이 층의 출력이고 다음 층의 입력입니다.',
      lnf: '마지막 층 출력의 마지막 위치를 정규화합니다.',
      outLogits: '임베딩 행렬 E 를 거꾸로 써서 어휘마다 점수를 냅니다 (가중치 공유).',
      softmax: '점수를 온도로 나눈 뒤 softmax 로 확률을 만듭니다.',
      sample:
        '확률에 따라 토큰 하나를 뽑아 뒤에 붙입니다. 다음 스텝은 새 토큰 1개만 모든 층을 지나고, 나머지는 캐시를 읽습니다 (decode).',
    },
    cacheCard: {
      title: 'KV cache · 지금 상태',
      entries: (layer, m, n, bytes) => `KV L${layer} (압축비 ${m}): 엔트리 ${n}개 × (${bytes} B)`,
      perToken: (bytes) => `토큰당 global KV: ${bytes} B`,
      formula: (main, idx, terms, total) => `= (${main} + ${idx}) × (${terms}) = ${total} B/token`,
      real: `실제 V4.1-Flash: (${REAL_MODEL.mainEntryBytes} + ${REAL_MODEL.indexerEntryBytes}) × (3 × ½ + 1) = ${REAL_MODEL.bytesPerToken} B/token`,
      readsTitle: '마지막 토큰이 읽은 global 쪽 엔트리',
      reads: (indexer, main, swa) =>
        `indexer K 스캔 ${indexer}개 · main KV gather ${main}개 · SWA 읽기 ${swa}개`,
      realReads:
        '실제 1M 컨텍스트 decode 1스텝: indexer K 174 MB, main KV 5.6 MB, SWA KV 3.0 MB (매뉴얼 추정)',
    },
    modelNote: (params, layers, vocab) =>
      `모델: 문자 단위, ${layers}층(encoder 5 + decoder 5), 파라미터 ${params.toLocaleString()}개, 어휘 ${vocab}. 이 저장소의 scripts/train_tiny_dsv41.py 가 페이지 본문과 매뉴얼 텍스트로 학습. 값은 모두 실제 계산 결과입니다.`,
    simplificationsTitle: '실제 모델과 다른 점 (축소·단순화)',
    simplifications: [
      `크기: 10층(실제 40층 = encoder 20 + decoder 20), 헤드 4개 × latent 16(실제 64 × 512), 인덱서 2 × 8(실제 128차원), Top-4(실제 512), 윈도우 8(실제 128), 후보 풀 2블록 × 4(실제 2,048 × 8), MoE shared 1 + routed 2/8(실제 1 + 6/384), FP4 scale 그룹 8채널(실제 16). Full 층은 L1·L3·L5(실제 L2·L8·L14·L20), Reindex 는 L7·L9(실제 L24·L28·L32·L36), SWA 전용은 L0(실제 L0·L1).`,
      '토큰은 글자 하나이고, 위치는 학습된 절대 위치 임베딩입니다 (실제 모델은 RoPE 를 적용한 뒤 양자화).',
      '인덱서 점수 함수(헤드별 ReLU 가중합)와 SWA 가 global 과 같은 query 를 쓰는 점은 매뉴얼이 가정한 형태를 그대로 따랐습니다.',
      'MoE 라우터의 sigmoid 게이트와 선택용 편향(부하 균형)은 매뉴얼에 없는 구현 세부입니다.',
      'prefill 에서 decoder 가 마지막 윈도우 토큰만 처리하는 근사는 매뉴얼 설명대로 구현했습니다. DSpark 초안 생성, persistent KV(SSD 보관)는 시뮬레이션하지 않습니다.',
      'FP4 저장은 학습 후반의 QAT 로 적응시켰습니다. 위 수치 중 실제 모델 값은 모두 매뉴얼에서 가져왔습니다.',
    ],
    sources:
      '출처: 저장소의 DeepseekV4.1_manual.docx (2026-09-21) — 매뉴얼이 인용한 원 출처는 DeepSeek-V4.1-Flash 테크 리포트(arXiv 2609.19969)와 Hugging Face 모델 카드입니다.',
    empty: '글자를 한 개 이상 입력하세요.',
    contextFull: (max) => `문맥 길이 ${max} 에 도달했습니다.`,
  },
  en: {
    title: 'DeepSeek-V4.1-Flash Simulator',
    author: 'Author: 최영하',
    subtitle:
      'A toy-scale model with the V4.1-Flash structure described in the manual (DeepseekV4.1_manual.docx) computes for real inside this page. Step through the CED encoder/decoder, the three CSA2 modes, indexer Top-K, the merged SWA softmax, FP4 KV storage, and MoE.',
    prompt: 'Prompt',
    promptHint: (max) => `Up to ${max} characters. One character is one token.`,
    prefill: (encoderTokens, decoderTokens) =>
      `Prefill · encoder runs on all ${encoderTokens} tokens, decoder only on the last ${decoderTokens}`,
    decode: (step, cached) =>
      `Decode ${step} · one new token passes every layer, cache for ${cached} tokens reused`,
    decodeOne: 'Decode: next token',
    decodeTen: 'Decode ×10',
    reset: 'Reset',
    temperature: 'Temperature',
    layer: 'Layer',
    head: 'Head',
    layerOption: (i, mode) => `L${i} · ${mode}`,
    headOption: (i) => `Head ${i + 1}`,
    focus: 'Focus token',
    autoplay: 'Autoplay',
    autoplayStop: 'Stop',
    speed: 'Speed',
    loopDecode: 'Decode a token after the last stage',
    result: 'Output',
    generatedText: 'Generated text (blue part written by the model)',
    nextToken: 'Next token',
    nextTop: 'Top 5 next-token candidates',
    prev: 'Previous stage',
    next: 'Next stage',
    stageNames: {
      tokens: 'Tokenize',
      embed: 'Embedding + position',
      ln1: 'LayerNorm₁',
      q: 'Main Q',
      swa: 'SWA K·V write',
      compress: 'KV entry build',
      fp4: 'FP4 store',
      indexerK: 'Indexer K',
      indexerQ: 'Indexer Q',
      scores: 'Indexer scores',
      pool: 'Candidate pool',
      topk: 'Top-K select',
      gather: 'main KV gather',
      logits: 'global·local logits',
      merge: 'Merged softmax',
      weighted: 'Weighted sum',
      wo: 'W_O',
      add1: 'Residual +',
      ln2: 'LayerNorm₂',
      router: 'MoE router',
      experts: 'Experts',
      add2: 'Residual +',
      lnf: 'LayerNorm_f',
      outLogits: 'Logits · Eᵀ',
      softmax: 'softmax',
      sample: 'Next token',
    },
    groups: {
      input: 'Input',
      embed: 'Embedding',
      query: 'Query · SWA',
      kv: 'global KV',
      select: 'Addressing',
      attention: 'Attention',
      moe: 'MoE',
      output: 'Output',
    },
    modeTitles: { swa: 'SWA only', full: 'Full', reuse: 'Reuse', reindex: 'Reindex' },
    modeNotes: {
      swa: 'No global KV; only the recent window is visible.',
      full: 'Builds main KV and indexer K, then selects Top-K.',
      reuse: 'Inherits both KV and Top-K; only the query and SWA are new.',
      reindex: 'Inherits KV; re-selects Top-K inside the candidate pool with its own indexer Q.',
    },
    encoder: 'Encoder (causal)',
    decoder: 'Decoder',
    ced: (fromLayer) => `projected from the L${fromLayer} output (CED)`,
    kvGroup: (index, layer, m) => `KV ${index} · built at L${layer} · compression ${m}`,
    reindexNote: 'Reindex layers only re-select Top-K inside the candidate pool',
    panels: {
      tokens: 'Tokens',
      embedding: 'Embedding',
      layerMap: 'Layer map (CSA2 modes)',
      block: (layer, mode) => `L${layer} · ${mode}`,
      query: 'Query and SWA K·V (recomputed in every layer)',
      kv: 'global KV entries',
      select: 'Indexer and Top-K (deciding what to read)',
      attention: 'Two branches → one softmax',
      moe: 'MoE feed-forward',
      output: 'Output',
      cache: 'KV cache',
    },
    encOnly: 'enc only',
    newToken: 'new',
    cache: 'cache',
    inherited: (layer) => `reused from L${layer}`,
    stateNote: {
      inherit: (from) => `This layer does not rebuild this value; it reuses the one from L${from}.`,
      none: 'This stage does not exist in this layer.',
    },
    window: 'window',
    partial: 'entry whose pair has not arrived (partial state)',
    topK: 'Top 10',
    probability: 'probability',
    focusTitle: (position, token) => `Focus: position ${position} "${token}"`,
    detail: {
      tokens:
        'Each character is looked up in the vocabulary to get an id. In prefill every prompt token passes the encoder layers, but only the tokens inside the last window pass the decoder layers.',
      embed: 'Row id of E plus row pos of P.',
      ln1: 'Each row is centered and scaled to unit variance, then multiplied by learned g and shifted by b.',
      q: 'Every layer builds its own Main Q from its hidden state. Each head gets a latent-sized query that is dotted directly with the shared latent c_j (no separate K).',
      swa: 'The SWA key and value of this token are written into the window. The window holds only the last w tokens, so its size does not depend on the context length.',
      compress:
        'Only Full layers build global KV. The encoder concatenates the hidden states of two adjacent tokens and merges them with W_c into one latent (compression 2); the decoder projects one latent per token from the encoder output (compression 1, CED). A token whose pair has not arrived uses a partial entry built from [h; 0].',
      fp4: 'The latent is stored as FP4 (E2M1) codes plus one E4M3 scale per channel group. It is dequantized right before attention, so only storage shrinks while the arithmetic stays high precision.',
      indexerK:
        'The stored entry is projected to a small dimension to make the indexer K, stored as MXFP4. It is used only to decide which entries to read.',
      indexerQ:
        'Only Full and Reindex layers build an Indexer Q and head weights w. Reuse layers never touch the indexer.',
      scores:
        'The focus token’s Indexer Q is dotted with each entry’s indexer K per head, passed through ReLU and weighted by w. Full layers score every visible entry; Reindex layers score only the candidate pool. The score’s job ends once the addresses are chosen.',
      pool: 'The first Full layer of the decoder keeps only the top blocks by block-max score as a candidate pool. Later Reindex layers score inside this pool only, so indexer cost no longer grows with the context.',
      topk: 'The k highest-scoring entries are selected. Unselected entries never get a logit, so their probability is exactly 0. Reuse layers take the selection of an earlier layer as is.',
      gather:
        'Only the selected entries’ main KV is fetched and dequantized. However long the context, one layer reads at most k entries.',
      logits:
        'The same q_t produces global logits (selected c_j) and local logits (k_s in the window). One c_j serves as both key and value.',
      merge:
        'The two logit lists are concatenated and passed through a single softmax, so a layer never attends to more than k + w positions.',
      weighted:
        'The global side uses c_j as its value and the SWA side its own v_s; the weighted sums form the head output and the heads are concatenated.',
      wo: 'The concatenated vector is multiplied by W_O to give the attention output.',
      add1: 'The attention output is added back onto x (residual connection).',
      ln2: 'Normalize again before the MoE block.',
      router:
        'The router gives every expert a sigmoid score; the top experts are chosen by score plus a load-balancing bias. The gate values are the scores normalized without the bias.',
      experts:
        'The shared expert always runs; only the chosen routed experts run and are mixed by their gates. The other experts are not computed.',
      add2: 'The MoE output is added on. This is the layer output and the next layer input.',
      lnf: 'The last position of the last layer output is normalized.',
      outLogits:
        'The embedding matrix E is used in reverse to score every vocabulary entry (tied weights).',
      softmax: 'Scores are divided by the temperature and turned into probabilities by softmax.',
      sample:
        'One token is drawn and appended. In the next step only the new token passes every layer; everything else is read from the caches (decode).',
    },
    cacheCard: {
      title: 'KV cache · current state',
      entries: (layer, m, n, bytes) =>
        `KV L${layer} (compression ${m}): ${n} entries × (${bytes} B)`,
      perToken: (bytes) => `global KV per token: ${bytes} B`,
      formula: (main, idx, terms, total) => `= (${main} + ${idx}) × (${terms}) = ${total} B/token`,
      real: `Real V4.1-Flash: (${REAL_MODEL.mainEntryBytes} + ${REAL_MODEL.indexerEntryBytes}) × (3 × ½ + 1) = ${REAL_MODEL.bytesPerToken} B/token`,
      readsTitle: 'Global-side entries read for the last token',
      reads: (indexer, main, swa) =>
        `indexer K scanned ${indexer} · main KV gathered ${main} · SWA read ${swa}`,
      realReads:
        'Real model, one decode step at 1M context: indexer K 174 MB, main KV 5.6 MB, SWA KV 3.0 MB (manual’s estimate)',
    },
    modelNote: (params, layers, vocab) =>
      `Model: character-level, ${layers} layers (encoder 5 + decoder 5), ${params.toLocaleString()} parameters, vocabulary ${vocab}, trained on this page’s text and the manual text by scripts/train_tiny_dsv41.py. Every value shown is a real computation.`,
    simplificationsTitle: 'How this differs from the real model (scaled down / simplified)',
    simplifications: [
      `Size: 10 layers (real 40 = encoder 20 + decoder 20), 4 heads × latent 16 (real 64 × 512), indexer 2 × 8 (real 128-d), Top-4 (real 512), window 8 (real 128), candidate pool 2 blocks × 4 (real 2,048 × 8), MoE shared 1 + routed 2/8 (real 1 + 6/384), FP4 scale group 8 channels (real 16). Full layers are L1·L3·L5 (real L2·L8·L14·L20), Reindex L7·L9 (real L24·L28·L32·L36), SWA-only L0 (real L0·L1).`,
      'Tokens are single characters and positions use learned absolute embeddings (the real model applies RoPE before quantization).',
      'The indexer score function (per-head ReLU weighted sum) and the SWA branch sharing the global query follow the manual’s stated assumptions.',
      'The MoE router’s sigmoid gate and selection bias (load balancing) are implementation details not covered by the manual.',
      'The prefill approximation in which the decoder processes only the last window of tokens follows the manual. DSpark drafting and persistent KV (SSD tier) are not simulated.',
      'FP4 storage was adapted with QAT in the last part of training. All real-model numbers above come from the manual.',
    ],
    sources:
      'Source: DeepseekV4.1_manual.docx in this repository (2026-09-21); the manual itself cites the DeepSeek-V4.1-Flash tech report (arXiv 2609.19969) and the Hugging Face model card.',
    empty: 'Type at least one character.',
    contextFull: (max) => `Context length ${max} reached.`,
  },
}
