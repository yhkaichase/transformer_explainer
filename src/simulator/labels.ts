import type { StageId } from './stages'

export type SimLang = 'ko' | 'en'

export interface SimLabels {
  title: string
  subtitle: string
  prompt: string
  promptHint: (max: number) => string
  prefill: string
  decode: (step: number, cached: number) => string
  decodeOne: string
  decodeTen: string
  reset: string
  temperature: string
  layer: string
  head: string
  layerOption: (i: number) => string
  headOption: (i: number) => string
  focus: string
  autoplay: string
  autoplayStop: string
  speed: string
  loopDecode: string
  prev: string
  next: string
  stageNames: Record<StageId, string>
  groups: { input: string; embed: string; attention: string; ffn: string; output: string }
  panels: {
    tokens: string
    embedding: string
    block: (layer: number) => string
    attentionRow: string
    ffnRow: string
    output: string
  }
  cache: string
  newToken: string
  topK: string
  logit: string
  probability: string
  focusTitle: (position: number, token: string) => string
  detail: Record<StageId, string>
  modelNote: (
    params: number,
    layers: number,
    heads: number,
    dModel: number,
    vocab: number,
  ) => string
  fullPage: string
  empty: string
  contextFull: (max: number) => string
}

export const LABELS: Record<SimLang, SimLabels> = {
  ko: {
    title: 'Transformer Simulator',
    subtitle:
      '페이지에 내장된 작은 트랜스포머가 실제로 계산합니다. 글을 바꾸고, 단계를 넘기며 가중치가 어떻게 적용되는지 보세요.',
    prompt: '프롬프트',
    promptHint: (max) => `최대 ${max}글자. 글자 하나가 토큰 하나입니다.`,
    prefill: 'Prefill · 프롬프트 토큰 전체를 한 번에 계산',
    decode: (step, cached) => `Decode ${step} · 새 토큰 1개만 계산, K·V 캐시 ${cached}개 재사용`,
    decodeOne: 'Decode: 다음 토큰',
    decodeTen: 'Decode ×10',
    reset: '처음으로',
    temperature: '온도',
    layer: '층',
    head: '헤드',
    layerOption: (i) => `${i + 1}층`,
    headOption: (i) => `헤드 ${i + 1}`,
    focus: '초점 토큰',
    autoplay: '자동 재생',
    autoplayStop: '정지',
    speed: '배속',
    loopDecode: '마지막 단계 뒤 다음 토큰 이어 쓰기',
    prev: '이전 단계',
    next: '다음 단계',
    stageNames: {
      tokens: '토큰화',
      embed: '임베딩 + 위치',
      ln1: 'LayerNorm₁',
      qkv: 'W_Q · W_K · W_V',
      heads: '헤드 분할',
      scores: '점수 · softmax',
      weighted: '가중합 A·V',
      wo: 'W_O',
      add1: '잔차 +',
      ln2: 'LayerNorm₂',
      ffn1: 'W₁',
      relu: 'ReLU',
      ffn2: 'W₂',
      add2: '잔차 +',
      lnf: 'LayerNorm_f',
      logits: '로짓 · Eᵀ',
      softmax: 'softmax',
      sample: '다음 토큰',
    },
    groups: {
      input: '입력',
      embed: '임베딩',
      attention: '어텐션',
      ffn: '피드포워드',
      output: '출력',
    },
    panels: {
      tokens: '토큰',
      embedding: '임베딩',
      block: (layer) => `트랜스포머 블록 · ${layer + 1}층`,
      attentionRow: '멀티헤드 셀프 어텐션',
      ffnRow: '피드포워드',
      output: '출력',
    },
    cache: '캐시',
    newToken: '새 토큰',
    topK: '상위 10개',
    logit: '로짓',
    probability: '확률',
    focusTitle: (position, token) => `초점: 위치 ${position} "${token}"`,
    detail: {
      tokens:
        '글자를 어휘 사전에서 찾아 번호(id)로 바꿉니다. 이 번호가 임베딩 행렬 E 의 행을 고릅니다.',
      embed: 'E 의 id 번째 행과 P 의 pos 번째 행을 더합니다.',
      ln1: '행마다 평균을 빼고 표준편차로 나눈 뒤, 학습된 g 와 b 를 곱하고 더합니다.',
      qkv: '같은 h 에 세 개의 행렬을 곱해 질의, 키, 값을 만듭니다. 히트맵이 가중치이고 아래 줄이 결과입니다.',
      heads: '48차원 벡터를 16차원씩 3개 헤드로 나눕니다. 색 구간이 헤드입니다.',
      scores:
        '초점 토큰의 q 와 앞선 토큰들의 k 를 내적하고 √16 으로 나눈 뒤 softmax 합니다. 뒤 토큰은 마스크됩니다.',
      weighted: '가중치로 v 를 섞어 헤드 출력을 만들고, 3개 헤드를 이어 붙입니다.',
      wo: '이어 붙인 벡터에 W_O 를 곱해 어텐션 출력을 만듭니다.',
      add1: '어텐션 출력을 원래 x 에 더합니다 (잔차 연결).',
      ln2: '피드포워드 전에 다시 정규화합니다.',
      ffn1: 'W₁ 로 48차원을 192차원으로 넓힙니다.',
      relu: '음수를 0 으로 자릅니다. 회색 칸이 0 이 된 자리입니다.',
      ffn2: 'W₂ 로 다시 48차원으로 줄입니다.',
      add2: '피드포워드 출력을 더합니다. 이것이 이 층의 출력이고 다음 층의 입력입니다.',
      lnf: '마지막 층 출력의 마지막 위치를 정규화합니다.',
      logits: '임베딩 행렬 E 를 거꾸로 써서 어휘마다 점수를 냅니다 (가중치 공유).',
      softmax: '점수를 온도로 나눈 뒤 softmax 로 확률을 만듭니다.',
      sample:
        '확률에 따라 토큰 하나를 뽑아 뒤에 붙입니다. 다음 계산은 새 토큰 1개만 하면 됩니다 (decode).',
    },
    modelNote: (params, layers, heads, dModel, vocab) =>
      `모델: 문자 단위 디코더 전용 트랜스포머, ${layers}층 × ${heads}헤드, d_model ${dModel}, 어휘 ${vocab}, 파라미터 ${params.toLocaleString()}개. 이 저장소의 scripts/train_tiny_model.py 가 페이지 본문으로 학습. 값은 모두 실제 계산 결과입니다.`,
    fullPage: '설명 페이지로',
    empty: '글자를 한 개 이상 입력하세요.',
    contextFull: (max) => `문맥 길이 ${max} 에 도달했습니다.`,
  },
  en: {
    title: 'Transformer Simulator',
    subtitle:
      'A small transformer embedded in this page computes for real. Change the text and step through the stages to see how the weights are applied.',
    prompt: 'Prompt',
    promptHint: (max) => `Up to ${max} characters. One character is one token.`,
    prefill: 'Prefill · all prompt tokens computed at once',
    decode: (step, cached) =>
      `Decode ${step} · only the new token is computed, ${cached} cached K·V reused`,
    decodeOne: 'Decode: next token',
    decodeTen: 'Decode ×10',
    reset: 'Reset',
    temperature: 'Temperature',
    layer: 'Layer',
    head: 'Head',
    layerOption: (i) => `Layer ${i + 1}`,
    headOption: (i) => `Head ${i + 1}`,
    focus: 'Focus token',
    autoplay: 'Autoplay',
    autoplayStop: 'Stop',
    speed: 'Speed',
    loopDecode: 'Decode a token after the last stage',
    prev: 'Previous stage',
    next: 'Next stage',
    stageNames: {
      tokens: 'Tokenize',
      embed: 'Embedding + position',
      ln1: 'LayerNorm₁',
      qkv: 'W_Q · W_K · W_V',
      heads: 'Split heads',
      scores: 'Scores · softmax',
      weighted: 'Weighted sum A·V',
      wo: 'W_O',
      add1: 'Residual +',
      ln2: 'LayerNorm₂',
      ffn1: 'W₁',
      relu: 'ReLU',
      ffn2: 'W₂',
      add2: 'Residual +',
      lnf: 'LayerNorm_f',
      logits: 'Logits · Eᵀ',
      softmax: 'softmax',
      sample: 'Next token',
    },
    groups: {
      input: 'Input',
      embed: 'Embedding',
      attention: 'Attention',
      ffn: 'Feed-forward',
      output: 'Output',
    },
    panels: {
      tokens: 'Tokens',
      embedding: 'Embedding',
      block: (layer) => `Transformer block · layer ${layer + 1}`,
      attentionRow: 'Multi-head self-attention',
      ffnRow: 'Feed-forward',
      output: 'Output',
    },
    cache: 'cache',
    newToken: 'new',
    topK: 'Top 10',
    logit: 'logit',
    probability: 'probability',
    focusTitle: (position, token) => `Focus: position ${position} "${token}"`,
    detail: {
      tokens:
        'Each character is looked up in the vocabulary to get an id. The id selects a row of the embedding matrix E.',
      embed: 'Row id of E plus row pos of P.',
      ln1: 'Each row is centered and scaled to unit variance, then multiplied by learned g and shifted by b.',
      qkv: 'The same h is multiplied by three matrices to make queries, keys, and values. Heatmaps are the weights, strips are the results.',
      heads: 'The 48-dimensional vectors are split into 3 heads of 16. Color bands mark the heads.',
      scores:
        'The q of the focus token is dotted with the k of every earlier token, divided by √16, and passed through softmax. Later tokens are masked.',
      weighted: 'The weights mix the v vectors into a head output; the 3 heads are concatenated.',
      wo: 'The concatenated vector is multiplied by W_O to give the attention output.',
      add1: 'The attention output is added back onto x (residual connection).',
      ln2: 'Normalize again before the feed-forward layer.',
      ffn1: 'W₁ widens 48 dimensions to 192.',
      relu: 'Negative values are cut to 0. Gray cells were zeroed.',
      ffn2: 'W₂ narrows back to 48 dimensions.',
      add2: 'The feed-forward output is added on. This is the layer output and the next layer input.',
      lnf: 'The last position of the last layer output is normalized.',
      logits:
        'The embedding matrix E is used in reverse to score every vocabulary entry (tied weights).',
      softmax: 'Scores are divided by the temperature and turned into probabilities by softmax.',
      sample:
        'One token is drawn from the probabilities and appended. The next computation only needs the new token (decode).',
    },
    modelNote: (params, layers, heads, dModel, vocab) =>
      `Model: character-level decoder-only transformer, ${layers} layers × ${heads} heads, d_model ${dModel}, vocabulary ${vocab}, ${params.toLocaleString()} parameters, trained on this page's text by scripts/train_tiny_model.py. Every value shown is a real computation.`,
    fullPage: 'Explanation page',
    empty: 'Type at least one character.',
    contextFull: (max) => `Context length ${max} reached.`,
  },
}
