/** 설명 수준. 임원용은 비유와 핵심만, 엔지니어용은 수식과 구조까지 보여 준다. */
export type Audience = 'executive' | 'engineer'

/** 표시 언어. 본문과 UI 문자열이 모두 언어별로 준비되어 있어야 한다. */
export type Locale = 'ko' | 'en'

/** 섹션 안에 끼워 넣는 인터랙티브 데모의 종류. */
export type InteractiveKind =
  | 'tokenizer'
  | 'flow'
  | 'temperature'
  | 'positional'
  | 'attention'
  | 'heatmap'
  | 'multihead'
  | 'ffn'
  | 'stack'
  | 'training'
  | 'summary'
  | 'pipeline'

/** 섹션 식별자. URL 해시와 DOM id 로 쓰인다. 순서는 structure.ts 가 정한다. */
export type SectionId =
  | 'playground'
  | 'intro'
  | 'tokens'
  | 'position'
  | 'attention'
  | 'multihead'
  | 'ffn'
  | 'stack'
  | 'output'
  | 'training'
  | 'summary'

export interface KeyTerm {
  term: string
  meaning: string
}

/** 언어와 무관한 섹션 구조. */
export interface SectionMeta {
  id: SectionId
  /** 본문 뒤에 순서대로 끼워 넣는 인터랙티브 데모들 */
  interactives?: InteractiveKind[]
}

/** 언어별 섹션 본문. ko.ts 와 en.ts 가 SectionId 마다 하나씩 제공한다. */
export interface SectionText {
  /** 본문 제목 */
  title: string
  /** 목차에 쓰는 짧은 제목 */
  shortTitle: string
  /** 한 줄 요약. 모든 청중에게 보인다. */
  tagline: string
  /** 비유. 임원용 설명의 중심이다. */
  analogy: string
  /** 임원용 본문 문단들. 엔지니어용에서도 보인다. */
  executive: string[]
  /** 엔지니어용 추가 문단들. 엔지니어용에서만 보인다. */
  engineer: string[]
  /** 엔지니어용에서 보여 주는 핵심 용어 */
  keyTerms?: KeyTerm[]
  /** 검수 상태. draft 는 화면에 "초안" 표시가 붙는다. 언어별로 따로 검수한다. */
  status: 'draft' | 'ready'
}

/** 화면에 그리는 섹션 = 구조 + 번호 + 언어별 본문. */
export interface Section extends SectionMeta, SectionText {
  number: number
}

export interface Source {
  label: string
  detail: string
  url: string
}

/** 전체 흐름 다이어그램의 단계 하나. section 은 자세히 보기 링크 대상. */
export interface FlowStage {
  section: SectionId
  label: string
  detail: string
}

/** 온도 데모의 후보 단어와 (모델이 계산했다고 가정한) 점수. 점수 내림차순으로 둔다. */
export interface Candidate {
  token: string
  score: number
}

/** 본문 밖의 UI 문자열. */
export interface UiStrings {
  documentTitle: string
  brand: string
  skipToContent: string
  audienceLegend: string
  audience: Record<Audience, { label: string; hint: string }>
  localeLegend: string
  tocTitle: string
  hero: {
    kicker: string
    title: string
    lead: string
    /** "지금은 [임원용] 설명을 보고 있습니다..." 처럼 청중 이름 앞뒤로 나뉜다. */
    modeBefore: string
    modeAfter: Record<Audience, string>
  }
  analogyTitle: string
  engineerTitle: string
  draftBadge: string
  tokenizer: {
    heading: string
    description: string
    inputLabel: string
    listLabel: string
    count: (n: number) => string
    note: string
    defaultText: string
  }
  flow: {
    heading: string
    description: string
    prev: string
    next: string
    stepLabel: (index: number, total: number) => string
    goToSection: string
    stages: FlowStage[]
  }
  temperature: {
    heading: string
    description: string
    /** 빈칸이 있는 문맥 문장. 예: "나는 아침에 커피를 ___" */
    context: string
    sliderLabel: string
    valueLabel: (temperature: number) => string
    draw: string
    drawMany: string
    resultHeading: string
    tableCaption: string
    columns: { rank: string; token: string; probability: string }
    candidates: Candidate[]
    /** 온도 구간별 한 줄 설명 */
    regimes: { low: string; neutral: string; high: string }
    note: string
  }
  positional: {
    heading: string
    description: string
    rowLabel: (position: number) => string
    columnLabel: (dimension: number) => string
    selectedHeading: (position: number) => string
    legendLabel: string
    tableCaption: string
    note: string
  }
  attention: {
    heading: string
    description: string
    sentence: string
    /** 예시 토큰 4개. 문장에 나오는 순서대로 둔다 (인과 마스크가 이 순서를 쓴다). */
    tokens: string[]
    /** 설명용 Q, K, V 벡터 (토큰 수 × 차원). 실제 모델이 학습한 값이 아니다. */
    q: number[][]
    k: number[][]
    v: number[][]
    /** V 의 두 성분에 붙인 설명용 이름 */
    valueDims: [string, string]
    queryLegend: string
    causalLabel: string
    weightsHeading: (token: string) => string
    masked: string
    maskedNote: (token: string) => string
    outputHeading: (token: string) => string
    detailsSummary: string
    steps: { vectors: string; scores: string; scaled: string; softmax: string; output: string }
    columns: {
      token: string
      q: string
      k: string
      v: string
      score: string
      scaled: string
      weight: string
    }
    note: string
  }
  ffn: {
    heading: string
    description: string
    tokens: string[]
    /** 각 토큰의 입력 벡터 (어텐션을 거친 뒤라고 가정) */
    inputs: number[][]
    w1: number[][]
    b1: number[]
    w2: number[][]
    b2: number[]
    selectLegend: string
    sharedLabel: string
    stageLabels: { input: string; preActivation: string; hidden: string; output: string }
    zeroNote: string
    note: string
  }
  stack: {
    heading: string
    description: string
    sliderLabel: string
    layerLabel: (count: number) => string
    layersCaption: (count: number) => string
    layerTitle: (index: number) => string
    attention: string
    ffn: string
    norm: string
    residual: string
    inputLabel: string
    outputLabel: string
    note: string
  }
  training: {
    heading: string
    description: string
    context: string
    answerLabel: string
    answer: string
    candidates: Candidate[]
    stepOnce: string
    stepMany: string
    reset: string
    stepsLabel: string
    lossLabel: string
    answerProbabilityLabel: string
    columns: { token: string; probability: string }
    correctMark: string
    tableCaption: string
    note: string
  }
  summary: {
    cardTitle: string
    steps: string[]
    takeawaysTitle: string
    takeaways: { lead: string; text: string }[]
    nextStepsTitle: string
    nextSteps: { label: string; detail: string; url: string }[]
  }
  heatmap: {
    heading: string
    description: string
    emptyTitle: string
    emptyBody: string
    errorsTitle: string
    exampleLegend: string
    layerLabel: string
    headLabel: string
    layerOption: (index: number) => string
    headOption: (index: number) => string
    tableCaption: (layer: number, head: number) => string
    cellTitle: (from: string, to: string, weight: string) => string
    strongest: (from: string, to: string, weight: string) => string
    legendLabel: string
    modelNote: (model: string, generatedAt: string) => string
  }
  multihead: {
    heading: string
    description: string
    thumbsLabel: string
  }
  pipeline: {
    heading: string
    description: string
    inputLabel: string
    charCount: (count: number, max: number) => string
    emptyInput: string
    stageTokens: string
    stageEmbeddings: string
    stageAttention: string
    stageOutput: string
    tokensLegend: string
    unknownNote: string
    embeddingNote: (dModel: number) => string
    vectorHeading: (token: string) => string
    layerLabel: string
    headLabel: string
    layerOption: (index: number) => string
    headOption: (index: number) => string
    attentionCaption: (layer: number, head: number) => string
    cellTitle: (from: string, to: string, weight: string) => string
    strongest: (from: string, to: string, weight: string) => string
    temperatureLabel: string
    probabilityCaption: string
    columns: { token: string; probability: string }
    appendOne: string
    appendMany: string
    reset: string
    modelNote: (paramCount: number, layers: number, heads: number, corpusChars: number) => string
    defaultText: string
  }
  /** 접힌 본문 문단을 펼치는 버튼 문구 */
  readMore: string
  sourcesTitle: string
}

export interface LocaleContent {
  ui: UiStrings
  sections: Section[]
  sources: Source[]
}
