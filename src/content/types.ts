/** 설명 수준. 임원용은 비유와 핵심만, 엔지니어용은 수식과 구조까지 보여 준다. */
export type Audience = 'executive' | 'engineer'

/** 표시 언어. 본문과 UI 문자열이 모두 언어별로 준비되어 있어야 한다. */
export type Locale = 'ko' | 'en'

/** 섹션 안에 끼워 넣는 인터랙티브 데모의 종류. */
export type InteractiveKind = 'tokenizer'

/** 섹션 식별자. URL 해시와 DOM id 로 쓰인다. 순서는 structure.ts 가 정한다. */
export type SectionId =
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
  interactive?: InteractiveKind
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
  sourcesTitle: string
}

export interface LocaleContent {
  ui: UiStrings
  sections: Section[]
  sources: Source[]
}
