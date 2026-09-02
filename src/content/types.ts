/** 설명 수준. 임원용은 비유와 핵심만, 엔지니어용은 수식과 구조까지 보여 준다. */
export type Audience = 'executive' | 'engineer'

/** 섹션 안에 끼워 넣는 인터랙티브 데모의 종류. */
export type InteractiveKind = 'tokenizer'

export interface KeyTerm {
  term: string
  meaning: string
}

export interface Section {
  /** URL 해시와 DOM id 로 쓰인다 (영문 소문자). */
  id: string
  number: number
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
  interactive?: InteractiveKind
  /** draft 는 화면에 "초안" 표시가 붙는다. */
  status: 'draft' | 'ready'
}

export interface Source {
  label: string
  detail: string
  url: string
}
