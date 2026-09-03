import type { SectionMeta } from './types'

/**
 * 섹션의 순서와 인터랙티브 데모 지정. 언어와 무관하다.
 * 새 섹션을 추가하면 types.ts 의 SectionId 와 ko.ts / en.ts 의 본문도 함께 추가해야 컴파일된다.
 */
export const SECTION_META: SectionMeta[] = [
  { id: 'intro', interactives: ['flow'] },
  { id: 'tokens', interactives: ['tokenizer'] },
  { id: 'position', interactives: ['positional'] },
  { id: 'attention', interactives: ['attention', 'heatmap'] },
  { id: 'multihead', interactives: ['multihead'] },
  { id: 'ffn', interactives: ['ffn'] },
  { id: 'stack', interactives: ['stack'] },
  { id: 'output', interactives: ['temperature'] },
  { id: 'training', interactives: ['training'] },
  { id: 'summary', interactives: ['summary'] },
]
