import { en } from './en'
import { ko } from './ko'
import { SECTION_META } from './structure'
import type { Locale, LocaleContent, SectionId, SectionText, Source, UiStrings } from './types'

export interface LocalePack {
  ui: UiStrings
  sections: Record<SectionId, SectionText>
  sources: Source[]
}

export const LOCALES: { value: Locale; label: string }[] = [
  { value: 'ko', label: '한국어' },
  { value: 'en', label: 'English' },
]

export const DEFAULT_LOCALE: Locale = 'ko'

const PACKS: Record<Locale, LocalePack> = { ko, en }

export function isLocale(value: unknown): value is Locale {
  return value === 'ko' || value === 'en'
}

/** 언어별 본문을 구조(structure.ts)와 합쳐 화면에 그릴 순서대로 돌려준다. */
export function getContent(locale: Locale): LocaleContent {
  const pack = PACKS[locale]
  return {
    ui: pack.ui,
    sources: pack.sources,
    sections: SECTION_META.map((meta, index) => ({
      ...meta,
      number: index + 1,
      ...pack.sections[meta.id],
    })),
  }
}
