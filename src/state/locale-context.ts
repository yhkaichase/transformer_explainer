import { createContext, useContext } from 'react'
import type { Locale, LocaleContent } from '../content/types'

export interface LocaleState {
  locale: Locale
  setLocale: (locale: Locale) => void
  /** 현재 언어의 본문과 UI 문자열 */
  content: LocaleContent
}

export const LocaleContext = createContext<LocaleState | null>(null)

export function useLocale(): LocaleState {
  const state = useContext(LocaleContext)
  if (!state) {
    throw new Error('useLocale 는 LocaleProvider 안에서만 사용할 수 있습니다.')
  }
  return state
}
