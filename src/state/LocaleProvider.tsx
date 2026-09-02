import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { DEFAULT_LOCALE, getContent, isLocale } from '../content'
import type { Locale } from '../content/types'
import { LocaleContext } from './locale-context'

export const LOCALE_STORAGE_KEY = 'transformer-explain:locale'
/** 공유용 링크: ?lang=en 처럼 URL 로 언어를 지정할 수 있다. */
export const LOCALE_QUERY_PARAM = 'lang'

function readInitialLocale(): Locale {
  try {
    const fromUrl = new URLSearchParams(window.location.search).get(LOCALE_QUERY_PARAM)
    if (isLocale(fromUrl)) return fromUrl
  } catch {
    // URL 을 읽을 수 없으면 저장된 값으로 넘어간다.
  }
  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY)
    if (isLocale(stored)) return stored
  } catch {
    // 저장소를 쓸 수 없는 환경에서는 기본값을 쓴다.
  }
  return DEFAULT_LOCALE
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readInitialLocale)

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)
    try {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, next)
    } catch {
      // 저장에 실패해도 화면 전환은 계속 동작해야 한다.
    }
    try {
      // 현재 언어를 URL 에 반영해 두면 주소를 그대로 복사해 공유할 수 있다.
      const url = new URL(window.location.href)
      url.searchParams.set(LOCALE_QUERY_PARAM, next)
      window.history.replaceState(window.history.state, '', url)
    } catch {
      // URL 을 바꿀 수 없는 환경은 무시한다.
    }
  }, [])

  const content = useMemo(() => getContent(locale), [locale])

  useEffect(() => {
    document.documentElement.lang = locale
    document.title = content.ui.documentTitle
  }, [locale, content])

  const value = useMemo(() => ({ locale, setLocale, content }), [locale, setLocale, content])

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
}
