/**
 * 설명용 토크나이저.
 *
 * 공백을 버리고, 글자 묶음·숫자 묶음·문장부호 하나를 각각 토큰으로 만든다.
 * 실제 언어 모델은 서브워드(subword) 토크나이저를 사용하므로 조각 수와 경계가 다르다.
 * 이 구현은 "문장이 조각으로 나뉜다"는 개념을 보여 주기 위한 것이다.
 */

export type TokenKind = 'word' | 'number' | 'punct'

export interface Token {
  /** 토큰 문자열 */
  text: string
  kind: TokenKind
  /** 문장 안에서의 순서 (0부터) */
  index: number
}

// \p{L}: 모든 문자(한글, 라틴 등), \p{M}: 결합 기호, \p{N}: 숫자
const TOKEN_PATTERN = /\p{L}[\p{L}\p{M}]*|\p{N}+|[^\s\p{L}\p{M}\p{N}]/gu

function classify(text: string): TokenKind {
  if (/^\p{N}+$/u.test(text)) return 'number'
  if (/^\p{L}/u.test(text)) return 'word'
  return 'punct'
}

export function tokenize(text: string): Token[] {
  const tokens: Token[] = []
  for (const match of text.matchAll(TOKEN_PATTERN)) {
    const value = match[0]
    tokens.push({ text: value, kind: classify(value), index: tokens.length })
  }
  return tokens
}
