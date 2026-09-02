import { describe, expect, it } from 'vitest'
import { tokenize } from './tokenize'

const texts = (input: string) => tokenize(input).map((t) => t.text)

describe('tokenize', () => {
  it('한국어 문장을 공백과 문장부호 기준으로 나눈다', () => {
    expect(texts('은행에 가서 돈을 찾았다.')).toEqual(['은행에', '가서', '돈을', '찾았다', '.'])
  })

  it('영어 문장을 나눈다', () => {
    expect(texts('The cat sat on the mat.')).toEqual(['The', 'cat', 'sat', 'on', 'the', 'mat', '.'])
  })

  it('문장부호는 각각 별도 토큰이 된다', () => {
    expect(texts('안녕하세요, 세계!')).toEqual(['안녕하세요', ',', '세계', '!'])
  })

  it('숫자와 글자를 구분한다', () => {
    const tokens = tokenize('2017년 논문')
    expect(tokens.map((t) => t.text)).toEqual(['2017', '년', '논문'])
    expect(tokens.map((t) => t.kind)).toEqual(['number', 'word', 'word'])
  })

  it('순서(index)를 0부터 매긴다', () => {
    expect(tokenize('a b c').map((t) => t.index)).toEqual([0, 1, 2])
  })

  it('빈 문자열과 공백만 있는 문자열은 빈 배열', () => {
    expect(tokenize('')).toEqual([])
    expect(tokenize('   \n\t ')).toEqual([])
  })

  it('여러 줄과 연속 공백을 처리한다', () => {
    expect(texts('첫 줄\n\n둘째   줄')).toEqual(['첫', '줄', '둘째', '줄'])
  })
})
