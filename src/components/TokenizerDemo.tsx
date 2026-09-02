import { useId, useState } from 'react'
import { tokenize } from '../lib/tokenize'

const DEFAULT_TEXT = '은행에 가서 돈을 찾았다.'

export function TokenizerDemo() {
  const [text, setText] = useState(DEFAULT_TEXT)
  const inputId = useId()
  const tokens = tokenize(text)

  return (
    <div className="demo" data-testid="tokenizer-demo">
      <div className="demo-header">
        <h3>직접 해보기: 문장을 토큰으로 나누기</h3>
        <p>문장을 바꿔 보세요. 조각(토큰)이 어떻게 나뉘는지 바로 보입니다.</p>
      </div>
      <label htmlFor={inputId} className="visually-hidden">
        토큰으로 나눌 문장
      </label>
      <textarea
        id={inputId}
        className="demo-input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={2}
        spellCheck={false}
      />
      <p className="demo-stat" aria-live="polite">
        토큰 {tokens.length}개
      </p>
      <ol className="token-list" aria-label="토큰 목록">
        {tokens.map((token) => (
          <li key={`${token.index}-${token.text}`} className={`token token-${token.kind}`}>
            <span className="token-index">{token.index}</span>
            <span className="token-text">{token.text}</span>
          </li>
        ))}
      </ol>
      <p className="demo-note">
        이 데모는 이해를 돕기 위해 공백과 문장부호 기준으로만 나눕니다. 실제 모델은 서브워드 단위로
        더 잘게 나누므로 조각 수와 경계가 다를 수 있습니다.
      </p>
    </div>
  )
}
