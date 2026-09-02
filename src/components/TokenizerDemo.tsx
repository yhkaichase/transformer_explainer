import { useId, useState } from 'react'
import { tokenize } from '../lib/tokenize'
import { useLocale } from '../state/locale-context'

export function TokenizerDemo() {
  const { content } = useLocale()
  const strings = content.ui.tokenizer
  // null 이면 아직 사용자가 고치지 않은 것이므로 현재 언어의 기본 문장을 보여 준다.
  const [edited, setEdited] = useState<string | null>(null)
  const inputId = useId()
  const text = edited ?? strings.defaultText
  const tokens = tokenize(text)

  return (
    <div className="demo" data-testid="tokenizer-demo">
      <div className="demo-header">
        <h3>{strings.heading}</h3>
        <p>{strings.description}</p>
      </div>
      <label htmlFor={inputId} className="visually-hidden">
        {strings.inputLabel}
      </label>
      <textarea
        id={inputId}
        className="demo-input"
        value={text}
        onChange={(event) => setEdited(event.target.value)}
        rows={2}
        spellCheck={false}
      />
      <p className="demo-stat" aria-live="polite">
        {strings.count(tokens.length)}
      </p>
      <ol className="token-list" aria-label={strings.listLabel}>
        {tokens.map((token) => (
          <li key={`${token.index}-${token.text}`} className={`token token-${token.kind}`}>
            <span className="token-index">{token.index}</span>
            <span className="token-text">{token.text}</span>
          </li>
        ))}
      </ol>
      <p className="demo-note">{strings.note}</p>
    </div>
  )
}
