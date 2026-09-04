import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { sampleIndex, softmax } from '../lib/math'
import { loadTinyModel, type TraceResult } from '../lib/tinyTransformer'
import { LABELS, type SimLang } from './labels'
import { MatrixCanvas } from './MatrixCanvas'
import { formatFormula, STAGE_GROUP, STAGES, type StageId } from './stages'

const DEFAULT_PROMPT: Record<SimLang, string> = {
  ko: '은행에 가서 돈을',
  en: 'I went to the bank to',
}
const PROMPT_MAX = 40
const TOP_K = 10
const ROW = 14
const CELL = 3
const WIDE_CELL = 1.5
const HEAT_CELL = 2
const AUTOPLAY_MS = 1600
const SPEEDS = [0.5, 1, 2, 4]

function maxAbs(values: ArrayLike<number>): number {
  let max = 0
  for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i]))
  return max || 1
}

function displayChar(char: string): string {
  if (char === ' ') return '␣'
  if (char === '\n') return '⏎'
  return char
}

const fmt = (value: number) => value.toFixed(2)

interface StripProps {
  title: string
  data: Float32Array
  rows: number
  cols: number
  cell?: number
  highlightRow?: number | null
  dimRowsBefore?: number | null
  separatorEvery?: number
  markZeros?: boolean
  stages: StageId[]
  active: StageId
  onSelectRow?: (row: number) => void
  scale?: number
}

/** 활성값 행렬(T × 열)을 띠 모양 히트맵으로 그린다. */
function Strip({
  title,
  data,
  rows,
  cols,
  cell = CELL,
  highlightRow = null,
  dimRowsBefore = null,
  separatorEvery,
  markZeros,
  stages,
  active,
  onSelectRow,
  scale,
}: StripProps) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-strip${isActive ? ' is-active' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>{title}</figcaption>
      <MatrixCanvas
        rows={rows}
        cols={cols}
        value={(r, c) => data[r * cols + c]}
        scale={scale ?? maxAbs(data)}
        mode="diverging"
        cellWidth={cell}
        cellHeight={ROW}
        highlightRow={highlightRow}
        dimRowsBefore={dimRowsBefore}
        separatorEvery={separatorEvery}
        markZeros={markZeros}
        label={`${title} ${rows}×${cols}`}
        tooltip={(r, c, v) => `${title}[${r}, ${c}] = ${v.toFixed(3)}`}
        onSelectRow={onSelectRow}
      />
    </figure>
  )
}

interface HeatProps {
  title: string
  data: Float32Array
  rows: number
  cols: number
  cell?: number
  stages: StageId[]
  active: StageId
  highlightRow?: number | null
}

/** 가중치 행렬 히트맵. */
function Heat({
  title,
  data,
  rows,
  cols,
  cell = HEAT_CELL,
  stages,
  active,
  highlightRow = null,
}: HeatProps) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-heat${isActive ? ' is-active' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>
        {title}{' '}
        <span className="sim-shape">
          {rows}×{cols}
        </span>
      </figcaption>
      <MatrixCanvas
        rows={rows}
        cols={cols}
        value={(r, c) => data[r * cols + c]}
        scale={maxAbs(data)}
        mode="diverging"
        cellWidth={cell}
        cellHeight={cell}
        highlightRow={highlightRow}
        label={`${title} ${rows}×${cols}`}
        tooltip={(r, c, v) => `${title}[${r}, ${c}] = ${v.toFixed(4)}`}
      />
    </figure>
  )
}

function Op({ children }: { children: ReactNode }) {
  return (
    <span className="sim-op" aria-hidden="true">
      {children}
    </span>
  )
}

interface BarItem {
  key: string
  label: string
  value: number
  display: string
}

function Bars({
  title,
  items,
  max,
  stages,
  active,
  highlightKey,
}: {
  title: string
  items: BarItem[]
  max: number
  stages: StageId[]
  active: StageId
  highlightKey?: string
}) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-bars${isActive ? ' is-active' : ''}`}
      data-stage={stages.join(' ')}
    >
      <figcaption>{title}</figcaption>
      <ol>
        {items.map((item) => (
          <li key={item.key} className={item.key === highlightKey ? 'is-picked' : undefined}>
            <span className="sim-bar-label">{item.label}</span>
            <span className="sim-bar-track" aria-hidden="true">
              <span
                className="sim-bar-fill"
                style={{ width: `${(Math.max(0, item.value) / max) * 100}%` }}
              />
            </span>
            <span className="sim-bar-value">{item.display}</span>
          </li>
        ))}
      </ol>
    </figure>
  )
}

interface SimulatorProps {
  /** 뽑기에 쓰는 난수. 테스트에서 고정한다. */
  random?: () => number
  initialLang?: SimLang
}

export function Simulator({ random = Math.random, initialLang }: SimulatorProps) {
  const model = useMemo(() => loadTinyModel(), [])
  const [lang, setLang] = useState<SimLang>(
    () =>
      initialLang ??
      (new URLSearchParams(window.location.search).get('lang') === 'en' ? 'en' : 'ko'),
  )
  const L = LABELS[lang]
  const [prompt, setPrompt] = useState<string | null>(null)
  const [generated, setGenerated] = useState('')
  const [temperature, setTemperature] = useState(1)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [focus, setFocus] = useState<number | null>(null)
  const [stageIndex, setStageIndex] = useState(0)
  const [autoplay, setAutoplay] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loopDecode, setLoopDecode] = useState(false)

  const promptText = prompt ?? DEFAULT_PROMPT[lang]
  const text = promptText + generated
  const ids = useMemo(() => model.encode(text), [model, text])
  const trace: TraceResult | null = useMemo(
    () => (ids.length > 0 ? model.forwardTrace(ids) : null),
    [model, ids],
  )
  const { dModel: D, dFf: F, nHeads: H, nLayers, context } = model.config
  const dh = D / H
  const T = ids.length
  const stage = STAGES[stageIndex]
  const mode: 'prefill' | 'decode' = generated.length === 0 ? 'prefill' : 'decode'
  const decodeStep = Array.from(generated).length
  const focusIndex = Math.max(0, Math.min(focus ?? T - 1, T - 1))
  const distribution = useMemo(
    () => (trace ? model.nextTokenDistribution(trace.logits, temperature) : []),
    [model, trace, temperature],
  )
  const canDecode = trace !== null && T < context
  const cacheRows = mode === 'decode' ? T - 1 : null

  // 자동 재생: 배속에 따라 단계를 넘기고, 옵션이 켜져 있으면 마지막 단계 뒤에 토큰을 하나 이어 쓴다.
  const tickRef = useRef<() => void>(() => {})
  useEffect(() => {
    if (!autoplay) return
    const id = window.setInterval(() => tickRef.current(), AUTOPLAY_MS / speed)
    return () => window.clearInterval(id)
  }, [autoplay, speed])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (event.key === 'ArrowRight') setStageIndex((i) => Math.min(STAGES.length - 1, i + 1))
      if (event.key === 'ArrowLeft') setStageIndex((i) => Math.max(0, i - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const decode = (steps: number, keepStage = false) => {
    let current = text
    for (let step = 0; step < steps; step++) {
      const currentIds = model.encode(current)
      if (currentIds.length >= context) break
      const candidates = model.nextTokenDistribution(
        model.forwardTrace(currentIds).logits,
        temperature,
      )
      current +=
        candidates[
          sampleIndex(
            candidates.map((c) => c.probability),
            random,
          )
        ].token
    }
    setGenerated(current.slice(promptText.length))
    setFocus(null)
    if (!keepStage) setStageIndex(STAGES.indexOf('sample'))
  }

  // 렌더 중에 ref 를 쓰지 않도록 effect 안에서 최신 tick 을 등록한다.
  useEffect(() => {
    tickRef.current = () => {
      if (stageIndex < STAGES.length - 1) {
        setStageIndex(stageIndex + 1)
        return
      }
      if (loopDecode && canDecode) decode(1, true)
      setStageIndex(0)
    }
  })

  const reset = () => {
    setGenerated('')
    setFocus(null)
    setStageIndex(0)
  }

  const shapeValues = { T, D, F, H, dh, V: model.vocab.length }
  const weights = model.layerWeights(layer)
  const layerTrace = trace?.layers[layer] ?? null
  const attention = layerTrace ? layerTrace.attentions[head] : null
  const row = (data: Float32Array, cols: number, r: number) => data.slice(r * cols, (r + 1) * cols)
  const topIds = distribution.slice(0, TOP_K)
  const logitMax = trace ? maxAbs(trace.logits) : 1
  const topEmbeddingRows = new Float32Array(TOP_K * D)
  topIds.forEach((candidate, i) =>
    topEmbeddingRows.set(
      model.tokenEmbedding.slice(candidate.id * D, (candidate.id + 1) * D),
      i * D,
    ),
  )
  const pickedKey = mode === 'decode' && trace ? String(trace.ids[T - 1]) : undefined

  const renderFocus = () => {
    if (!trace || !layerTrace || !attention) return null
    const i = focusIndex
    const token = displayChar(trace.tokens[i])
    const one = (
      title: string,
      data: Float32Array,
      cols: number,
      extra: Partial<StripProps> = {},
    ) => (
      <Strip
        title={title}
        data={data}
        rows={1}
        cols={cols}
        stages={[stage]}
        active={stage}
        {...extra}
      />
    )
    const x = trace.embeddings[i]
    switch (stage) {
      case 'tokens':
        return (
          <p className="sim-eq-text">
            "{token}" → id {trace.ids[i]} → E[{trace.ids[i]}]
          </p>
        )
      case 'embed':
        return (
          <div className="sim-eq">
            {one(`E[${trace.ids[i]}]`, trace.tokenEmbeddings[i], D)}
            <Op>+</Op>
            {one(`P[${i}]`, trace.positionEmbeddings[i], D)}
            <Op>=</Op>
            {one('x', x, D)}
          </div>
        )
      case 'ln1':
        return (
          <div className="sim-eq">
            {one('x', x, D)}
            <Op>→ LN₁ →</Op>
            {one('h', row(layerTrace.ln1, D, i), D)}
            {one('g', weights.ln1G, D)}
            {one('b', weights.ln1B, D)}
          </div>
        )
      case 'qkv':
        return (
          <div className="sim-eq-stack">
            {(
              [
                ['q', weights.wq, weights.bq, layerTrace.q, 'W_Q', 'b_Q'],
                ['k', weights.wk, weights.bk, layerTrace.k, 'W_K', 'b_K'],
                ['v', weights.wv, weights.bv, layerTrace.v, 'W_V', 'b_V'],
              ] as const
            ).map(([name, w, b, out, wName, bName]) => (
              <div className="sim-eq" key={name}>
                {one('h', row(layerTrace.ln1, D, i), D)}
                <Op>·</Op>
                <Heat title={wName} data={w} rows={D} cols={D} stages={[stage]} active={stage} />
                <Op>+</Op>
                {one(bName, b, D)}
                <Op>=</Op>
                {one(name, row(out, D, i), D, { separatorEvery: dh })}
              </div>
            ))}
          </div>
        )
      case 'heads':
        return (
          <div className="sim-eq">
            {one('q', row(layerTrace.q, D, i), D, { separatorEvery: dh })}
            {one('k', row(layerTrace.k, D, i), D, { separatorEvery: dh })}
            {one('v', row(layerTrace.v, D, i), D, { separatorEvery: dh })}
          </div>
        )
      case 'scores': {
        const q = row(layerTrace.q, D, i).slice(head * dh, (head + 1) * dh)
        const raw = Array.from({ length: i + 1 }, (_, j) => {
          const k = row(layerTrace.k, D, j).slice(head * dh, (head + 1) * dh)
          let dot = 0
          for (let e = 0; e < dh; e++) dot += q[e] * k[e]
          return dot
        })
        const scaled = raw.map((s) => s / Math.sqrt(dh))
        const probs = softmax(scaled)
        return (
          <table className="sim-table">
            <thead>
              <tr>
                <th>j</th>
                <th>token</th>
                <th>q·k</th>
                <th>÷√{dh}</th>
                <th>softmax</th>
              </tr>
            </thead>
            <tbody>
              {raw.map((value, j) => (
                <tr key={j} className={probs[j] === Math.max(...probs) ? 'is-picked' : undefined}>
                  <td>{j}</td>
                  <td>{displayChar(trace.tokens[j])}</td>
                  <td>{fmt(value)}</td>
                  <td>{fmt(scaled[j])}</td>
                  <td>
                    <span className="sim-bar-track">
                      <span className="sim-bar-fill" style={{ width: `${probs[j] * 100}%` }} />
                    </span>{' '}
                    {fmt(probs[j])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      case 'weighted':
        return (
          <div className="sim-eq">
            {one(`A[${i}]`, Float32Array.from(attention[i]), T, { scale: 1 })}
            <Op>· V_{head + 1} →</Op>
            {one(
              `head_${head + 1}`,
              row(layerTrace.headsConcat, D, i).slice(head * dh, (head + 1) * dh),
              dh,
            )}
            <Op>concat →</Op>
            {one('concat', row(layerTrace.headsConcat, D, i), D, { separatorEvery: dh })}
          </div>
        )
      case 'wo':
        return (
          <div className="sim-eq">
            {one('concat', row(layerTrace.headsConcat, D, i), D, { separatorEvery: dh })}
            <Op>·</Op>
            <Heat title="W_O" data={weights.wo} rows={D} cols={D} stages={[stage]} active={stage} />
            <Op>+</Op>
            {one('b_O', weights.bo, D)}
            <Op>=</Op>
            {one('attn', row(layerTrace.projected, D, i), D)}
          </div>
        )
      case 'add1':
        return (
          <div className="sim-eq">
            {one('x', layer === 0 ? x : row(trace.layers[layer - 1].residual2, D, i), D)}
            <Op>+</Op>
            {one('attn', row(layerTrace.projected, D, i), D)}
            <Op>=</Op>
            {one("x'", row(layerTrace.residual1, D, i), D)}
          </div>
        )
      case 'ln2':
        return (
          <div className="sim-eq">
            {one("x'", row(layerTrace.residual1, D, i), D)}
            <Op>→ LN₂ →</Op>
            {one('h', row(layerTrace.ln2, D, i), D)}
          </div>
        )
      case 'ffn1':
        return (
          <div className="sim-eq">
            {one('h', row(layerTrace.ln2, D, i), D)}
            <Op>·</Op>
            <Heat
              title="W₁"
              data={weights.w1}
              rows={D}
              cols={F}
              cell={WIDE_CELL}
              stages={[stage]}
              active={stage}
            />
            <Op>+ b₁ =</Op>
            {one('u', row(layerTrace.ffnPre, F, i), F, { cell: WIDE_CELL })}
          </div>
        )
      case 'relu':
        return (
          <div className="sim-eq">
            {one('u', row(layerTrace.ffnPre, F, i), F, { cell: WIDE_CELL })}
            <Op>→ max(0, ·) →</Op>
            {one('a', row(layerTrace.ffnHidden, F, i), F, { cell: WIDE_CELL, markZeros: true })}
          </div>
        )
      case 'ffn2':
        return (
          <div className="sim-eq">
            {one('a', row(layerTrace.ffnHidden, F, i), F, { cell: WIDE_CELL, markZeros: true })}
            <Op>·</Op>
            <Heat
              title="W₂"
              data={weights.w2}
              rows={F}
              cols={D}
              cell={WIDE_CELL}
              stages={[stage]}
              active={stage}
            />
            <Op>+ b₂ =</Op>
            {one('ffn', row(layerTrace.ffnOut, D, i), D)}
          </div>
        )
      case 'add2':
        return (
          <div className="sim-eq">
            {one("x'", row(layerTrace.residual1, D, i), D)}
            <Op>+</Op>
            {one('ffn', row(layerTrace.ffnOut, D, i), D)}
            <Op>=</Op>
            {one("x''", row(layerTrace.residual2, D, i), D)}
          </div>
        )
      case 'lnf':
        return (
          <div className="sim-eq">
            {one("x''_T", row(trace.layers[nLayers - 1].residual2, D, T - 1), D)}
            <Op>→ LN_f →</Op>
            {one('h_T', row(trace.final, D, T - 1), D)}
          </div>
        )
      case 'logits':
        return (
          <div className="sim-eq">
            {one('h_T', row(trace.final, D, T - 1), D)}
            <Op>·</Op>
            <Heat
              title="Eᵀ (top 10 rows)"
              data={topEmbeddingRows}
              rows={TOP_K}
              cols={D}
              cell={CELL}
              stages={[stage]}
              active={stage}
            />
            <Op>=</Op>
            <Bars
              title="z"
              items={topIds.map((c) => ({
                key: String(c.id),
                label: displayChar(c.token),
                value: trace.logits[c.id],
                display: fmt(trace.logits[c.id]),
              }))}
              max={logitMax}
              stages={[stage]}
              active={stage}
            />
          </div>
        )
      case 'softmax':
      case 'sample':
        return (
          <div className="sim-eq">
            <Bars
              title={`p (T = ${temperature.toFixed(1)})`}
              items={topIds.map((c) => ({
                key: String(c.id),
                label: displayChar(c.token),
                value: c.probability,
                display: `${(c.probability * 100).toFixed(1)}%`,
              }))}
              max={1}
              stages={[stage]}
              active={stage}
              highlightKey={pickedKey}
            />
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="sim">
      <header className="sim-header">
        <div className="sim-brand">
          <span className="brand-mark" aria-hidden="true">
            T
          </span>
          <span>{L.title}</span>
        </div>
        <p className="sim-subtitle">{L.subtitle}</p>
        <div className="sim-header-actions">
          <fieldset className="segmented">
            <legend className="visually-hidden">Language</legend>
            {(['ko', 'en'] as const).map((value) => (
              <label
                key={value}
                className={`segmented-option${lang === value ? ' is-active' : ''}`}
              >
                <input
                  type="radio"
                  name="sim-lang"
                  value={value}
                  checked={lang === value}
                  onChange={() => setLang(value)}
                />
                <span className="segmented-label">{value === 'ko' ? '한국어' : 'English'}</span>
              </label>
            ))}
          </fieldset>
          <a className="sim-link" href="./transformer-explain.html">
            {L.fullPage}
          </a>
        </div>
      </header>

      <section className="sim-controls">
        <div className="sim-prompt">
          <label htmlFor="sim-prompt">{L.prompt}</label>
          <textarea
            id="sim-prompt"
            value={promptText}
            maxLength={PROMPT_MAX}
            rows={2}
            spellCheck={false}
            onChange={(event) => {
              setPrompt(event.target.value)
              setGenerated('')
              setFocus(null)
            }}
          />
          <p className="sim-hint">{L.promptHint(PROMPT_MAX)}</p>
        </div>
        <div className="sim-run">
          <p className={`sim-mode sim-mode-${mode}`} aria-live="polite">
            {mode === 'prefill' ? L.prefill : L.decode(decodeStep, T - 1)}
          </p>
          <div className="demo-actions">
            <button
              type="button"
              className="button-primary"
              onClick={() => decode(1)}
              disabled={!canDecode}
            >
              {L.decodeOne}
            </button>
            <button type="button" onClick={() => decode(10)} disabled={!canDecode}>
              {L.decodeTen}
            </button>
            <button type="button" onClick={reset} disabled={generated === ''}>
              {L.reset}
            </button>
          </div>
          {!canDecode && trace && <p className="sim-hint">{L.contextFull(context)}</p>}
          <div className="sim-selects">
            <label htmlFor="sim-temp">{L.temperature}</label>
            <input
              id="sim-temp"
              type="range"
              min={0.1}
              max={3}
              step={0.1}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
            <output htmlFor="sim-temp">{temperature.toFixed(1)}</output>
            <label htmlFor="sim-layer">{L.layer}</label>
            <select
              id="sim-layer"
              value={layer}
              onChange={(event) => setLayer(Number(event.target.value))}
            >
              {Array.from({ length: nLayers }, (_, i) => (
                <option key={i} value={i}>
                  {L.layerOption(i)}
                </option>
              ))}
            </select>
            <label htmlFor="sim-head">{L.head}</label>
            <select
              id="sim-head"
              value={head}
              onChange={(event) => setHead(Number(event.target.value))}
            >
              {Array.from({ length: H }, (_, i) => (
                <option key={i} value={i}>
                  {L.headOption(i)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <section className="sim-stepper" aria-label="stages">
        <div className="sim-stepper-row">
          <button
            type="button"
            onClick={() => setStageIndex((i) => Math.max(0, i - 1))}
            disabled={stageIndex === 0}
            aria-label={L.prev}
          >
            ◀
          </button>
          <ol className="sim-stages">
            {STAGES.map((id, i) => (
              <li
                key={id}
                className={`group-${STAGE_GROUP[id]}${i === stageIndex ? ' is-active' : ''}`}
              >
                <button
                  type="button"
                  onClick={() => setStageIndex(i)}
                  aria-current={i === stageIndex ? 'step' : undefined}
                >
                  <span className="sim-stage-num">{i + 1}</span>
                  {L.stageNames[id]}
                </button>
              </li>
            ))}
          </ol>
          <button
            type="button"
            onClick={() => setStageIndex((i) => Math.min(STAGES.length - 1, i + 1))}
            disabled={stageIndex === STAGES.length - 1}
            aria-label={L.next}
          >
            ▶
          </button>
          <button
            type="button"
            className={autoplay ? 'is-on' : undefined}
            onClick={() => setAutoplay((a) => !a)}
            aria-pressed={autoplay}
          >
            {autoplay ? L.autoplayStop : L.autoplay}
          </button>
          <label className="sim-speed">
            <span>{L.speed}</span>
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              {SPEEDS.map((value) => (
                <option key={value} value={value}>
                  {value}×
                </option>
              ))}
            </select>
          </label>
          <label className="sim-loop">
            <input
              type="checkbox"
              checked={loopDecode}
              onChange={(event) => setLoopDecode(event.target.checked)}
            />
            <span>{L.loopDecode}</span>
          </label>
        </div>
        <p className="sim-formula" aria-live="polite">
          <span className={`sim-group-tag group-${STAGE_GROUP[stage]}`}>
            {L.groups[STAGE_GROUP[stage]]}
          </span>
          <strong>{L.stageNames[stage]}</strong>
          <code>{formatFormula(stage, shapeValues)}</code>
        </p>
      </section>

      {!trace || !layerTrace || !attention ? (
        <p className="sim-empty">{L.empty}</p>
      ) : (
        <>
          <div className="sim-diagram" data-testid="sim-diagram">
            <section
              className={`sim-panel sim-panel-input sim-el${stage === 'tokens' ? ' is-active' : ''}`}
            >
              <h3>{L.panels.tokens}</h3>
              <ol className="sim-tokens" aria-label={L.focus}>
                {trace.tokens.map((tok, i) => {
                  const isNew = mode === 'decode' && i === T - 1
                  return (
                    <li
                      key={i}
                      className={`${i === focusIndex ? 'is-focus' : ''}${isNew ? ' is-new' : ''}`}
                    >
                      <button
                        type="button"
                        onClick={() => setFocus(i)}
                        aria-pressed={i === focusIndex}
                      >
                        <span className="sim-pos">{i}</span>
                        <span className="sim-char">{displayChar(tok)}</span>
                        <span className="sim-id">{trace.ids[i]}</span>
                        {isNew && <span className="sim-badge">{L.newToken}</span>}
                      </button>
                    </li>
                  )
                })}
              </ol>
            </section>

            <Op>→</Op>

            <section className="sim-panel">
              <h3>{L.panels.embedding}</h3>
              <div className="sim-row">
                <Strip
                  title="E[id]"
                  data={Float32Array.from(trace.tokenEmbeddings.flatMap((r) => Array.from(r)))}
                  rows={T}
                  cols={D}
                  highlightRow={focusIndex}
                  stages={['embed']}
                  active={stage}
                  onSelectRow={setFocus}
                />
                <Op>+</Op>
                <Strip
                  title="P[pos]"
                  data={Float32Array.from(trace.positionEmbeddings.flatMap((r) => Array.from(r)))}
                  rows={T}
                  cols={D}
                  highlightRow={focusIndex}
                  stages={['embed']}
                  active={stage}
                  onSelectRow={setFocus}
                />
                <Op>=</Op>
                <Strip
                  title="x"
                  data={Float32Array.from(trace.embeddings.flatMap((r) => Array.from(r)))}
                  rows={T}
                  cols={D}
                  highlightRow={focusIndex}
                  stages={['embed', 'add1']}
                  active={stage}
                  onSelectRow={setFocus}
                />
              </div>
            </section>

            <Op>→</Op>

            <section className="sim-panel sim-block">
              <h3>
                {L.panels.block(layer)}
                <span className="sim-layer-tabs">
                  {Array.from({ length: nLayers }, (_, i) => (
                    <button
                      key={i}
                      type="button"
                      className={i === layer ? 'is-active' : undefined}
                      onClick={() => setLayer(i)}
                      aria-pressed={i === layer}
                    >
                      {i + 1}
                    </button>
                  ))}
                </span>
              </h3>
              <div className="sim-subrow">
                <h4>{L.panels.attentionRow}</h4>
                <div className="sim-row">
                  <Strip
                    title="LN₁"
                    data={layerTrace.ln1}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    stages={['ln1']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <div className={`sim-el sim-weights${stage === 'qkv' ? ' is-active' : ''}`}>
                    <Heat
                      title="W_Q"
                      data={weights.wq}
                      rows={D}
                      cols={D}
                      stages={['qkv']}
                      active={stage}
                    />
                    <Heat
                      title="W_K"
                      data={weights.wk}
                      rows={D}
                      cols={D}
                      stages={['qkv']}
                      active={stage}
                    />
                    <Heat
                      title="W_V"
                      data={weights.wv}
                      rows={D}
                      cols={D}
                      stages={['qkv']}
                      active={stage}
                    />
                  </div>
                  <Op>→</Op>
                  <Strip
                    title="Q"
                    data={layerTrace.q}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    separatorEvery={dh}
                    stages={['qkv', 'heads']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Strip
                    title={`K${cacheRows ? ` (${L.cache} ${cacheRows})` : ''}`}
                    data={layerTrace.k}
                    rows={T}
                    cols={D}
                    dimRowsBefore={cacheRows}
                    separatorEvery={dh}
                    stages={['qkv', 'heads']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Strip
                    title={`V${cacheRows ? ` (${L.cache} ${cacheRows})` : ''}`}
                    data={layerTrace.v}
                    rows={T}
                    cols={D}
                    dimRowsBefore={cacheRows}
                    separatorEvery={dh}
                    stages={['qkv', 'heads']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>→</Op>
                  <figure className={`sim-el sim-attn${stage === 'scores' ? ' is-active' : ''}`}>
                    <figcaption>
                      A · {L.headOption(head)}{' '}
                      <span className="sim-shape">
                        {T}×{T}
                      </span>
                    </figcaption>
                    <MatrixCanvas
                      rows={T}
                      cols={T}
                      value={(r, c) => attention[r][c]}
                      scale={1}
                      mode="sequential"
                      cellWidth={Math.max(4, Math.min(ROW, Math.floor(220 / T)))}
                      cellHeight={ROW}
                      gap={1}
                      highlightRow={focusIndex}
                      dimRowsBefore={cacheRows}
                      label={`attention ${T}×${T}`}
                      tooltip={(r, c, v) =>
                        `A[${r}, ${c}] ${displayChar(trace.tokens[r])} → ${displayChar(trace.tokens[c])} = ${v.toFixed(3)}`
                      }
                      onSelectRow={setFocus}
                    />
                    <div className="sim-minis">
                      {layerTrace.attentions.map((matrix, h) => (
                        <button
                          key={h}
                          type="button"
                          className={`sim-mini${h === head ? ' is-active' : ''}`}
                          onClick={() => setHead(h)}
                          aria-pressed={h === head}
                          aria-label={L.headOption(h)}
                        >
                          <MatrixCanvas
                            rows={T}
                            cols={T}
                            value={(r, c) => matrix[r][c]}
                            scale={1}
                            mode="sequential"
                            cellWidth={Math.max(1, Math.floor(40 / T))}
                            cellHeight={Math.max(1, Math.floor(40 / T))}
                            label={L.headOption(h)}
                          />
                        </button>
                      ))}
                    </div>
                  </figure>
                  <Op>·V →</Op>
                  <Strip
                    title="concat"
                    data={layerTrace.headsConcat}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    separatorEvery={dh}
                    stages={['weighted']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <Heat
                    title="W_O"
                    data={weights.wo}
                    rows={D}
                    cols={D}
                    stages={['wo']}
                    active={stage}
                  />
                  <Op>→</Op>
                  <Strip
                    title="attn"
                    data={layerTrace.projected}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['wo']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>+x →</Op>
                  <Strip
                    title="x'"
                    data={layerTrace.residual1}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['add1']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                </div>
              </div>
              <div className="sim-subrow">
                <h4>{L.panels.ffnRow}</h4>
                <div className="sim-row">
                  <Strip
                    title="LN₂"
                    data={layerTrace.ln2}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['ln2']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <Heat
                    title="W₁"
                    data={weights.w1}
                    rows={D}
                    cols={F}
                    cell={WIDE_CELL}
                    stages={['ffn1']}
                    active={stage}
                  />
                  <Op>→</Op>
                  <Strip
                    title="u"
                    data={layerTrace.ffnPre}
                    rows={T}
                    cols={F}
                    cell={WIDE_CELL}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['ffn1']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>ReLU →</Op>
                  <Strip
                    title="a"
                    data={layerTrace.ffnHidden}
                    rows={T}
                    cols={F}
                    cell={WIDE_CELL}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    markZeros
                    stages={['relu']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <Heat
                    title="W₂"
                    data={weights.w2}
                    rows={F}
                    cols={D}
                    cell={WIDE_CELL}
                    stages={['ffn2']}
                    active={stage}
                  />
                  <Op>→</Op>
                  <Strip
                    title="ffn"
                    data={layerTrace.ffnOut}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['ffn2']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>+x' →</Op>
                  <Strip
                    title="x''"
                    data={layerTrace.residual2}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['add2']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                </div>
              </div>
            </section>

            <Op>→</Op>

            <section className="sim-panel sim-panel-output">
              <h3>{L.panels.output}</h3>
              <div className="sim-row sim-row-wrap">
                <Strip
                  title="h_T"
                  data={row(trace.final, D, T - 1)}
                  rows={1}
                  cols={D}
                  stages={['lnf']}
                  active={stage}
                />
                <Op>· Eᵀ →</Op>
                <Bars
                  title={`z · ${L.topK}`}
                  items={topIds.map((c) => ({
                    key: String(c.id),
                    label: displayChar(c.token),
                    value: trace.logits[c.id],
                    display: fmt(trace.logits[c.id]),
                  }))}
                  max={logitMax}
                  stages={['logits']}
                  active={stage}
                />
                <Op>softmax / T →</Op>
                <Bars
                  title={`p · ${L.probability}`}
                  items={topIds.map((c) => ({
                    key: String(c.id),
                    label: displayChar(c.token),
                    value: c.probability,
                    display: `${(c.probability * 100).toFixed(1)}%`,
                  }))}
                  max={1}
                  stages={['softmax', 'sample']}
                  active={stage}
                  highlightKey={pickedKey}
                />
              </div>
            </section>
          </div>

          <section className="sim-focus" aria-live="polite">
            <h3>
              {L.focusTitle(focusIndex, displayChar(trace.tokens[focusIndex]))}
              <span className="sim-focus-stage">{L.stageNames[stage]}</span>
            </h3>
            <p className="sim-detail">{L.detail[stage]}</p>
            {renderFocus()}
          </section>
        </>
      )}

      <footer className="sim-footer">
        <p>{L.modelNote(model.training.paramCount, nLayers, H, D, model.vocab.length)}</p>
      </footer>
    </div>
  )
}
