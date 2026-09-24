import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { sampleIndex } from '../lib/math'
import {
  entryValue,
  kvBytesPerToken,
  loadTinyDsv41,
  type Dsv41LayerTrace,
  type Dsv41Trace,
  type EntryTrace,
} from '../lib/tinyDsv41'
import { CELL, displayChar, fmt, maxAbs, ROW, WIDE_CELL } from '../simulator/format'
import { MatrixCanvas } from '../simulator/MatrixCanvas'
import { Bars, Heat, Op, Strip, type StripProps } from '../simulator/parts'
import { LABELS, REAL_MODEL, type SimLang } from './labels'
import {
  formatFormula,
  STAGE_GROUP,
  STAGES,
  stageState,
  type StageId,
  type StageState,
} from './stages'

const DEFAULT_PROMPT: Record<SimLang, string> = {
  ko: 'DeepSeek-V4.1-Flash의 KV cache는',
  en: 'The KV cache of DeepSeek-V4.1-Flash',
}
const PROMPT_MAX = 40
const TOP_K = 10
const AUTOPLAY_MS = 1600
const SPEEDS = [0.5, 1, 2, 4]

interface SimulatorProps {
  /** 뽑기에 쓰는 난수. 테스트에서 고정한다. */
  random?: () => number
  initialLang?: SimLang
}

function row(data: Float32Array, cols: number, r: number): Float32Array {
  return data.subarray(r * cols, (r + 1) * cols)
}

function countOnes(mask: Uint8Array | null, from: number, to: number): number {
  if (!mask) return 0
  let count = 0
  for (let i = from; i < to; i++) count += mask[i]
  return count
}

/** 0/1 마스크를 순차 색으로 그리는 작은 히트맵 */
function MaskHeat({
  title,
  mask,
  rows,
  cols,
  highlightRow,
  stages,
  active,
  inherited = false,
  tooltip,
}: {
  title: string
  mask: Uint8Array
  rows: number
  cols: number
  highlightRow: number | null
  stages: StageId[]
  active: StageId
  inherited?: boolean
  tooltip?: (r: number, c: number, v: number) => string
}) {
  const isActive = stages.includes(active)
  return (
    <figure
      className={`sim-el sim-attn${isActive ? ' is-active' : ''}${inherited ? ' is-inherited' : ''}`}
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
        value={(r, c) => mask[r * cols + c]}
        scale={1}
        mode="sequential"
        cellWidth={Math.max(3, Math.min(ROW, Math.floor(200 / Math.max(1, cols))))}
        cellHeight={ROW}
        gap={1}
        highlightRow={highlightRow}
        label={`${title} ${rows}×${cols}`}
        tooltip={tooltip ?? ((r, c, v) => `${title}[${r}, ${c}] = ${v}`)}
      />
    </figure>
  )
}

function StateNote({ children }: { children: ReactNode }) {
  return <p className="sim-state-note">{children}</p>
}

export function Dsv41Simulator({ random = Math.random, initialLang }: SimulatorProps) {
  const model = useMemo(() => loadTinyDsv41(), [])
  const [lang, setLang] = useState<SimLang>(
    () =>
      initialLang ??
      (new URLSearchParams(window.location.search).get('lang') === 'en' ? 'en' : 'ko'),
  )
  const L = LABELS[lang]
  const [prompt, setPrompt] = useState<string | null>(null)
  const [generated, setGenerated] = useState('')
  const [temperature, setTemperature] = useState(1)
  const [layer, setLayer] = useState(1)
  const [head, setHead] = useState(0)
  const [focus, setFocus] = useState<number | null>(null)
  const [stageIndex, setStageIndex] = useState(0)
  const [autoplay, setAutoplay] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loopDecode, setLoopDecode] = useState(false)

  const cfg = model.config
  const {
    dModel: D,
    nHeads: H,
    dLatent: DC,
    window: W,
    topK: K,
    poolBlock: B,
    poolBlocks: NB,
    indexerHeads: HI,
    dIndexer: DI,
    experts: E,
    expertsTopK: ET,
    context,
    encoderLayers,
  } = cfg
  const HD = H * DC
  const promptText = prompt ?? DEFAULT_PROMPT[lang]
  const text = promptText + generated
  const ids = useMemo(() => model.encode(text), [model, text])
  const promptTokens = Math.min(model.encode(promptText).length, ids.length)
  const decoderStart = model.decoderStartFor(promptTokens)
  const trace: Dsv41Trace | null = useMemo(
    () => (ids.length > 0 ? model.forwardTrace(ids, decoderStart) : null),
    [model, ids, decoderStart],
  )
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
  const spec = cfg.layers[layer]
  const isDecoder = layer >= encoderLayers
  const layerTrace: Dsv41LayerTrace | null = trace?.layers[layer] ?? null
  const entries: EntryTrace | null = layerTrace?.entries ?? null
  const n = entries?.n ?? 0
  const decoderFullLayer = cfg.layers.findIndex((s, i) => s.mode === 'full' && i >= encoderLayers)
  const poolMask: Uint8Array | null =
    layerTrace?.pool ??
    (spec.mode === 'reindex' ? (trace?.layers[decoderFullLayer].pool ?? null) : null)
  const weights = model.layerWeights(layer)
  const bytes = kvBytesPerToken(cfg)
  const state = (id: StageId): StageState => stageState(id, spec.mode, isDecoder)
  const stateOf = state(stage)
  const kvSourceLabel = (index: number) => {
    const order = bytes.sources.findIndex((s) => s.layer === index)
    return order >= 0 ? order + 1 : 0
  }

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
        model.forwardTrace(currentIds, decoderStart).logits,
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

  const shapeValues = {
    T,
    D,
    H,
    dc: DC,
    HD,
    n,
    k: K,
    w: W,
    m: entries?.m ?? spec.m ?? 1,
    G: DC / cfg.fp4Group,
    mainB: bytes.mainEntry,
    idxB: bytes.indexerEntry,
    DI,
    HI,
    NB,
    B,
    poolSize: NB * B,
    E,
    ET,
    V: model.vocab.length,
  }
  const topIds = distribution.slice(0, TOP_K)
  const logitMax = trace ? maxAbs(trace.logits) : 1
  const topEmbeddingRows = new Float32Array(TOP_K * D)
  topIds.forEach((candidate, i) =>
    topEmbeddingRows.set(
      model.tokenEmbedding.subarray(candidate.id * D, (candidate.id + 1) * D),
      i * D,
    ),
  )
  const pickedKey = mode === 'decode' && trace ? String(trace.ids[T - 1]) : undefined
  const windowStart = layerTrace ? Math.max(layerTrace.start, focusIndex - W + 1) : 0
  const entryTokens = (i: number) => {
    if (!entries) return ''
    const positions = Array.from({ length: entries.m }, (_, s) => entries.entryPos[i] + s).filter(
      (p) => p < T,
    )
    return positions.map((p) => displayChar(trace!.tokens[p])).join('')
  }
  const focusEntry = entries ? Math.min(n - 1, Math.floor(focusIndex / entries.m)) : -1

  // 마지막 토큰이 읽은 양 (KV cache 카드)
  const reads = trace
    ? trace.layers.reduce(
        (acc, lt) => {
          const t = T - 1
          if (lt.range) acc.indexer += countOnes(lt.range, t * lt.n, (t + 1) * lt.n)
          if (lt.selected) acc.main += countOnes(lt.selected, t * lt.n, (t + 1) * lt.n)
          acc.swa += Math.max(0, t - Math.max(lt.start, t - W + 1) + 1)
          return acc
        },
        { indexer: 0, main: 0, swa: 0 },
      )
    : { indexer: 0, main: 0, swa: 0 }

  const renderFocus = () => {
    if (!trace || !layerTrace) return null
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
    const skipped = isDecoder && i < layerTrace.start
    const layerStages: StageId[] = [
      'ln1',
      'q',
      'swa',
      'indexerQ',
      'scores',
      'pool',
      'topk',
      'gather',
      'logits',
      'merge',
      'weighted',
      'wo',
      'add1',
      'ln2',
      'router',
      'experts',
      'add2',
    ]
    if (skipped && layerStages.includes(stage))
      return (
        <StateNote>
          {L.encOnly}: {L.detail.tokens}
        </StateNote>
      )
    if (stateOf === 'none') return <StateNote>{L.stateNote.none}</StateNote>
    const inheritedFrom =
      stateOf === 'inherit'
        ? stage === 'topk'
          ? (layerTrace.selectedFrom ?? spec.src ?? 0)
          : stage === 'pool'
            ? decoderFullLayer
            : (spec.src ?? 0)
        : null
    const inheritNote =
      inheritedFrom !== null ? <StateNote>{L.stateNote.inherit(inheritedFrom)}</StateNote> : null
    const x = row(trace.x0, D, i)
    const xIn = layer === 0 ? x : row(trace.layers[layer - 1].x2, D, i)
    switch (stage) {
      case 'tokens':
        return (
          <p className="sim-eq-text">
            "{token}" → id {trace.ids[i]} → E[{trace.ids[i]}]
            {i < decoderStart ? ` · ${L.encOnly}` : ''}
          </p>
        )
      case 'embed':
        return (
          <div className="sim-eq">
            {one(`E[${trace.ids[i]}]`, row(trace.tokenEmbeddings, D, i), D)}
            <Op>+</Op>
            {one(`P[${i}]`, row(trace.positionEmbeddings, D, i), D)}
            <Op>=</Op>
            {one('x', x, D)}
          </div>
        )
      case 'ln1':
        return (
          <div className="sim-eq">
            {one('x', xIn, D)}
            <Op>→ LN₁ →</Op>
            {one('h', row(layerTrace.ln1, D, i), D)}
          </div>
        )
      case 'q':
        return (
          <div className="sim-eq">
            {one('h', row(layerTrace.ln1, D, i), D)}
            <Op>·</Op>
            <Heat
              title="W_Q"
              data={weights.wq}
              rows={D}
              cols={HD}
              stages={[stage]}
              active={stage}
            />
            <Op>+ b_Q =</Op>
            {one('q', row(layerTrace.q, HD, i), HD, { separatorEvery: DC })}
          </div>
        )
      case 'swa':
        return (
          <div className="sim-eq-stack">
            {(
              [
                ['k', weights.wk, layerTrace.kSwa, 'W_K'],
                ['v', weights.wv, layerTrace.vSwa, 'W_V'],
              ] as const
            ).map(([name, w, out, wName]) => (
              <div className="sim-eq" key={name}>
                {one('h', row(layerTrace.ln1, D, i), D)}
                <Op>·</Op>
                <Heat title={wName} data={w} rows={D} cols={HD} stages={[stage]} active={stage} />
                <Op>=</Op>
                {one(name, row(out, HD, i), HD, { separatorEvery: DC })}
              </div>
            ))}
            <p className="sim-eq-text">
              {L.window}: s ∈ [{windowStart}, {i}] →{' '}
              {Array.from({ length: i - windowStart + 1 }, (_, s) =>
                displayChar(trace.tokens[windowStart + s]),
              ).join(' ')}
            </p>
          </div>
        )
      case 'compress': {
        if (!entries) return null
        const e = focusEntry
        const partial = entries.m === 2 && entries.entryPos[e] === i
        return (
          <div className="sim-eq-stack">
            {inheritNote}
            <div className="sim-eq">
              {one(
                `[h_${entries.entryPos[e]}${entries.m === 2 ? `; h_${entries.entryPos[e] + 1}` : ''}]`,
                row(entries.merged, entries.m * D, e),
                entries.m * D,
                { cell: WIDE_CELL, separatorEvery: D },
              )}
              <Op>·</Op>
              <Heat
                title="W_c"
                data={model.layerWeights(entries.layer).wc!}
                rows={entries.m * D}
                cols={DC}
                stages={[stage]}
                active={stage}
              />
              <Op>+ b_c =</Op>
              {one(`c_${e}`, row(partial ? entries.cPartRaw : entries.cRaw, DC, e), DC)}
            </div>
            {partial && <p className="sim-eq-text">{L.partial}</p>}
          </div>
        )
      }
      case 'fp4': {
        if (!entries) return null
        const e = focusEntry
        const G = DC / cfg.fp4Group
        return (
          <div className="sim-eq-stack">
            {inheritNote}
            <div className="sim-eq">
              {one(`c_${e}`, row(entries.cRaw, DC, e), DC)}
              <Op>→ FP4 →</Op>
              <p className="sim-eq-text">
                codes [{Array.from(entries.codes.subarray(e * DC, (e + 1) * DC)).join(' ')}]
                <br />
                scale [
                {Array.from(entries.scales.subarray(e * G, (e + 1) * G))
                  .map((v) => v.toFixed(4))
                  .join(' ')}
                ]
              </p>
              <Op>→ dequant →</Op>
              {one(`C_${e}`, row(entries.c, DC, e), DC)}
            </div>
            <p className="sim-eq-text">
              {DC} × 4 bit + {G} × 8 bit = {bytes.mainEntry} B (real: {REAL_MODEL.latent} ch →{' '}
              {REAL_MODEL.mainEntryBytes} B)
            </p>
          </div>
        )
      }
      case 'indexerK': {
        if (!entries) return null
        const e = focusEntry
        return (
          <div className="sim-eq-stack">
            {inheritNote}
            <div className="sim-eq">
              {one(`C_${e}`, row(entries.c, DC, e), DC)}
              <Op>·</Op>
              <Heat
                title="W_I"
                data={model.layerWeights(entries.layer).wi!}
                rows={DC}
                cols={DI}
                stages={[stage]}
                active={stage}
              />
              <Op>→ MXFP4 (scale {entries.kiScale[e]}) →</Op>
              {one(`K^I_${e}`, row(entries.ki, DI, e), DI)}
            </div>
            <p className="sim-eq-text">
              {DI} × 4 bit + 1 B = {bytes.indexerEntry} B (real: {REAL_MODEL.indexerDim} ch →{' '}
              {REAL_MODEL.indexerEntryBytes} B)
            </p>
          </div>
        )
      }
      case 'indexerQ':
        if (!layerTrace.qi || !layerTrace.wIdx) return null
        return (
          <div className="sim-eq">
            {one('h', row(layerTrace.ln1, D, i), D)}
            <Op>·</Op>
            <Heat
              title="W^I_Q"
              data={weights.wqi!}
              rows={D}
              cols={HI * DI}
              stages={[stage]}
              active={stage}
            />
            <Op>=</Op>
            {one('q^I', row(layerTrace.qi, HI * DI, i), HI * DI, { separatorEvery: DI })}
            <Op>, w =</Op>
            {one('w', row(layerTrace.wIdx, HI, i), HI, { cell: 12 })}
          </div>
        )
      case 'scores':
      case 'topk': {
        if (!entries || !layerTrace.scores || !layerTrace.range || !layerTrace.selected) {
          if (stateOf === 'inherit' && entries && layerTrace.selected) {
            const chosen = Array.from({ length: n }, (_, j) => j).filter(
              (j) => layerTrace.selected![i * n + j],
            )
            return (
              <div className="sim-eq-stack">
                {inheritNote}
                <p className="sim-eq-text">
                  T_t = {'{'}
                  {chosen.join(', ')}
                  {'}'} → {chosen.map(entryTokens).join(' | ')}
                </p>
              </div>
            )
          }
          return null
        }
        const rows = Array.from({ length: n }, (_, j) => j).filter(
          (j) => layerTrace.range![i * n + j],
        )
        const qi = layerTrace.qi!
        const wIdx = layerTrace.wIdx!
        const maxScore = Math.max(1e-9, ...rows.map((j) => layerTrace.scores![i * n + j]))
        return (
          <table className="sim-table">
            <thead>
              <tr>
                <th>j</th>
                <th>entry</th>
                {Array.from({ length: HI }, (_, h) => (
                  <th key={h}>
                    w_{h + 1}·ReLU(q^I_{h + 1}·K^I)
                  </th>
                ))}
                <th>S</th>
                <th>Top-{K}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((j) => {
                const key =
                  entries.m === 2 && entries.entryPos[j] === i ? entries.kiPart : entries.ki
                const terms = Array.from({ length: HI }, (_, h) => {
                  let dot = 0
                  for (let d = 0; d < DI; d++) dot += qi[i * HI * DI + h * DI + d] * key[j * DI + d]
                  return wIdx[i * HI + h] * Math.max(0, dot)
                })
                const s = layerTrace.scores![i * n + j]
                const picked = layerTrace.selected![i * n + j] === 1
                return (
                  <tr key={j} className={picked ? 'is-picked' : undefined}>
                    <td>{j}</td>
                    <td>{entryTokens(j)}</td>
                    {terms.map((v, h) => (
                      <td key={h}>{fmt(v)}</td>
                    ))}
                    <td>
                      <span className="sim-bar-track">
                        <span
                          className="sim-bar-fill"
                          style={{ width: `${Math.max(0, s / maxScore) * 100}%` }}
                        />
                      </span>{' '}
                      {s.toFixed(3)}
                    </td>
                    <td>{picked ? '✓' : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )
      }
      case 'pool': {
        if (!entries || !poolMask) return null
        const source = trace.layers[decoderFullLayer]
        const nb = Math.ceil(n / B)
        return (
          <div className="sim-eq-stack">
            {inheritNote}
            <table className="sim-table">
              <thead>
                <tr>
                  <th>block</th>
                  <th>entries</th>
                  <th>max S (L{decoderFullLayer})</th>
                  <th>pool</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: nb }, (_, b) => {
                  const members = Array.from({ length: B }, (_, s) => b * B + s).filter(
                    (j) => j < n && source.range![i * n + j],
                  )
                  const max = members.length
                    ? Math.max(...members.map((j) => source.scores![i * n + j]))
                    : null
                  const inPool = members.some((j) => poolMask[i * n + j])
                  return (
                    <tr key={b} className={inPool ? 'is-picked' : undefined}>
                      <td>{b}</td>
                      <td>
                        {members.length ? `${members[0]}–${members[members.length - 1]}` : '—'}
                      </td>
                      <td>{max === null ? '—' : max.toFixed(3)}</td>
                      <td>{inPool ? '✓' : ''}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )
      }
      case 'gather': {
        if (!entries || !layerTrace.selected) return null
        const chosen = Array.from({ length: n }, (_, j) => j).filter(
          (j) => layerTrace.selected![i * n + j],
        )
        return (
          <div className="sim-eq-stack">
            {chosen.map((j) => (
              <div className="sim-eq" key={j}>
                <p className="sim-eq-text">
                  C_{j} ({entryTokens(j)}) codes [
                  {Array.from(entries.codes.subarray(j * DC, (j + 1) * DC)).join(' ')}]
                </p>
                <Op>→ dequant →</Op>
                {one(`c_${j}`, entryValue(entries, i, j), DC)}
              </div>
            ))}
          </div>
        )
      }
      case 'logits':
      case 'merge': {
        const q = row(layerTrace.q, HD, i).subarray(head * DC, (head + 1) * DC)
        const items: { label: string; kind: string; logit: number; weight: number }[] = []
        if (entries && layerTrace.selected)
          for (let j = 0; j < n; j++)
            if (layerTrace.selected[i * n + j])
              items.push({
                label: `c_${j} (${entryTokens(j)})`,
                kind: 'global',
                logit: layerTrace.logitGlobal[(head * T + i) * n + j],
                weight: layerTrace.attnGlobal[(head * T + i) * n + j],
              })
        for (let s = windowStart; s <= i; s++)
          items.push({
            label: `k_${s} (${displayChar(trace.tokens[s])})`,
            kind: 'local',
            logit: layerTrace.logitLocal[(head * T + i) * T + s],
            weight: layerTrace.attnLocal[(head * T + i) * T + s],
          })
        const qNorm = q.length
        return (
          <table className="sim-table">
            <thead>
              <tr>
                <th>{L.headOption(head)}</th>
                <th>branch</th>
                <th>q·key / √{qNorm}</th>
                {stage === 'merge' && <th>softmax</th>}
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => (
                <tr key={index}>
                  <td>{item.label}</td>
                  <td>{item.kind}</td>
                  <td>{fmt(item.logit)}</td>
                  {stage === 'merge' && (
                    <td>
                      <span className="sim-bar-track">
                        <span className="sim-bar-fill" style={{ width: `${item.weight * 100}%` }} />
                      </span>{' '}
                      {item.weight.toFixed(3)}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      case 'weighted': {
        const aGlobal = entries
          ? layerTrace.attnGlobal.subarray((head * T + i) * n, (head * T + i + 1) * n)
          : null
        const aLocal = layerTrace.attnLocal.subarray((head * T + i) * T, (head * T + i + 1) * T)
        return (
          <div className="sim-eq">
            {aGlobal && one(`a_global · ${L.headOption(head)}`, aGlobal, n, { scale: 1 })}
            {one(`a_local`, aLocal, T, { scale: 1 })}
            <Op>→</Op>
            {one(
              `head_${head + 1}`,
              row(layerTrace.o, HD, i).subarray(head * DC, (head + 1) * DC),
              DC,
            )}
            <Op>concat →</Op>
            {one('o', row(layerTrace.o, HD, i), HD, { separatorEvery: DC })}
          </div>
        )
      }
      case 'wo':
        return (
          <div className="sim-eq">
            {one('o', row(layerTrace.o, HD, i), HD, { separatorEvery: DC })}
            <Op>·</Op>
            <Heat
              title="W_O"
              data={weights.wo}
              rows={HD}
              cols={D}
              stages={[stage]}
              active={stage}
            />
            <Op>+ b_O =</Op>
            {one('y', row(layerTrace.y, D, i), D)}
          </div>
        )
      case 'add1':
        return (
          <div className="sim-eq">
            {one('x', xIn, D)}
            <Op>+</Op>
            {one('y', row(layerTrace.y, D, i), D)}
            <Op>=</Op>
            {one("x'", row(layerTrace.x1, D, i), D)}
          </div>
        )
      case 'ln2':
        return (
          <div className="sim-eq">
            {one("x'", row(layerTrace.x1, D, i), D)}
            <Op>→ LN₂ →</Op>
            {one('h', row(layerTrace.ln2, D, i), D)}
          </div>
        )
      case 'router':
        return (
          <table className="sim-table">
            <thead>
              <tr>
                <th>expert</th>
                <th>r = σ(·)</th>
                <th>bias</th>
                <th>r + bias</th>
                <th>gate</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: E }, (_, e) => {
                const r = layerTrace.router[i * E + e]
                const picked = layerTrace.expertMask[i * E + e] === 1
                return (
                  <tr key={e} className={picked ? 'is-picked' : undefined}>
                    <td>{e}</td>
                    <td>{r.toFixed(3)}</td>
                    <td>{weights.rbias[e].toFixed(3)}</td>
                    <td>{(r + weights.rbias[e]).toFixed(3)}</td>
                    <td>{picked ? layerTrace.gates[i * E + e].toFixed(3) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )
      case 'experts': {
        const chosen = Array.from({ length: E }, (_, e) => e).filter(
          (e) => layerTrace.expertMask[i * E + e],
        )
        return (
          <div className="sim-eq-stack">
            <div className="sim-eq">
              {one('shared', row(layerTrace.shared, D, i), D)}
              {chosen.map((e) => (
                <span key={e} className="sim-eq">
                  <Op>+ {layerTrace.gates[i * E + e].toFixed(2)} ×</Op>
                  {one(
                    `expert_${e}`,
                    layerTrace.expertOut.subarray((i * E + e) * D, (i * E + e + 1) * D),
                    D,
                  )}
                </span>
              ))}
              <Op>=</Op>
              {one(
                'ffn',
                Float32Array.from(
                  row(layerTrace.shared, D, i),
                  (v, d) => v + layerTrace.routed[i * D + d],
                ),
                D,
              )}
            </div>
          </div>
        )
      }
      case 'add2':
        return (
          <div className="sim-eq">
            {one("x'", row(layerTrace.x1, D, i), D)}
            <Op>+ ffn =</Op>
            {one("x''", row(layerTrace.x2, D, i), D)}
          </div>
        )
      case 'lnf':
        return (
          <div className="sim-eq">
            {one("x''_T", row(trace.layers[cfg.layers.length - 1].x2, D, T - 1), D)}
            <Op>→ LN_f →</Op>
            {one('h_T', row(trace.final, D, T - 1), D)}
          </div>
        )
      case 'outLogits':
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

  const layerChip = (i: number) => {
    const s = cfg.layers[i]
    const order = kvSourceLabel(i)
    return (
      <button
        key={i}
        type="button"
        className={`sim-lm-chip mode-${s.mode}${i === layer ? ' is-active' : ''}`}
        onClick={() => setLayer(i)}
        aria-pressed={i === layer}
        aria-label={L.layerOption(i, L.modeTitles[s.mode])}
        title={`${L.modeTitles[s.mode]} · ${L.modeNotes[s.mode]}`}
      >
        <span className="sim-lm-num">L{i}</span>
        <span className="sim-lm-mode">{L.modeTitles[s.mode]}</span>
        {order > 0 && <span className="sim-lm-kv">KV {order}</span>}
      </button>
    )
  }

  return (
    <div className="sim sim-dsv41">
      <header className="sim-header">
        <div className="sim-brand">
          <span className="brand-mark" aria-hidden="true">
            D
          </span>
          <span>{L.title}</span>
        </div>
        <p className="sim-subtitle">{L.subtitle}</p>
        <div className="sim-header-actions">
          <span className="sim-author">{L.author}</span>
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
            {mode === 'prefill' ? L.prefill(T, T - decoderStart) : L.decode(decodeStep, T - 1)}
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
              {cfg.layers.map((s, i) => (
                <option key={i} value={i}>
                  {L.layerOption(i, L.modeTitles[s.mode])}
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

        <div className="sim-result" data-testid="sim-result">
          <h3>{L.result}</h3>
          <p className="sim-result-label">{L.generatedText}</p>
          <p className="sim-result-text">
            <span>{promptText}</span>
            <span className="sim-gen">{generated}</span>
            <span className="sim-caret" aria-hidden="true">
              ▌
            </span>
          </p>
          {trace && topIds.length > 0 && (
            <>
              <p className="sim-result-next">
                {L.nextToken}: <strong>{displayChar(topIds[0].token)}</strong>{' '}
                {(topIds[0].probability * 100).toFixed(1)}%
              </p>
              <Bars
                title={L.nextTop}
                items={topIds.slice(0, 5).map((c) => ({
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
            </>
          )}
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
            {STAGES.map((id, i) => {
              const st = state(id)
              return (
                <li
                  key={id}
                  className={`group-${STAGE_GROUP[id]}${i === stageIndex ? ' is-active' : ''}${st !== 'compute' ? ` is-${st}` : ''}`}
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
              )
            })}
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
          {stateOf !== 'compute' && (
            <span className={`sim-state-tag is-${stateOf}`}>
              {stateOf === 'inherit'
                ? L.inherited(
                    stage === 'pool'
                      ? decoderFullLayer
                      : (layerTrace?.selectedFrom ?? spec.src ?? 0),
                  )
                : L.stateNote.none}
            </span>
          )}
        </p>
      </section>

      {!trace || !layerTrace ? (
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
                      className={`${i === focusIndex ? 'is-focus' : ''}${isNew ? ' is-new' : ''}${i < decoderStart ? ' is-enc' : ''}`}
                    >
                      <button
                        type="button"
                        onClick={() => setFocus(i)}
                        aria-pressed={i === focusIndex}
                      >
                        <span className="sim-pos">{i}</span>
                        <span className="sim-char">{displayChar(tok)}</span>
                        <span className="sim-id">{trace.ids[i]}</span>
                        {i < decoderStart && (
                          <span className="sim-badge sim-badge-enc">{L.encOnly}</span>
                        )}
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
                  data={trace.tokenEmbeddings}
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
                  data={trace.positionEmbeddings}
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
                  data={trace.x0}
                  rows={T}
                  cols={D}
                  highlightRow={focusIndex}
                  stages={['embed']}
                  active={stage}
                  onSelectRow={setFocus}
                />
              </div>
            </section>

            <Op>→</Op>

            <section className="sim-panel sim-layer-map" aria-label={L.panels.layerMap}>
              <h3>{L.panels.layerMap}</h3>
              <div className="sim-lm-row">
                <span className="sim-lm-label">{L.encoder}</span>
                <div className="sim-lm-chips">
                  {Array.from({ length: encoderLayers }, (_, i) => layerChip(i))}
                </div>
              </div>
              <p className="sim-lm-arrow">↓ {L.ced(encoderLayers - 1)}</p>
              <div className="sim-lm-row">
                <span className="sim-lm-label">{L.decoder}</span>
                <div className="sim-lm-chips">
                  {Array.from({ length: cfg.layers.length - encoderLayers }, (_, i) =>
                    layerChip(encoderLayers + i),
                  )}
                </div>
              </div>
              <ul className="sim-legend">
                {(['full', 'reindex', 'reuse', 'swa'] as const).map((m) => (
                  <li key={m}>
                    <span className={`sim-legend-swatch mode-${m}`} aria-hidden="true" />
                    <strong>{L.modeTitles[m]}</strong> {L.modeNotes[m]}
                  </li>
                ))}
              </ul>
              <p className="sim-lm-note">
                {L.prefill(T, T - decoderStart)}. {L.reindexNote}.
              </p>
            </section>

            <Op>→</Op>

            <section className={`sim-panel sim-block mode-${spec.mode}`}>
              <h3>
                {L.panels.block(layer, L.modeTitles[spec.mode])}
                <span className="sim-block-note">{L.modeNotes[spec.mode]}</span>
              </h3>

              <div className="sim-subrow">
                <h4>{L.panels.query}</h4>
                <div className="sim-row">
                  <Strip
                    title="LN₁"
                    data={layerTrace.ln1}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={layerTrace.start || cacheRows}
                    stages={['ln1']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <div
                    className={`sim-el sim-weights${stage === 'q' || stage === 'swa' ? ' is-active' : ''}`}
                  >
                    <Heat
                      title="W_Q"
                      data={weights.wq}
                      rows={D}
                      cols={HD}
                      stages={['q']}
                      active={stage}
                    />
                    <Heat
                      title="W_K"
                      data={weights.wk}
                      rows={D}
                      cols={HD}
                      stages={['swa']}
                      active={stage}
                    />
                    <Heat
                      title="W_V"
                      data={weights.wv}
                      rows={D}
                      cols={HD}
                      stages={['swa']}
                      active={stage}
                    />
                  </div>
                  <Op>→</Op>
                  <Strip
                    title="Q (main)"
                    data={layerTrace.q}
                    rows={T}
                    cols={HD}
                    highlightRow={focusIndex}
                    dimRowsBefore={Math.max(layerTrace.start, cacheRows ?? 0) || null}
                    separatorEvery={DC}
                    stages={['q']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Strip
                    title={`K_w (${L.window} ${W})`}
                    data={layerTrace.kSwa}
                    rows={T}
                    cols={HD}
                    highlightRow={focusIndex}
                    dimRowsBefore={windowStart || null}
                    separatorEvery={DC}
                    stages={['swa']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Strip
                    title="V_w"
                    data={layerTrace.vSwa}
                    rows={T}
                    cols={HD}
                    highlightRow={focusIndex}
                    dimRowsBefore={windowStart || null}
                    separatorEvery={DC}
                    stages={['swa']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                </div>
              </div>

              <div className="sim-subrow">
                <h4>
                  {L.panels.kv}
                  {entries && entries.layer !== layer && (
                    <span className="sim-inherit-tag">{L.inherited(entries.layer)}</span>
                  )}
                </h4>
                {entries ? (
                  <div className="sim-row">
                    <Strip
                      title={`[h_{${entries.m}j}${entries.m === 2 ? '; h_{2j+1}' : ''}]`}
                      data={entries.merged}
                      rows={n}
                      cols={entries.m * D}
                      cell={WIDE_CELL}
                      highlightRow={focusEntry}
                      separatorEvery={D}
                      stages={['compress']}
                      active={stage}
                      inherited={entries.layer !== layer}
                    />
                    <Op>·</Op>
                    <Heat
                      title="W_c"
                      data={model.layerWeights(entries.layer).wc!}
                      rows={entries.m * D}
                      cols={DC}
                      stages={['compress']}
                      active={stage}
                    />
                    <Op>→</Op>
                    <Strip
                      title="c (raw)"
                      data={entries.cRaw}
                      rows={n}
                      cols={DC}
                      cell={6}
                      highlightRow={focusEntry}
                      dimRowsBefore={mode === 'decode' ? n - 1 : null}
                      stages={['compress']}
                      active={stage}
                      inherited={entries.layer !== layer}
                    />
                    <Op>FP4 →</Op>
                    <Strip
                      title={`C (${bytes.mainEntry} B)`}
                      data={entries.c}
                      rows={n}
                      cols={DC}
                      cell={6}
                      highlightRow={focusEntry}
                      dimRowsBefore={mode === 'decode' ? n - 1 : null}
                      stages={['fp4', 'gather']}
                      active={stage}
                      inherited={entries.layer !== layer}
                    />
                    <Strip
                      title="scale"
                      data={entries.scales}
                      rows={n}
                      cols={DC / cfg.fp4Group}
                      cell={10}
                      highlightRow={focusEntry}
                      stages={['fp4']}
                      active={stage}
                      inherited={entries.layer !== layer}
                    />
                    <Op>·</Op>
                    <Heat
                      title="W_I"
                      data={model.layerWeights(entries.layer).wi!}
                      rows={DC}
                      cols={DI}
                      cell={4}
                      stages={['indexerK']}
                      active={stage}
                    />
                    <Op>MXFP4 →</Op>
                    <Strip
                      title={`K^I (${bytes.indexerEntry} B)`}
                      data={entries.ki}
                      rows={n}
                      cols={DI}
                      cell={8}
                      highlightRow={focusEntry}
                      dimRowsBefore={mode === 'decode' ? n - 1 : null}
                      stages={['indexerK']}
                      active={stage}
                      inherited={entries.layer !== layer}
                    />
                  </div>
                ) : (
                  <p className="sim-state-note">{L.modeNotes.swa}</p>
                )}
              </div>

              <div className="sim-subrow">
                <h4>
                  {L.panels.select}
                  {spec.mode === 'reuse' && layerTrace.selectedFrom !== null && (
                    <span className="sim-inherit-tag">{L.inherited(layerTrace.selectedFrom)}</span>
                  )}
                </h4>
                {entries ? (
                  <div className="sim-row">
                    {layerTrace.qi && layerTrace.wIdx && (
                      <>
                        <Strip
                          title="q^I"
                          data={layerTrace.qi}
                          rows={T}
                          cols={HI * DI}
                          cell={6}
                          highlightRow={focusIndex}
                          dimRowsBefore={layerTrace.start || null}
                          separatorEvery={DI}
                          stages={['indexerQ']}
                          active={stage}
                          onSelectRow={setFocus}
                        />
                        <Strip
                          title="w"
                          data={layerTrace.wIdx}
                          rows={T}
                          cols={HI}
                          cell={10}
                          highlightRow={focusIndex}
                          dimRowsBefore={layerTrace.start || null}
                          stages={['indexerQ']}
                          active={stage}
                          onSelectRow={setFocus}
                        />
                        <Op>·K^I →</Op>
                        <Strip
                          title="S (scores)"
                          data={layerTrace.scores!}
                          rows={T}
                          cols={n}
                          cell={Math.max(3, Math.min(ROW, Math.floor(200 / n)))}
                          highlightRow={focusIndex}
                          dimRowsBefore={layerTrace.start || null}
                          markZeros
                          stages={['scores']}
                          active={stage}
                          onSelectRow={setFocus}
                        />
                      </>
                    )}
                    {poolMask && (
                      <MaskHeat
                        title={`P (pool ≤ ${NB * B})`}
                        mask={poolMask}
                        rows={T}
                        cols={n}
                        highlightRow={focusIndex}
                        stages={['pool']}
                        active={stage}
                        inherited={spec.mode === 'reindex'}
                        tooltip={(r, c, v) => `pool[${r}, ${c}] ${entryTokens(c)} = ${v}`}
                      />
                    )}
                    {layerTrace.selected && (
                      <MaskHeat
                        title={`T_t (Top-${K})`}
                        mask={layerTrace.selected}
                        rows={T}
                        cols={n}
                        highlightRow={focusIndex}
                        stages={['topk', 'gather']}
                        active={stage}
                        inherited={spec.mode === 'reuse'}
                        tooltip={(r, c, v) => `selected[${r}, ${c}] ${entryTokens(c)} = ${v}`}
                      />
                    )}
                  </div>
                ) : (
                  <p className="sim-state-note">{L.stateNote.none}</p>
                )}
              </div>

              <div className="sim-subrow">
                <h4>{L.panels.attention}</h4>
                <div className="sim-row">
                  {entries && (
                    <figure
                      className={`sim-el sim-attn${stage === 'logits' || stage === 'merge' ? ' is-active' : ''}`}
                    >
                      <figcaption>
                        A_global · {L.headOption(head)}{' '}
                        <span className="sim-shape">
                          {T}×{n}
                        </span>
                      </figcaption>
                      <MatrixCanvas
                        rows={T}
                        cols={n}
                        value={(r, c) => layerTrace.attnGlobal[(head * T + r) * n + c]}
                        scale={1}
                        mode="sequential"
                        cellWidth={Math.max(3, Math.min(ROW, Math.floor(200 / n)))}
                        cellHeight={ROW}
                        gap={1}
                        highlightRow={focusIndex}
                        dimRowsBefore={cacheRows}
                        label={`global attention ${T}×${n}`}
                        tooltip={(r, c, v) =>
                          `A[${r}, ${c}] ${displayChar(trace.tokens[r])} → ${entryTokens(c)} = ${v.toFixed(3)}`
                        }
                        onSelectRow={setFocus}
                      />
                      <div className="sim-minis">
                        {Array.from({ length: H }, (_, h) => (
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
                              cols={n}
                              value={(r, c) => layerTrace.attnGlobal[(h * T + r) * n + c]}
                              scale={1}
                              mode="sequential"
                              cellWidth={Math.max(1, Math.floor(40 / n))}
                              cellHeight={Math.max(1, Math.floor(40 / T))}
                              label={L.headOption(h)}
                            />
                          </button>
                        ))}
                      </div>
                    </figure>
                  )}
                  <figure
                    className={`sim-el sim-attn${stage === 'logits' || stage === 'merge' ? ' is-active' : ''}`}
                  >
                    <figcaption>
                      A_local · {L.headOption(head)}{' '}
                      <span className="sim-shape">
                        {T}×{T}
                      </span>
                    </figcaption>
                    <MatrixCanvas
                      rows={T}
                      cols={T}
                      value={(r, c) => layerTrace.attnLocal[(head * T + r) * T + c]}
                      scale={1}
                      mode="sequential"
                      cellWidth={Math.max(3, Math.min(ROW, Math.floor(200 / T)))}
                      cellHeight={ROW}
                      gap={1}
                      highlightRow={focusIndex}
                      dimRowsBefore={cacheRows}
                      label={`local attention ${T}×${T}`}
                      tooltip={(r, c, v) =>
                        `A[${r}, ${c}] ${displayChar(trace.tokens[r])} → ${displayChar(trace.tokens[c])} = ${v.toFixed(3)}`
                      }
                      onSelectRow={setFocus}
                    />
                  </figure>
                  <Op>→</Op>
                  <Strip
                    title="o"
                    data={layerTrace.o}
                    rows={T}
                    cols={HD}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    separatorEvery={DC}
                    stages={['weighted']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>·</Op>
                  <Heat
                    title="W_O"
                    data={weights.wo}
                    rows={HD}
                    cols={D}
                    stages={['wo']}
                    active={stage}
                  />
                  <Op>→</Op>
                  <Strip
                    title="y"
                    data={layerTrace.y}
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
                    data={layerTrace.x1}
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
                <h4>{L.panels.moe}</h4>
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
                    title="W_r"
                    data={weights.wr}
                    rows={D}
                    cols={E}
                    cell={4}
                    stages={['router']}
                    active={stage}
                  />
                  <Op>σ →</Op>
                  <Strip
                    title="r · gate"
                    data={layerTrace.gates}
                    rows={T}
                    cols={E}
                    cell={10}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    markZeros
                    scale={1}
                    stages={['router']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>→</Op>
                  <Strip
                    title="shared"
                    data={layerTrace.shared}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['experts']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>+</Op>
                  <Strip
                    title="routed"
                    data={layerTrace.routed}
                    rows={T}
                    cols={D}
                    highlightRow={focusIndex}
                    dimRowsBefore={cacheRows}
                    stages={['experts']}
                    active={stage}
                    onSelectRow={setFocus}
                  />
                  <Op>+x' →</Op>
                  <Strip
                    title="x''"
                    data={layerTrace.x2}
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
                  stages={['outLogits']}
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

            <section className="sim-panel sim-cache" data-testid="sim-cache">
              <h3>{L.cacheCard.title}</h3>
              <ul>
                {bytes.sources.map((s) => {
                  const e = trace.layers[s.layer].entries!
                  return (
                    <li key={s.layer}>
                      {L.cacheCard.entries(s.layer, s.m, e.n, bytes.mainEntry + bytes.indexerEntry)}{' '}
                      = {e.n * (bytes.mainEntry + bytes.indexerEntry)} B
                    </li>
                  )
                })}
              </ul>
              <p className="sim-cache-total">
                {L.cacheCard.perToken(bytes.perToken)}
                <br />
                <code>
                  {L.cacheCard.formula(
                    bytes.mainEntry,
                    bytes.indexerEntry,
                    bytes.sources.map((s) => (s.m === 1 ? '1' : `½`)).join(' + '),
                    bytes.perToken,
                  )}
                </code>
                <br />
                <span className="sim-cache-real">{L.cacheCard.real}</span>
              </p>
              <p className="sim-cache-reads">
                <strong>{L.cacheCard.readsTitle}</strong>
                <br />
                {L.cacheCard.reads(reads.indexer, reads.main, reads.swa)}
                <br />
                <span className="sim-cache-real">{L.cacheCard.realReads}</span>
              </p>
            </section>
          </div>

          <section className="sim-focus" aria-live="polite">
            <h3>
              {L.focusTitle(focusIndex, displayChar(trace.tokens[focusIndex]))}
              <span className="sim-focus-stage">{L.stageNames[stage]}</span>
              <span className={`sim-focus-stage mode-${spec.mode}`}>
                L{layer} · {L.modeTitles[spec.mode]}
              </span>
            </h3>
            <p className="sim-detail">{L.detail[stage]}</p>
            {renderFocus()}
          </section>
        </>
      )}

      <footer className="sim-footer">
        <p>{L.modelNote(model.training.paramCount, cfg.layers.length, model.vocab.length)}</p>
        <details className="sim-simplifications">
          <summary>{L.simplificationsTitle}</summary>
          <ul>
            {L.simplifications.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </details>
        <p>{L.sources}</p>
      </footer>
    </div>
  )
}
