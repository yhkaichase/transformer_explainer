// 페이지마다 HTML 파일 하나로 합친다 (CSS, JS, 파비콘을 인라인).
// 결과: dist/transformer-explain.html (설명 페이지), dist/transformer-simulator.html (트랜스포머 시뮬레이터),
//       dist/DCv4.1-simulator.html (DeepSeek-V4.1-Flash 시뮬레이터).
// Node.js 없이 브라우저에서 바로 열 수 있다 (더블클릭, 인트라넷, 메일 첨부).
//
// npm run build 는 두 페이지가 코드를 공유하는 청크를 만들므로, 여기서는 페이지마다 따로
// 자기 완결적인 번들을 만든 뒤 인라인한다. 사용: npm run build:single
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { build } from 'vite'

const root = resolve('.')
const dist = join(root, 'dist')
const cache = join(root, 'node_modules', '.cache', 'single-file')
const PAGES = [
  { source: 'index.html', output: 'transformer-explain.html' },
  { source: 'simulator.html', output: 'transformer-simulator.html' },
  { source: 'DCv4.1-simulator.html', output: 'DCv4.1-simulator.html' },
]

function assetPath(outDir, href) {
  // base 경로(/ 또는 /transformer_explainer/)와 무관하게 /assets/ 이후만 쓴다.
  const marker = '/assets/'
  const index = href.indexOf(marker)
  if (index < 0) throw new Error(`알 수 없는 자산 경로: ${href}`)
  return join(outDir, href.slice(index + 1))
}

async function buildPage(source) {
  const outDir = join(cache, source.replace(/\.html$/, ''))
  await build({
    configFile: false,
    root,
    base: '/',
    logLevel: 'warn',
    plugins: [react()],
    build: {
      outDir,
      emptyOutDir: true,
      rollupOptions: { input: join(root, source) },
      chunkSizeWarningLimit: 2400,
    },
  })
  return outDir
}

function inlinePage(outDir, source, output) {
  let html = readFileSync(join(outDir, source), 'utf8')
  let inlined = 0

  html = html.replace(/<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_, href) => {
    inlined += 1
    return `<style>${readFileSync(assetPath(outDir, href), 'utf8')}</style>`
  })

  html = html.replace(/<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g, (_, src) => {
    inlined += 1
    // 인라인 스크립트 안의 </script 는 태그를 닫아 버리므로 이스케이프한다.
    const code = readFileSync(assetPath(outDir, src), 'utf8').replace(/<\/script/gi, '<\\/script')
    return `<script type="module">${code}</script>`
  })

  if (/<link rel="modulepreload"/.test(html)) {
    throw new Error(
      `${source}: 공유 청크가 남아 있습니다. 페이지별 번들이 자기 완결적이지 않습니다.`,
    )
  }

  html = html.replace(/<link rel="icon"[^>]*href="([^"]+)"[^>]*>/g, (_, href) => {
    const file = join(outDir, href.replace(/^\//, ''))
    try {
      const svg = readFileSync(file).toString('base64')
      return `<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,${svg}" />`
    } catch {
      return ''
    }
  })

  if (inlined < 2) throw new Error(`${source}: CSS 또는 JS 를 인라인하지 못했습니다.`)
  mkdirSync(dist, { recursive: true })
  const target = join(dist, output)
  writeFileSync(target, html)
  console.log(`생성: dist/${output} (${Math.round(statSync(target).size / 1024)} KB)`)
}

for (const page of PAGES) {
  const outDir = await buildPage(page.source)
  inlinePage(outDir, page.source, page.output)
}
