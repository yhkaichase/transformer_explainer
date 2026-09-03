// dist/ 의 빌드 결과를 HTML 파일 하나로 합친다 (CSS, JS, 파비콘을 인라인).
// 결과: dist/transformer-explain.html. Node.js 없이 브라우저에서 바로 열 수 있다 (더블클릭, 인트라넷, 메일 첨부).
// 사용: npm run build && node scripts/build-single-file.mjs
import { readFileSync, writeFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const dist = 'dist'
const output = join(dist, 'transformer-explain.html')

function assetPath(href) {
  // base 경로(/ 또는 /transformer_explain/)와 무관하게 /assets/ 이후만 쓴다.
  const marker = '/assets/'
  const index = href.indexOf(marker)
  if (index < 0) throw new Error(`알 수 없는 자산 경로: ${href}`)
  return join(dist, href.slice(index + 1))
}

let html = readFileSync(join(dist, 'index.html'), 'utf8')
let inlined = 0

html = html.replace(/<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_, href) => {
  inlined += 1
  return `<style>${readFileSync(assetPath(href), 'utf8')}</style>`
})

html = html.replace(/<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g, (_, src) => {
  inlined += 1
  // 인라인 스크립트 안의 </script 는 태그를 닫아 버리므로 이스케이프한다.
  const code = readFileSync(assetPath(src), 'utf8').replace(/<\/script/gi, '<\\/script')
  return `<script type="module">${code}</script>`
})

html = html.replace(/<link rel="icon"[^>]*href="([^"]+)"[^>]*>/g, (_, href) => {
  const file = href.startsWith('/') ? join(dist, href.replace(/^\/[^/]*\//, '')) : join(dist, href)
  try {
    const svg = readFileSync(file).toString('base64')
    return `<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,${svg}" />`
  } catch {
    return ''
  }
})

if (inlined < 2)
  throw new Error(
    'CSS 또는 JS 를 인라인하지 못했습니다. npm run build 를 먼저 실행했는지 확인하세요.',
  )
writeFileSync(output, html)
console.log(`생성: ${output} (${Math.round(statSync(output).size / 1024)} KB)`)
