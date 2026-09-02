# CLAUDE.md

트랜스포머(Transformer)를 **주니어 엔지니어와 임원**에게 쉽게 설명하는 인터랙티브 웹 페이지.
참고 프로젝트(poloclub/transformer-explainer)보다 더 쉽게, 한국어로, 비유 중심으로 설명한다.

## 명령

| 명령                | 설명                                                 |
| ------------------- | ---------------------------------------------------- |
| `npm run dev`       | 개발 서버 (Vite)                                     |
| `npm run check`     | lint + format:check + typecheck + 단위 테스트 + 빌드 |
| `npm run lint`      | oxlint                                               |
| `npm run format`    | prettier --write                                     |
| `npm run typecheck` | tsc -b                                               |
| `npm test`          | vitest run                                           |
| `npm run e2e`       | playwright (빌드 후 `vite preview` 에 대해 실행)     |
| `npm run build`     | dist/ 생성                                           |

완료 기준: `npm run check` 와 `npm run e2e` 가 모두 통과해야 한다. CI(.github/workflows/ci.yml)도 같은 순서로 돈다.

## 스택

Vite 8 + React 19 + TypeScript 6(strict) / oxlint + Prettier / Vitest + Testing Library / Playwright.
UI 프레임워크나 CSS 라이브러리는 쓰지 않는다. 스타일은 `src/styles/global.css` 의 디자인 토큰(라이트/다크)으로 관리한다.
의존성 추가는 꼭 필요할 때만, 이유를 커밋 메시지에 적는다.

## 구조

```
src/
  content/      본문 데이터. sections.ts(섹션 본문), sources.ts(참고 자료), types.ts
  components/   화면 조각. SectionView 가 섹션 하나를 그린다. 인터랙티브 데모는 *Demo.tsx
  state/        설명 수준(임원용/엔지니어용) 컨텍스트
  lib/          순수 계산 함수 (math.ts: softmax/어텐션, tokenize.ts). 반드시 테스트와 함께
  styles/       global.css
e2e/            Playwright 스모크 테스트
docs/plan.md    콘텐츠 계획, 설계 원칙, 열린 질문
```

## 콘텐츠 규칙 (가장 중요)

- 청중은 둘이다. `executive`(임원용)는 수식·기호 없이 비유와 결과만. `engineer`(엔지니어용)는 정확한 용어와 수식까지.
  섹션 데이터의 `executive` 문단은 두 모드 모두에 보이고, `engineer` 문단은 엔지니어용에서만 보인다.
- 모든 사실은 검증된 출처가 있어야 한다. 기본 출처는 Vaswani et al. 2017 ("Attention Is All You Need").
  구체적 숫자(차원, 층 수, 파라미터 수 등)는 출처가 확인된 것만 쓴다. 확인 못 한 주장은 넣지 않는다.
- 새 섹션은 `status: 'draft'` 로 시작하고, 내용 검수 후 `'ready'` 로 바꾼다. draft 는 화면에 "초안" 배지가 붙는다.
- 인터랙티브 데모가 실제 모델과 다르게 단순화된 것이면 화면에 그 사실을 밝힌다 (TokenizerDemo 의 안내문 참고).
- 언어는 한국어. 전문 용어는 처음 나올 때 괄호로 영문을 병기한다.

## 코드 규칙

- 접근성: 폼 요소에는 label, 목차/데모에는 aria-label, 상태 변화에는 aria-live. 색만으로 정보를 구분하지 않는다.
- `src/lib` 의 함수는 순수 함수로 두고 단위 테스트를 붙인다. 컴포넌트는 Testing Library 로 사용자 관점에서 테스트한다.
- 새 인터랙티브 데모를 추가할 때: `content/types.ts` 의 `InteractiveKind` 에 이름을 추가하고, `SectionView.tsx` 의 `INTERACTIVES` 에 컴포넌트를 등록한 뒤, 섹션 데이터의 `interactive` 에 지정한다.
- 커밋 메시지는 한국어 또는 영어 자유. 무엇을 왜 바꿨는지 첫 줄에 적는다.

## 하네스

- `.claude/hooks/session-start.sh`: Claude Code on the web 세션 시작 시 `npm install` (로컬에서는 건너뜀).
- `.github/workflows/ci.yml`: 푸시/PR 마다 check + e2e.
- `.github/workflows/deploy.yml`: main 푸시 시 GitHub Pages 배포. 저장소 Settings → Pages → Source 를 "GitHub Actions" 로 한 번 설정해야 한다. 빌드 시 `BASE_PATH=/transformer_explain/` 이 주입된다.
- Playwright 는 1.56.1 로 고정. 원격 세션의 사전 설치 Chromium(빌드 1194)과 맞추기 위해서다. 올릴 때는 `npx playwright install chromium` 이 필요하다.
