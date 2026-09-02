# 트랜스포머 쉽게 이해하기

ChatGPT 같은 AI 의 핵심 구조인 **트랜스포머(Transformer)** 를 주니어 엔지니어와 임원 눈높이에서 설명하는 인터랙티브 웹 페이지입니다.

- **임원용** 모드: 수식 없이 비유와 결과 중심으로 읽습니다.
- **엔지니어용** 모드: 같은 페이지에서 수식과 구조 설명이 추가됩니다.

참고 프로젝트 [Transformer Explainer](https://github.com/poloclub/transformer-explainer)(Georgia Tech Polo Club)가 브라우저에서 실제 GPT-2 를 실행하며 내부를 보여 준다면, 이 프로젝트는 그보다 한 단계 쉬운 설명을 목표로 합니다. 자세한 계획은 [docs/plan.md](docs/plan.md) 에 있습니다.

## 시작하기

Node.js 22.12 이상이 필요합니다 (`.nvmrc` 참고).

```bash
npm install
npm run dev        # http://localhost:5173
```

## 스크립트

| 명령              | 설명                                                    |
| ----------------- | ------------------------------------------------------- |
| `npm run dev`     | 개발 서버                                               |
| `npm run build`   | 타입 검사 후 `dist/` 생성                               |
| `npm run preview` | 빌드 결과 미리보기                                      |
| `npm run check`   | lint, 포맷 검사, 타입 검사, 단위 테스트, 빌드를 한 번에 |
| `npm run lint`    | oxlint                                                  |
| `npm run format`  | Prettier 로 전체 포맷                                   |
| `npm test`        | Vitest 단위 테스트                                      |
| `npm run e2e`     | Playwright 브라우저 테스트 (빌드 결과에 대해 실행)      |

처음 `npm run e2e` 를 실행하기 전에 브라우저를 한 번 설치합니다.

```bash
npx playwright install chromium
```

## 구조

```
src/content/     본문 데이터 (섹션, 참고 자료)
src/components/  화면 조각과 인터랙티브 데모
src/lib/         softmax, 어텐션 등 순수 계산 함수 (테스트 포함)
src/state/       설명 수준(임원용/엔지니어용) 상태
e2e/             Playwright 스모크 테스트
docs/plan.md     콘텐츠 계획과 열린 질문
```

## 배포

`main` 브랜치에 푸시하면 `.github/workflows/deploy.yml` 이 GitHub Pages 로 배포합니다.
저장소 **Settings → Pages → Build and deployment → Source** 를 **GitHub Actions** 로 한 번 설정해야 합니다.
배포 주소는 `https://<사용자>.github.io/transformer_explain/` 형태입니다.

## 참고 자료

- Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
- Transformer Explainer — https://github.com/poloclub/transformer-explainer
