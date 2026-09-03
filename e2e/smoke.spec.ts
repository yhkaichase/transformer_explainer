import { expect, test } from '@playwright/test'

test('첫 화면이 렌더링되고 설명 수준을 전환할 수 있다', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle(/트랜스포머/)
  await expect(page.getByRole('heading', { level: 1 })).toContainText('트랜스포머')

  const executive = page.getByRole('radio', { name: /임원용/ })
  const engineer = page.getByRole('radio', { name: /엔지니어용/ })
  await expect(executive).toBeChecked()
  await expect(page.getByText('엔지니어를 위한 더 깊은 설명')).toHaveCount(0)

  await engineer.check()
  await expect(engineer).toBeChecked()
  await expect(page.getByText('엔지니어를 위한 더 깊은 설명').first()).toBeVisible()

  // 새로고침해도 선택이 유지된다.
  await page.reload()
  await expect(page.getByRole('radio', { name: /엔지니어용/ })).toBeChecked()
})

test('목차 링크로 섹션으로 이동하고 토크나이저 데모가 동작한다', async ({ page }) => {
  await page.goto('/')
  await page
    .getByRole('navigation', { name: '목차' })
    .getByRole('link', { name: /어텐션/ })
    .first()
    .click()
  await expect(page).toHaveURL(/#attention$/)
  await expect(page.getByRole('heading', { level: 2, name: /어텐션/ }).first()).toBeInViewport()

  const demo = page.getByTestId('tokenizer-demo')
  await demo.getByRole('textbox').fill('Hello, world')
  await expect(demo.getByText('토큰 3개')).toBeVisible()
})

test('영어 버전: ?lang=en 링크로 열리고 토글로 한국어로 돌아온다', async ({ page }) => {
  await page.goto('/?lang=en')
  await expect(page).toHaveTitle(/Transformers/)
  await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Transformers')
  await expect(page.getByRole('radio', { name: /Executive/ })).toBeChecked()
  await expect(page.getByRole('navigation', { name: 'Contents' })).toBeVisible()
  await expect(page.getByTestId('tokenizer-demo').getByText('9 tokens')).toBeVisible()

  await page.getByRole('radio', { name: 'English' }).check()
  await page.getByRole('radio', { name: '한국어' }).check()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('트랜스포머')
  await expect(page).toHaveURL(/lang=ko/)
  await expect(page.locator('html')).toHaveAttribute('lang', 'ko')

  // 새로고침해도 언어가 유지된다 (URL 파라미터).
  await page.reload()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('트랜스포머')
})

test('화면 스크린샷을 남긴다', async ({ page }) => {
  await page.goto('/')
  await page.screenshot({ path: 'test-results/home-executive.png', fullPage: true })
  await page.getByRole('radio', { name: /엔지니어용/ }).check()
  await page.screenshot({ path: 'test-results/home-engineer.png', fullPage: true })
  await page.getByRole('radio', { name: 'English' }).check()
  await page.screenshot({ path: 'test-results/home-english.png', fullPage: true })
})

test('온도 슬라이더, 흐름 단계, 위치 인코딩 데모가 동작한다', async ({ page }) => {
  await page.goto('/')

  const temperature = page.getByTestId('temperature-demo')
  await temperature.getByRole('slider').fill('0.2')
  await expect(temperature.getByRole('row').nth(1)).toContainText(/9\d\.\d%/)
  await temperature.getByRole('button', { name: '10번 뽑기' }).click()
  await expect(temperature.getByTestId('draw-results').locator('.draw-chip')).toHaveCount(10)

  const flow = page.getByTestId('flow-diagram')
  await flow.getByRole('button', { name: '다음 단계' }).click()
  await expect(flow.getByText('2단계 / 6단계')).toBeVisible()
  await expect(flow.getByRole('link', { name: '이 단계 자세히 보기' })).toHaveAttribute(
    'href',
    '#position',
  )

  const positional = page.getByTestId('positional-demo')
  await positional.getByRole('button', { name: '위치 5' }).click()
  await expect(positional.getByRole('heading', { level: 4 })).toHaveText('위치 5의 패턴')
})

test('어텐션, 피드포워드, 층 쌓기, 학습 데모가 동작한다', async ({ page }) => {
  await page.goto('/')

  const attention = page.getByTestId('attention-demo')
  await attention.getByRole('checkbox').check()
  await expect(attention.getByText('가려짐')).toHaveCount(3)
  await attention.getByRole('radio', { name: '돈' }).check()
  await expect(attention.getByText('가려짐')).toHaveCount(1)

  const ffn = page.getByTestId('ffn-demo')
  await ffn.getByRole('radio', { name: '돈' }).check()
  await expect(ffn.getByText('[1.00, 0.10]')).toBeVisible()

  const stack = page.getByTestId('stack-demo')
  await stack.getByRole('slider').fill('3')
  await expect(stack.getByRole('list', { name: '층 수' }).getByRole('listitem')).toHaveCount(3)

  const training = page.getByTestId('training-demo')
  await training.getByRole('button', { name: '10번 학습' }).click()
  await expect(training.getByText('학습 횟수').locator('..')).toContainText('10')
})

test('정리 섹션의 요약 카드는 엔지니어용에서 다음 단계 링크를 보여 준다', async ({ page }) => {
  await page.goto('/')
  const card = page.getByTestId('summary-card')
  await expect(card.getByRole('heading', { name: '1분 요약' })).toBeVisible()
  await expect(card.getByRole('link', { name: '원 논문 읽기' })).toHaveCount(0)
  await page.getByRole('radio', { name: /엔지니어용/ }).check()
  await expect(card.getByRole('link', { name: '원 논문 읽기' })).toHaveAttribute(
    'href',
    'https://arxiv.org/abs/1706.03762',
  )
  await expect(page.getByText('초안')).toHaveCount(0)
})

test('실제 값 데이터가 없으면 어텐션 히트맵과 헤드 비교 자리에 안내가 나온다', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('attention-heatmap-empty')).toBeVisible()
  await expect(page.getByTestId('multihead-empty')).toBeVisible()
  await expect(page.getByText('실제 값 데이터가 아직 없습니다')).toHaveCount(2)
})
