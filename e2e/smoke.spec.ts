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

test('화면 스크린샷을 남긴다', async ({ page }) => {
  await page.goto('/')
  await page.screenshot({ path: 'test-results/home-executive.png', fullPage: true })
  await page.getByRole('radio', { name: /엔지니어용/ }).check()
  await page.screenshot({ path: 'test-results/home-engineer.png', fullPage: true })
})
