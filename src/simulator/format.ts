/** 두 시뮬레이터 페이지가 함께 쓰는 표시 상수와 도우미 (컴포넌트가 아닌 것만). */

export const ROW = 14
export const CELL = 3
export const WIDE_CELL = 1.5
export const HEAT_CELL = 2

export function maxAbs(values: ArrayLike<number>): number {
  let max = 0
  for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i]))
  return max || 1
}

export function displayChar(char: string): string {
  if (char === ' ') return '␣'
  if (char === '\n') return '⏎'
  return char
}

export const fmt = (value: number) => value.toFixed(2)
