import { useEffect, useRef, useState, type RefObject } from 'react'

interface HScrollProps {
  /** 가로로 스크롤되는 구조도 요소 */
  target: RefObject<HTMLElement | null>
  label: string
}

/**
 * 넓은 구조도 위에 두는 가로 스크롤바. 구조도의 가로 스크롤과 양방향으로 맞춘다.
 * 구조도가 세로로 길어 아래쪽 스크롤바가 화면 밖으로 나가도, 위에 붙은 이 막대로 좌우를 움직일 수 있다.
 */
export function HScroll({ target, label }: HScrollProps) {
  const barRef = useRef<HTMLDivElement>(null)
  // 이 컴포넌트가 스크롤 위치를 바꾸는 요소. 렌더마다 target 에서 옮겨 담는다 (prop 을 직접 고치지 않기 위해).
  const elRef = useRef<HTMLElement | null>(null)
  const [size, setSize] = useState({ scroll: 0, client: 0 })

  // 의존성 없이 렌더마다 다시 재서 내용 폭이 바뀌어도(토큰 수, 층 변경) 따라간다. 값이 같으면 상태를 바꾸지 않는다.
  useEffect(() => {
    const el = target.current
    elRef.current = el
    if (!el) return
    const measure = () =>
      setSize((prev) =>
        prev.scroll === el.scrollWidth && prev.client === el.clientWidth
          ? prev
          : { scroll: el.scrollWidth, client: el.clientWidth },
      )
    const follow = () => {
      const bar = barRef.current
      if (bar && bar.scrollLeft !== el.scrollLeft) bar.scrollLeft = el.scrollLeft
    }
    measure()
    follow()
    el.addEventListener('scroll', follow)
    window.addEventListener('resize', measure)
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure)
    observer?.observe(el)
    return () => {
      el.removeEventListener('scroll', follow)
      window.removeEventListener('resize', measure)
      observer?.disconnect()
    }
  })

  if (size.scroll <= size.client + 1) return null
  return (
    <div
      className="sim-hscroll"
      ref={barRef}
      title={label}
      aria-hidden="true"
      tabIndex={-1}
      onScroll={(event) => {
        const el = elRef.current
        const left = event.currentTarget.scrollLeft
        if (el && el.scrollLeft !== left) el.scrollLeft = left
      }}
    >
      <div style={{ width: size.scroll }} />
    </div>
  )
}
