interface DataEmptyStateProps {
  heading: string
  title: string
  body: string
  errorsTitle: string
  errors: string[]
  testId: string
}

/** 실제 값 데이터 파일이 없거나 깨졌을 때 보여 주는 안내. */
export function DataEmptyState({
  heading,
  title,
  body,
  errorsTitle,
  errors,
  testId,
}: DataEmptyStateProps) {
  return (
    <div className="demo demo-empty" data-testid={testId}>
      <div className="demo-header">
        <h3>{heading}</h3>
      </div>
      <p className="empty-title">{title}</p>
      <p className="demo-note">{body}</p>
      {errors.length > 0 && (
        <div className="empty-errors">
          <p className="empty-title">{errorsTitle}</p>
          <ul>
            {errors.map((error) => (
              <li key={error}>
                <code>{error}</code>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
