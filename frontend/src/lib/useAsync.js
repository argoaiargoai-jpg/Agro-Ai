import { useCallback, useEffect, useRef, useState } from 'react'

/** Minimal data-loading hook: { data, loading, error, reload }. Ignores stale responses. */
export function useAsync(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null })
  const seq = useRef(0)
  const run = useCallback(() => {
    const id = ++seq.current
    setState((s) => ({ ...s, loading: true, error: null }))
    fn().then(
      (data) => id === seq.current && setState({ data, loading: false, error: null }),
      (error) => id === seq.current && setState({ data: null, loading: false, error }),
    )
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { run() }, [run])
  return { ...state, reload: run }
}
