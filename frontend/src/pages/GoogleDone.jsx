import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageLoader } from '../components/ui'
import { useAuth } from '../context/AuthContext'

export default function GoogleDone() {
  const { completeGoogle } = useAuth()
  const nav = useNavigate()
  const ran = useRef(false)
  useEffect(() => {
    if (ran.current) return
    ran.current = true
    completeGoogle().then(() => nav('/dashboard', { replace: true })).catch(() => nav('/login?oauth_error=oauth_cancelled', { replace: true }))
  }, [completeGoogle, nav])
  return <PageLoader />
}
