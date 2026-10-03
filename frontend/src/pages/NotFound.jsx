import { Link } from 'react-router-dom'
import { Compass, Lock } from 'lucide-react'
import { EmptyState } from '../components/ui'

export function NotFound() {
  return <div className="container" style={{ paddingTop: 80 }}><div className="card"><EmptyState icon={Compass} title="Page not found" action={<Link to="/" className="btn btn-primary">Go home</Link>}>The page you're looking for doesn't exist or has moved.</EmptyState></div></div>
}

export function Forbidden() {
  return <div className="card"><EmptyState icon={Lock} title="You don't have access" action={<Link to="/dashboard" className="btn btn-primary">Back to dashboard</Link>}>This area is for administrators only.</EmptyState></div>
}
