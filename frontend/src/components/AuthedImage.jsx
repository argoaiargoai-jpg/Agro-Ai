import { useEffect, useState } from 'react'
import { ImageOff } from 'lucide-react'
import { api } from '../lib/api'
import { Skeleton } from './ui'

/** Images are private, so fetch with the bearer token and render via an object URL. */
export default function AuthedImage({ analysisId, alt, style, className }) {
  const [src, setSrc] = useState(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let url, dead = false
    setSrc(null); setFailed(false)
    api.blob(`/analyses/${analysisId}/image`)
      .then((b) => { if (!dead) { url = URL.createObjectURL(b); setSrc(url) } })
      .catch(() => !dead && setFailed(true))
    return () => { dead = true; if (url) URL.revokeObjectURL(url) }
  }, [analysisId])
  if (failed) return <div className={className} style={{ ...style, display: 'grid', placeItems: 'center', background: '#eef3ec', color: '#78877c' }}><ImageOff size={22} /></div>
  if (!src) return <Skeleton h="100%" w="100%" style={style} />
  return <img src={src} alt={alt} className={className} style={{ objectFit: 'cover', ...style }} />
}
