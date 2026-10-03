import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Camera, CameraOff, ImagePlus, RefreshCw, UploadCloud, X } from 'lucide-react'
import { PageHead } from '../components/AnalysisBits'
import { ScanPreview, StageList } from '../components/Scan'
import { Alert, Button, Field } from '../components/ui'
import { useConfig } from '../context/ConfigContext'
import { useToast } from '../context/ToastContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { fmtBytes } from '../lib/format'

const TYPES = ['image/jpeg', 'image/png', 'image/webp']

function CameraCapture({ onCapture }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [state, setState] = useState('idle') // idle | starting | live | error
  const [err, setErr] = useState(null)
  const [facing, setFacing] = useState('environment')

  const stop = useCallback(() => { streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null }, [])
  const start = useCallback(async (mode) => {
    stop()
    if (!navigator.mediaDevices?.getUserMedia) {
      setState('error'); setErr('Camera access is not supported in this browser (or the page is not served over HTTPS).'); return
    }
    setState('starting'); setErr(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: mode, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play().catch(() => {}) }
      setState('live')
    } catch (e) {
      setState('error')
      setErr(e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser settings, or upload a photo instead.'
        : e.name === 'NotFoundError' ? 'No camera was found on this device. Upload a photo instead.'
        : 'We could not start the camera. Another app may be using it.')
    }
  }, [stop])

  useEffect(() => stop, [stop])

  function snap() {
    const v = videoRef.current
    if (!v?.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    c.toBlob((blob) => {
      if (!blob) return
      onCapture(new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' }))
      stop(); setState('idle')
    }, 'image/jpeg', 0.92)
  }

  if (state === 'idle') return (
    <div className="dropzone" style={{ cursor: 'default' }}>
      <div className="state-icon"><Camera size={26} /></div>
      <strong>Use your camera</strong>
      <p className="small muted">Hold the leaf in the frame with good light, then capture.</p>
      <Button onClick={() => start(facing)} icon={Camera}>Start camera</Button>
    </div>
  )
  if (state === 'error') return (
    <div className="dropzone" style={{ cursor: 'default' }}>
      <div className="state-icon"><CameraOff size={26} /></div>
      <Alert tone="warn" style={{ textAlign: 'left' }}>{err}</Alert>
      <Button variant="secondary" onClick={() => start(facing)}>Try again</Button>
    </div>
  )
  return (
    <div>
      <div className="preview">
        <video ref={videoRef} playsInline muted aria-label="Camera preview" />
        <div className="cam-frame" />
        {state === 'starting' && <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#fff' }}><span className="spinner lg" /></div>}
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 20, marginTop: 16 }}>
        <Button variant="secondary" size="sm" onClick={() => { stop(); setState('idle') }}>Cancel</Button>
        <button className="shutter" onClick={snap} aria-label="Capture photo" disabled={state !== 'live'}><i /></button>
        <Button variant="secondary" size="sm" icon={RefreshCw} onClick={() => { const m = facing === 'environment' ? 'user' : 'environment'; setFacing(m); start(m) }}>Flip</Button>
      </div>
    </div>
  )
}

export default function Analyze() {
  const nav = useNavigate()
  const toast = useToast()
  const { config } = useConfig()
  const [tab, setTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [source, setSource] = useState('upload')
  const [preview, setPreview] = useState(null)
  const [crop, setCrop] = useState('')
  const [notes, setNotes] = useState('')
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(null) // null | 'uploading' | 'ml' | 'guidance'  (real backend stages, no fake progress)
  const input = useRef(null)
  const maxBytes = config.max_upload_mb * 1024 * 1024

  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function pick(f, src = 'upload') {
    setError(null)
    if (!f) return
    if (!TYPES.includes(f.type)) return setError('Unsupported file type. Please choose a JPEG, PNG or WebP image.')
    if (f.size > maxBytes) return setError(`That image is ${fmtBytes(f.size)}. The limit is ${config.max_upload_mb} MB.`)
    if (f.size === 0) return setError('That file is empty.')
    setFile(f); setSource(src)
  }

  async function submit() {
    setBusy(true); setError(null)
    const form = new FormData()
    form.append('file', file)
    form.append('source', source)
    if (crop) form.append('crop_type', crop)
    if (notes.trim()) form.append('notes', notes.trim())
    let poll
    try {
      setStage('uploading')
      const a = await api.upload('/analyses', form)
      setStage('ml')
      // The backend commits the first stage before it starts the second, so we can show what is REALLY happening.
      poll = setInterval(async () => {
        try { const cur = await api.get(`/analyses/${a.id}`); if (cur.result?.stage === 'ml_done') setStage('guidance') } catch { /* keep waiting */ }
      }, 1000)
      let ok = true
      try { await api.post(`/analyses/${a.id}/analyze`) }
      catch (e) { ok = false; toast(errorMessage(e), 'error') } // the result page explains the state and offers a retry
      clearInterval(poll)
      if (ok) { setStage('done'); await new Promise((r) => setTimeout(r, 650)) }   // show the finished state briefly: it is real, the backend has answered
      nav(`/analysis/${a.id}`)
    } catch (e) {
      setError(errorMessage(e))
      if (e instanceof ApiError && e.code === 'maintenance') toast(e.message, 'error')
      setStage(null)
    } finally { clearInterval(poll); setBusy(false) }
  }

  return (
    <>
      <PageHead title="Analyze a crop" subtitle="Upload a plant image and let AGRO AI analyze it. A clear, well-lit photo of a single leaf works best." />
      <div className="grid grid-main">
        <div className="card card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {config.maintenance_mode && <Alert tone="warn">AGRO AI is in maintenance mode, so new uploads are paused.</Alert>}
          {error && <Alert tone="error">{error}</Alert>}
          {file ? (
            busy ? (
              <div className="analyze-hero">
                <ScanPreview src={preview} active={stage !== 'done'} alt="Your image being analyzed" />
                <StageList stage={stage || 'uploading'} />
              </div>
            ) : (
              <div className="preview">
                <img src={preview} alt="Selected crop" />
                <Button variant="secondary" size="sm" className="remove" icon={X} onClick={() => setFile(null)} aria-label="Remove image">Remove</Button>
              </div>
            )
          ) : (
            <>
              <div className="tabs" role="tablist" style={{ alignSelf: 'flex-start' }}>
                <button role="tab" className="tab" aria-selected={tab === 'upload'} onClick={() => setTab('upload')}><UploadCloud size={16} /> Upload</button>
                <button role="tab" className="tab" aria-selected={tab === 'camera'} onClick={() => setTab('camera')}><Camera size={16} /> Camera</button>
              </div>
              {tab === 'upload' ? (
                <div
                  className={`dropzone${drag ? ' drag' : ''}`} role="button" tabIndex={0}
                  onClick={() => input.current?.click()}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), input.current?.click())}
                  onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]) }}
                >
                  <div className="state-icon"><ImagePlus size={26} /></div>
                  <strong>Drag a photo here, or click to browse</strong>
                  <p className="small muted">JPEG, PNG or WebP · up to {config.max_upload_mb} MB</p>
                  <input ref={input} type="file" accept={TYPES.join(',')} hidden data-testid="file-input"
                    onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }} />
                </div>
              ) : <CameraCapture onCapture={(f) => pick(f, 'camera')} />}
            </>
          )}
        </div>

        <div className="card card-pad">
          <form className="form" onSubmit={(e) => { e.preventDefault(); submit() }}>
            <Field as="select" label="Crop (optional)" value={crop} onChange={(e) => setCrop(e.target.value)}>
              <option value="">Not sure / other</option>
              {config.supported_crops.map((c) => <option key={c}>{c}</option>)}
            </Field>
            <Field as="textarea" label="Notes (optional)" maxLength={1000} placeholder="e.g. Yellow spots on lower leaves, started last week" value={notes} onChange={(e) => setNotes(e.target.value)} />
            <Button type="submit" size="lg" block loading={busy && stage !== 'done'} disabled={!file || busy || config.maintenance_mode} icon={UploadCloud}>
              {stage === 'uploading' ? 'Uploading…' : stage === 'done' ? 'Done' : stage ? 'Analyzing…' : 'Analyze image'}
            </Button>
            <p className="field-hint">Your photo is stored privately and only visible to you.</p>
          </form>
        </div>
      </div>
    </>
  )
}
