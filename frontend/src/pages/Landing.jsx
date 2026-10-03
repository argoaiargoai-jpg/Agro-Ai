import { Link } from 'react-router-dom'
import { ArrowRight, BarChart3, Camera, CheckCircle2, CloudUpload, History, Leaf as LeafIcon, ShieldCheck, Smartphone, Sprout, Sparkles, Zap } from 'lucide-react'
import { Logo } from '../components/ui'
import { Leaf, Reveal } from '../components/Reveal'
import { useAuth } from '../context/AuthContext'

const FEATURES = [
  { icon: Camera, title: 'Capture in the field', text: 'Use your phone camera or upload existing photos. Built for sunlight, mud and one-handed use.' },
  { icon: LeafIcon, title: 'Leaf-level insight', text: 'Submit a crop image and get a structured health report you can act on the same day.' },
  { icon: History, title: 'Every scan on record', text: 'A searchable history of each field check so you can compare what changed and when.' },
  { icon: BarChart3, title: 'Trends over time', text: 'See how your crops are doing across the season, not just a single snapshot.' },
  { icon: Smartphone, title: 'Works on any device', text: 'A responsive web app that feels at home on a phone, tablet or office desktop.' },
  { icon: ShieldCheck, title: 'Private by default', text: 'Verified accounts, encrypted sessions, and images only you can see.' },
]
const STEPS = [
  { icon: Camera, title: 'Capture', text: 'Take a clear photo of the affected leaf or plant, or upload one from your gallery.' },
  { icon: CloudUpload, title: 'Analyze', text: 'AGRO AI processes the image and prepares a plain-language health report.' },
  { icon: CheckCircle2, title: 'Act', text: 'Review findings, keep them in your history, and share with your agronomist.' },
]
const CROPS = ['Tomato', 'Potato', 'Maize', 'Rice', 'Wheat', 'Cotton', 'Soybean', 'Chilli', 'Sugarcane', 'Banana']
const WHY = ['Spot problems days earlier than a walk-through', 'One place for every photo, note and field check', 'Share clear records with agronomists and advisors', 'Simple enough for the whole farm team']

export default function Landing() {
  const { status } = useAuth()
  const authed = status === 'authed'
  return (
    <>
      <header className="nav">
        <div className="container nav-in">
          <Logo />
          <nav className="nav-links" aria-label="Sections">
            <a href="#features">Features</a><a href="#how">How it works</a><a href="#crops">Crops</a>
          </nav>
          <div className="nav-actions">
            {authed ? <Link to="/dashboard" className="btn btn-primary btn-sm">Open dashboard</Link> : (<>
              <Link to="/login" className="btn btn-ghost btn-sm">Sign in</Link>
              <Link to="/register" className="btn btn-primary btn-sm">Get started</Link>
            </>)}
          </div>
        </div>
      </header>

      <main>
        <section className="hero">
          <Leaf size={64} className="hero-leaf" style={{ left: '1%', top: '58%', animation: 'sway 7s ease-in-out infinite' }} color="#86efac" />
          <Leaf size={38} className="hero-leaf" style={{ left: '46%', top: '8%', animation: 'sway 9s -2s ease-in-out infinite' }} color="#bef264" />
          <Leaf size={50} className="hero-leaf" style={{ right: '3%', bottom: '8%', animation: 'sway 8s -4s ease-in-out infinite' }} color="#4ade80" />
          <div className="container hero-grid">
            <div>
              <span className="eyebrow"><Sprout size={15} /> Crop health intelligence</span>
              <h1>Spot crop trouble <em>before</em> it spreads.</h1>
              <p className="lead">AGRO AI helps growers check plant health from a single photo, keep a record of every scan, and make confident decisions in the field.</p>
              <div className="hero-cta">
                <Link to={authed ? '/analyze' : '/register'} className="btn btn-primary btn-lg">{authed ? 'Analyze a crop' : 'Start for free'} <ArrowRight size={18} /></Link>
                <a href="#how" className="btn btn-secondary btn-lg">See how it works</a>
              </div>
              <div className="hero-note">
                <span><CheckCircle2 size={16} /> Free to start</span>
                <span><CheckCircle2 size={16} /> Works on your phone</span>
                <span><CheckCircle2 size={16} /> Your data stays private</span>
              </div>
            </div>
            <div className="mock" aria-hidden="true">
              <div className="chip c1"><i><Camera size={15} /></i> Photo captured</div>
              <div className="chip c2"><i><Sparkles size={15} /></i> Report ready</div>
              <div className="mock-img"><div className="mock-scan" />
                <svg viewBox="0 0 200 150" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                  <path d="M100 25c-38 10-55 45-45 85 40 0 75-30 80-75-12-8-24-10-35-10z" fill="rgba(255,255,255,.22)" />
                  <path d="M62 105c22-22 40-44 66-62" stroke="rgba(255,255,255,.7)" strokeWidth="3" fill="none" strokeLinecap="round" />
                </svg>
              </div>
              <div className="mock-row">
                <div style={{ flex: 1 }}><strong style={{ fontFamily: 'var(--font-display)' }}>Scanning leaf…</strong><div className="mock-bar"><i style={{ width: '72%' }} /></div></div>
                <span className="badge ok"><span className="pulse-dot" /> Live</span>
              </div>
            </div>
          </div>
        </section>

        <div className="marquee" aria-label="Crops">
          <div className="marquee-track">
            {[...CROPS, ...CROPS].map((c, i) => <span key={i} className="crop-chip"><LeafIcon size={16} /> {c}</span>)}
          </div>
        </div>

        <section className="section" id="features">
          <div className="container">
            <Reveal className="section-head"><h2>Everything you need to monitor your crops</h2><p>A focused toolkit for growers, agronomists and farm teams.</p></Reveal>
            <div className="grid grid-3">
              {FEATURES.map(({ icon: I, title, text }, i) => (
                <Reveal key={title} as="article" delay={(i % 3) * 90} className="card feature"><div className="stat-icon"><I size={22} /></div><h3>{title}</h3><p>{text}</p></Reveal>
              ))}
            </div>
          </div>
        </section>

        <section className="section" id="how" style={{ background: 'linear-gradient(180deg, #fff, var(--mint-50))', borderBlock: '1px solid var(--line)' }}>
          <div className="container">
            <Reveal className="section-head"><h2>From photo to decision in three steps</h2><p>No lab, no waiting, no special equipment.</p></Reveal>
            <div className="grid grid-3 steps">
              {STEPS.map(({ icon: I, title, text }, i) => (
                <Reveal key={title} as="article" delay={i * 120} className="card step"><div className="stat-icon"><I size={22} /></div><h3>{title}</h3><p>{text}</p></Reveal>
              ))}
            </div>
          </div>
        </section>

        <section className="section">
          <div className="container split">
            <Reveal variant="left">
              <span className="eyebrow"><Zap size={15} /> Why growers choose AGRO AI</span>
              <h2 style={{ fontSize: 'clamp(28px,3.6vw,40px)' }}>Less guessing. More growing.</h2>
              <p className="muted" style={{ margin: '14px 0 24px', fontSize: 17 }}>Replace scattered phone photos and memory with a clean, searchable record of your fields.</p>
              <Link to={authed ? '/analyze' : '/register'} className="btn btn-primary">Try it now <ArrowRight size={17} /></Link>
            </Reveal>
            <Reveal variant="zoom" delay={120} className="split-panel">
              <h3 style={{ fontSize: 22, position: 'relative', zIndex: 1 }}>Built for the field</h3>
              <ul className="check-list">{WHY.map((w) => <li key={w}><CheckCircle2 size={20} />{w}</li>)}</ul>
            </Reveal>
          </div>
        </section>

        <section className="section" id="crops" style={{ paddingTop: 20 }}>
          <div className="container">
            <Reveal className="section-head"><h2>Built around the crops you grow</h2><p>Start with the staples and expand as the platform grows.</p></Reveal>
            <Reveal variant="zoom" className="crops">{CROPS.slice(0, 6).map((c) => <span key={c} className="crop-chip"><LeafIcon size={16} /> {c}</span>)}</Reveal>
          </div>
        </section>

        <Reveal as="section" variant="zoom" className="container" style={{ paddingBottom: 24 }}>
          <div className="band">
            <h2>Ready to see your fields more clearly?</h2>
            <p>Create a free account and run your first scan today.</p>
            <div className="hero-cta"><Link to={authed ? '/analyze' : '/register'} className="btn btn-leaf btn-lg">{authed ? 'Analyze a crop' : 'Create free account'} <ArrowRight size={18} /></Link></div>
          </div>
        </Reveal>
      </main>

      <footer className="footer">
        <div className="container footer-in"><Logo light /><span>© {new Date().getFullYear()} AGRO AI. All rights reserved.</span></div>
      </footer>
    </>
  )
}
