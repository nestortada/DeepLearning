import { useEffect, useRef, useState } from 'react'
import './App.css'
import './NetworkJourney.css'
import NetworkJourney from './NetworkJourney.jsx'

const maxImageBytes = 10 * 1024 * 1024
const scoreFormat = new Intl.NumberFormat('es-CO', { minimumFractionDigits: 4, maximumFractionDigits: 4 })
const percentFormat = new Intl.NumberFormat('es-CO', { style: 'percent', maximumFractionDigits: 1 })
const modelNames = { resnet34: 'ResNet34', resnet18: 'ResNet18', efficientnet_b0: 'EfficientNet-B0' }
const connectionError = 'No se pudo conectar con el servidor de análisis. Comprueba que esté iniciado.'
const apiUrl = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '')

const steps = [
  { n: '01', title: 'Sube la retinografía', text: 'Imagen de fondo de ojo en JPG o PNG, centrada en el disco óptico.' },
  { n: '02', title: 'La red la analiza', text: 'Un modelo ResNet con transfer learning, entrenado con el dataset PAPILA, estima el puntaje.' },
  { n: '03', title: 'Interpreta el mapa', text: 'Grad-CAM muestra en qué regiones de la imagen se apoyó el modelo.' },
]

function IrisMark({ size = 40 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true">
      <defs>
        <radialGradient id="irisFill" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#050b18" />
          <stop offset="38%" stopColor="#050b18" />
          <stop offset="40%" stopColor="#38e1d4" />
          <stop offset="100%" stopColor="#4f8cff" />
        </radialGradient>
      </defs>
      <circle cx="24" cy="24" r="22" fill="url(#irisFill)" />
      <circle cx="24" cy="24" r="22" fill="none" stroke="rgba(255,255,255,.35)" strokeWidth="1" />
      <circle cx="18.5" cy="18.5" r="3" fill="rgba(255,255,255,.85)" />
    </svg>
  )
}

function ScoreGauge({ score, threshold, glaucoma }) {
  const radius = 70
  const circumference = 2 * Math.PI * radius
  const clamped = Math.min(Math.max(score, 0), 1)
  const angle = threshold * 360 - 90
  const tx = 90 + radius * Math.cos((angle * Math.PI) / 180)
  const ty = 90 + radius * Math.sin((angle * Math.PI) / 180)
  return (
    <div className={`gauge ${glaucoma ? 'is-alert' : 'is-healthy'}`}>
      <svg viewBox="0 0 180 180" role="img" aria-label={`Puntaje ${scoreFormat.format(score)} de 1`}>
        <circle className="gauge-track" cx="90" cy="90" r={radius} />
        <circle
          className="gauge-value"
          cx="90" cy="90" r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
          transform="rotate(-90 90 90)"
        />
        <circle className="gauge-threshold" cx={tx} cy={ty} r="5" />
      </svg>
      <div className="gauge-center">
        <strong>{percentFormat.format(clamped)}</strong>
        <span>puntaje glaucoma</span>
      </div>
    </div>
  )
}

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [view, setView] = useState('overlay')
  const activeRequest = useRef(null)

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  useEffect(() => () => activeRequest.current?.abort(), [])

  const changeImage = () => {
    activeRequest.current?.abort()
    activeRequest.current = null
    setResult(null)
    setLoading(false)
    setError('')
    setImage(null)
    setPreview(null)
    setView('overlay')
  }

  const handleFile = (file) => {
    if (!file) return
    changeImage()
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Selecciona una imagen JPG o PNG.')
    } else if (!file.size || file.size > maxImageBytes) {
      setError('Selecciona una imagen con contenido de hasta 10 MiB.')
    } else {
      setImage(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  const handleImageChange = (event) => {
    handleFile(event.target.files[0])
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files[0])
  }

  const analyzeImage = async () => {
    if (!image || activeRequest.current) return
    const request = new AbortController()
    activeRequest.current = request
    setLoading(true)
    setError('')
    setResult(null)
    const timeout = window.setTimeout(() => request.abort(), 60_000)
    try {
      const response = await fetch(`${apiUrl}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': image.type },
        body: image,
        cache: 'no-store',
        signal: request.signal,
      })
      const data = response.headers.get('content-type')?.includes('application/json') ? await response.json() : null
      if (activeRequest.current !== request) return
      if (response.ok && data) {
        setResult(data)
      } else {
        setError(typeof data?.detail === 'string' ? data.detail : connectionError)
      }
    } catch {
      if (activeRequest.current === request) {
        setError(request.signal.aborted ? 'El análisis tardó demasiado. Intenta nuevamente.' : connectionError)
      }
    } finally {
      window.clearTimeout(timeout)
      if (activeRequest.current === request) {
        activeRequest.current = null
        setLoading(false)
      }
    }
  }

  const glaucoma = result?.label === 1
  const views = result ? {
    original: { src: preview, label: 'Original', alt: 'Imagen original del fondo de ojo' },
    overlay: { src: 'data:image/png;base64,' + result.overlay_png, label: 'Superposición', alt: 'Grad-CAM superpuesto sobre la imagen original' },
    heatmap: { src: 'data:image/png;base64,' + result.heatmap_png, label: 'Mapa de calor', alt: 'Mapa de calor Grad-CAM del logit de glaucoma' },
  } : null

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <IrisMark />
          <div>
            <span className="brand-name">Glauco<em>AI</em></span>
            <span className="brand-sub">Detección asistida de glaucoma</span>
          </div>
        </div>
        <span className="pill">Proyecto académico · Deep Learning</span>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow"><span className="dot" /> Análisis de retinografía con IA</span>
          <h1>
            El glaucoma avanza <span className="grad">en silencio</span>.<br />
            Detectarlo a tiempo cambia todo.
          </h1>
          <p className="lead">
            Es una de las principales causas de ceguera irreversible en el mundo y muchas personas
            no saben que lo tienen. GlaucoAI analiza una imagen de fondo de ojo y estima, con una red
            neuronal convolucional, si presenta signos compatibles con glaucoma.
          </p>
          <div className="hero-stats">
            <div><strong>Fondo de ojo</strong><span>retinografía a color</span></div>
            <div><strong>ResNet</strong><span>transfer learning</span></div>
            <div><strong>Grad-CAM</strong><span>explicabilidad visual</span></div>
          </div>
        </div>
        <div className="hero-eye" aria-hidden="true">
          <div className="ring r1" />
          <div className="ring r2" />
          <div className="ring r3" />
          <div className="iris">
            <div className="pupil" />
            <div className="glint" />
          </div>
          <div className="scanline" />
        </div>
      </section>

      <main className="workspace">
        <section className="card upload-card">
          <div className="card-head">
            <span className="step-badge">1</span>
            <div>
              <h2>Carga la imagen</h2>
              <p>Arrastra una retinografía o selecciónala desde tu equipo.</p>
            </div>
          </div>

          <label
            className={`dropzone ${dragging ? 'is-dragging' : ''} ${preview ? 'has-image' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            {preview ? (
              <div className="preview">
                <img src={preview} alt="Vista previa del fondo de ojo" />
                {loading && <div className="scan-overlay"><div className="scan-beam" /></div>}
              </div>
            ) : (
              <div className="dz-empty">
                <div className="dz-icon">
                  <svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true">
                    <path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"
                      fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <strong>Arrastra tu imagen aquí</strong>
                <span>o haz clic para buscarla</span>
                <small>JPG o PNG · hasta 10 MiB</small>
              </div>
            )}
            <input
              type="file"
              aria-label="Seleccionar imagen de fondo de ojo"
              accept="image/png, image/jpeg"
              onChange={handleImageChange}
            />
          </label>

          {image && (
            <div className="file-row">
              <span className="file-name" title={image.name}>{image.name}</span>
              <button type="button" className="ghost" onClick={changeImage}>Cambiar</button>
            </div>
          )}

          <button type="button" className="primary" disabled={!image || loading} onClick={analyzeImage}>
            {loading ? <><span className="spinner" /> Analizando…</> : 'Analizar imagen'}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        <section className={`card result-card ${result ? (glaucoma ? 'tone-alert' : 'tone-healthy') : ''}`}
          aria-busy={loading}>
          <div className="card-head">
            <span className="step-badge">2</span>
            <div>
              <h2>Resultado</h2>
              <p>{result ? 'Resultado experimental del modelo' : 'Aquí verás la clasificación y el mapa Grad-CAM.'}</p>
            </div>
          </div>

          <div role="status">
            {result ? (
              <div className="result">
                <div className="verdict">
                  <ScoreGauge score={result.score_glaucoma} threshold={result.threshold} glaucoma={glaucoma} />
                  <div className="verdict-text">
                    <span className={`tag ${glaucoma ? 'tag-alert' : 'tag-healthy'}`}>
                      {glaucoma ? 'Signos compatibles' : 'Sin signos detectados'}
                    </span>
                    <h3>{glaucoma ? 'Glaucoma' : 'Sano'}</h3>
                    <dl>
                      <div><dt>Puntaje</dt><dd>{scoreFormat.format(result.score_glaucoma)}</dd></div>
                      <div><dt>Umbral</dt><dd>{String(result.threshold).replace('.', ',')}</dd></div>
                      {result.model && <div><dt>Modelo</dt><dd>{modelNames[result.model] ?? result.model}</dd></div>}
                    </dl>
                  </div>
                </div>

                <div className="viewer">
                  <div className="tabs" role="tablist" aria-label="Vista de la imagen">
                    {Object.entries(views).map(([key, v]) => (
                      <button key={key} type="button" role="tab" aria-selected={view === key}
                        className={view === key ? 'active' : ''} onClick={() => setView(key)}>
                        {v.label}
                      </button>
                    ))}
                  </div>
                  <figure className="viewer-frame">
                    <img src={views[view].src} alt={views[view].alt} />
                  </figure>
                  <p className="note">
                    {!result.has_signal && 'El mapa es uniforme para esta imagen. '}
                    Grad-CAM resalta las regiones asociadas al puntaje del modelo; no garantiza localizar estructuras médicamente relevantes.
                  </p>
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <div className={`pulse-eye ${loading ? 'is-loading' : ''}`}><IrisMark size={64} /></div>
                <p>{loading ? 'Calculando el puntaje y el mapa Grad-CAM…' : 'Carga una imagen y pulsa «Analizar imagen».'}</p>
              </div>
            )}
          </div>
        </section>
      </main>

      {preview && (
        <a className="journey-link" href="#recorrido">
          Ver cómo la red procesa esta imagen, paso a paso ↓
        </a>
      )}

      <NetworkJourney preview={preview} result={result} />

      <section className="how">
        <h2>¿Cómo funciona?</h2>
        <div className="how-grid">
          {steps.map((s) => (
            <article key={s.n} className="how-item">
              <span className="how-n">{s.n}</span>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
            </article>
          ))}
        </div>
      </section>

      <aside className="disclaimer">
        <strong>Aviso importante.</strong> Esta herramienta tiene fines académicos y experimentales y no
        ofrece un diagnóstico médico. El modelo puede producir falsos positivos y falsos negativos.
        Ante cualquier duda, consulta con un oftalmólogo.
      </aside>

      <footer className="footer">GlaucoAI · Proyecto de Deep Learning · Dataset PAPILA</footer>
    </div>
  )
}

export default App
