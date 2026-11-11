import { useEffect, useRef, useState } from 'react'
import './App.css'

const maxImageBytes = 10 * 1024 * 1024
const scoreFormat = new Intl.NumberFormat('es-CO', { minimumFractionDigits: 4, maximumFractionDigits: 4 })
const modelNames = { resnet34: 'ResNet34', resnet18: 'ResNet18', efficientnet_b0: 'EfficientNet-B0' }
const connectionError = 'No se pudo conectar con el servidor de análisis. Comprueba que esté iniciado.'
const apiUrl = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '')

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
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
        headers: {
          'Content-Type': image.type,
        },
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

  return (
    <main className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">◉</div>
          <div>
            <h1>GlaucoAI</h1>
            <p>Inteligencia artificial para la salud visual</p>
          </div>
        </div>
      </header>

      <section className="main-content">
        <div className="welcome">
          <span className="welcome-tag">TU VISIÓN NOS IMPORTA</span>
          <h2>¡Hola! 👋</h2>
          <h3>Exploremos la salud visual.</h3>
          <p>
            Carga una imagen de fondo de ojo para analizarla
            mediante nuestro modelo de inteligencia artificial.
          </p>
        </div>

        <section className="upload-card">
          <div className="upload-heading">
            <h2>Analiza una imagen</h2>
            <p>Selecciona una imagen para comenzar.</p>
          </div>

          <label
            className="upload-area"
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleDrop}
          >
            {preview ? (
              <img src={preview} alt="Vista previa del fondo de ojo" />
            ) : (
              <>
                <span className="upload-icon">↑</span>
                <strong>Arrastra tu imagen aquí</strong>
                <span>o haz clic para buscarla en tu equipo</span>
                <small>Formatos admitidos: JPG y PNG · Hasta 10 MiB</small>
              </>
            )}

            <input
              type="file"
              aria-label="Seleccionar imagen de fondo de ojo"
              accept="image/png, image/jpeg"
              onChange={handleImageChange}
            />
          </label>

          {image && (
            <div className="file-info">
              <p className="file-name">Imagen seleccionada: {image.name}</p>
              <button type="button" className="change-image" onClick={changeImage}>
                Cambiar imagen
              </button>
            </div>
          )}

          <button type="button" className="analyze-button" disabled={!image || loading}
            onClick={analyzeImage}>
            {loading ? 'Analizando imagen…' : 'Analizar imagen'}
          </button>
          {error && <p className="analysis-error" role="alert">{error}</p>}

          <p className="disclaimer">
            Esta herramienta tiene fines académicos y experimentales.
            No ofrece un diagnóstico médico. El modelo puede producir falsos positivos y falsos negativos.
          </p>
          <div className="result-placeholder" aria-busy={loading}>
            <div className="result-icon" aria-hidden="true">✦</div>
            <div role="status">
              <h3>{result ? 'Resultado experimental' : 'Tu resultado aparecerá aquí'}</h3>
              {result ? (
                <>
                  {result.model && <p>Modelo utilizado <strong>{modelNames[result.model] ?? result.model}</strong></p>}
                  <p>Clase asignada por el modelo: <strong>{result.label === 1 ? 'Glaucoma' : 'Sano'}</strong></p>
                  <p>Puntaje experimental de glaucoma: <strong>{scoreFormat.format(result.score_glaucoma)}</strong></p>
                  <p>Escala de 0 a 1 sin calibración clínica. Umbral de decisión <strong>{String(result.threshold).replace('.', ',')}</strong>.</p>
                </>
              ) : (
                <p>{loading ? 'Generando el puntaje y Grad-CAM…' : 'Consulta la clasificación, el puntaje experimental y el mapa Grad-CAM.'}</p>
              )}
            </div>
            {result && (
              <>
                <div className="result-images">
                  <figure>
                    <img src={preview} alt="Imagen original del fondo de ojo" />
                    <figcaption>Imagen original</figcaption>
                  </figure>
                  <figure>
                    <img src={'data:image/png;base64,' + result.overlay_png} alt="Grad-CAM superpuesto sobre la imagen original" />
                    <figcaption>Superposición</figcaption>
                  </figure>
                  <figure>
                    <img src={'data:image/png;base64,' + result.heatmap_png} alt="Mapa de calor Grad-CAM del logit de glaucoma" />
                    <figcaption>Mapa de calor Grad-CAM</figcaption>
                  </figure>
                </div>
                {!result.has_signal && <p>El mapa es uniforme para esta imagen.</p>}
                <p>Grad-CAM resalta regiones asociadas al puntaje del modelo y no garantiza localizar estructuras médicamente relevantes.</p>
              </>
            )}
          </div>
        </section>
      </section>

      <footer className="footer">
        GlaucoAI · Proyecto de Deep Learning
      </footer>
    </main>
  )
}

export default App
