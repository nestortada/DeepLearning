import { useEffect, useRef, useState } from 'react'

const SIZE = 224

const KERNELS = [
  { name: 'Bordes verticales', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  { name: 'Bordes horizontales', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  { name: 'Contornos (Laplaciano)', k: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] },
  { name: 'Realce de detalle', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
]

const STAGES = [
  { name: 'conv1 + maxpool', ch: 64, size: 56, what: 'bordes y cambios de color' },
  { name: 'layer1', ch: 64, size: 56, what: 'texturas simples' },
  { name: 'layer2', ch: 128, size: 28, what: 'vasos y curvas' },
  { name: 'layer3', ch: 256, size: 14, what: 'disco y copa óptica' },
  { name: 'layer4', ch: 512, size: 7, what: 'patrones de la excavación' },
]

// --- Utilidades de imagen (se ejecutan de verdad sobre la imagen subida) ---

function letterbox(img) {
  const c = document.createElement('canvas')
  c.width = c.height = SIZE
  const g = c.getContext('2d')
  g.fillStyle = '#000'
  g.fillRect(0, 0, SIZE, SIZE)
  const s = SIZE / Math.max(img.width, img.height)
  const w = img.width * s
  const h = img.height * s
  g.drawImage(img, (SIZE - w) / 2, (SIZE - h) / 2, w, h)
  return g.getImageData(0, 0, SIZE, SIZE)
}

function toGray(data) {
  const out = new Float32Array(SIZE * SIZE)
  for (let i = 0; i < out.length; i++) {
    out[i] = (0.299 * data.data[i * 4] + 0.587 * data.data[i * 4 + 1] + 0.114 * data.data[i * 4 + 2]) / 255
  }
  return out
}

function convolve(src, w, h, k) {
  const out = new Float32Array(w * h)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) sum += src[(y + j) * w + x + i] * k[j + 1][i + 1]
      out[y * w + x] = sum
    }
  }
  return out
}

const relu = (src) => src.map((v) => Math.max(0, v))

function maxPool(src, w, h, f = 2) {
  const nw = Math.floor(w / f)
  const nh = Math.floor(h / f)
  const out = new Float32Array(nw * nh)
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      let m = -Infinity
      for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) m = Math.max(m, src[(y * f + j) * w + x * f + i])
      out[y * nw + x] = m
    }
  }
  return { data: out, w: nw, h: nh }
}

// Mapa de color tipo "inferno" simplificado
function colorize(t) {
  const stops = [[5, 11, 24], [40, 40, 120], [56, 140, 200], [56, 225, 212], [255, 230, 140]]
  const p = Math.min(Math.max(t, 0), 1) * (stops.length - 1)
  const i = Math.floor(p)
  const f = p - i
  const a = stops[i]
  const b = stops[Math.min(i + 1, stops.length - 1)]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

function paint(canvas, src, w, h, { signed = false } = {}) {
  if (!canvas) return
  canvas.width = w
  canvas.height = h
  const g = canvas.getContext('2d')
  const img = g.createImageData(w, h)
  let lo = Infinity
  let hi = -Infinity
  for (const v of src) { if (v < lo) lo = v; if (v > hi) hi = v }
  if (signed) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m }
  const range = hi - lo || 1
  for (let i = 0; i < w * h; i++) {
    const [r, gg, b] = colorize((src[i] - lo) / range)
    img.data[i * 4] = r
    img.data[i * 4 + 1] = gg
    img.data[i * 4 + 2] = b
    img.data[i * 4 + 3] = 255
  }
  g.putImageData(img, 0, 0)
}

function paintChannel(canvas, data, ch) {
  if (!canvas) return
  canvas.width = canvas.height = SIZE
  const g = canvas.getContext('2d')
  const img = g.createImageData(SIZE, SIZE)
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data[i * 4 + ch] = data.data[i * 4 + ch]
    img.data[i * 4 + 3] = 255
  }
  g.putImageData(img, 0, 0)
}

function Step({ n, title, children, active }) {
  return (
    <article className={`jr-step ${active ? 'is-active' : 'is-pending'}`}>
      <div className="jr-rail"><span className="jr-n">{n}</span></div>
      <div className="jr-body">
        <h3>{title}</h3>
        {children}
      </div>
    </article>
  )
}

function Kernel({ k }) {
  return (
    <div className="kernel" aria-label="Filtro 3×3">
      {k.flat().map((v, i) => <span key={i} className={v > 0 ? 'pos' : v < 0 ? 'neg' : ''}>{v}</span>)}
    </div>
  )
}

export default function NetworkJourney({ preview, result }) {
  const [features, setFeatures] = useState(null)
  const [kernelIdx, setKernelIdx] = useState(0)
  const inputRef = useRef(null)
  const chRefs = [useRef(null), useRef(null), useRef(null)]
  const convRef = useRef(null)
  const reluRef = useRef(null)
  const poolRef = useRef(null)
  const deepRefs = useRef([])

  // Calcula las etapas reales a partir de la imagen subida
  useEffect(() => {
    if (!preview) return
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      const data = letterbox(img)
      setFeatures({ data, gray: toGray(data) })
    }
    img.src = preview
    return () => { cancelled = true }
  }, [preview])

  useEffect(() => {
    if (!features) return
    const { data } = features
    if (inputRef.current) {
      inputRef.current.width = inputRef.current.height = SIZE
      inputRef.current.getContext('2d').putImageData(data, 0, 0)
    }
    chRefs.forEach((r, i) => paintChannel(r.current, data, i))
  }, [features]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!features) return
    const conv = convolve(features.gray, SIZE, SIZE, KERNELS[kernelIdx].k)
    paint(convRef.current, conv, SIZE, SIZE, { signed: true })
    const r = relu(conv)
    paint(reluRef.current, r, SIZE, SIZE)
    const p = maxPool(r, SIZE, SIZE)
    paint(poolRef.current, p.data, p.w, p.h)
    // Etapas profundas: agrupaciones sucesivas para ilustrar la reducción espacial
    let cur = { data: r, w: SIZE, h: SIZE }
    STAGES.forEach((s, i) => {
      while (cur.w > s.size) cur = maxPool(cur.data, cur.w, cur.h)
      paint(deepRefs.current[i], cur.data, cur.w, cur.h)
    })
  }, [features, kernelIdx])

  const score = result?.score_glaucoma
  const threshold = result?.threshold ?? 0.5
  const logit = score != null ? Math.log(Math.max(score, 1e-6) / Math.max(1 - score, 1e-6)) : null
  const fmt = (v, d = 3) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d })

  return (
    <section className="journey" id="recorrido">
      <div className="journey-head">
        <span className="eyebrow"><span className="dot" /> Dentro de la red neuronal</span>
        <h2>¿Qué hace la red con <span className="grad">tu imagen</span>?</h2>
        <p>
          Este es el recorrido paso a paso, desde los píxeles hasta la clasificación. Los pasos 1 a 4 se calculan
          de verdad en tu navegador sobre la imagen que subes, con filtros clásicos. Los pasos 5 y 6 resumen las
          capas internas de ResNet34, y los pasos 7 y 8 usan la respuesta real del modelo.
        </p>
      </div>

      {!preview ? (
        <div className="journey-empty">Sube una imagen arriba para ver el recorrido completo.</div>
      ) : (
        <div className="jr-list">
          <Step n="1" title="Entrada: se ajusta a 224 × 224 píxeles" active>
            <p>La red siempre recibe el mismo tamaño. La imagen se reescala sin deformarla y se rellena con negro (<em>letterbox</em>).</p>
            <div className="jr-row">
              <figure><canvas ref={inputRef} /><figcaption>224 × 224 × 3</figcaption></figure>
            </div>
          </Step>

          <Step n="2" title="Tres canales de color: rojo, verde y azul" active>
            <p>
              Cada píxel son tres números entre 0 y 255. Se normalizan con la media y la desviación de ImageNet,
              igual que en el preentrenamiento. El canal verde suele mostrar mejor los vasos y el disco óptico.
            </p>
            <div className="jr-row three">
              {['Rojo', 'Verde', 'Azul'].map((c, i) => (
                <figure key={c}><canvas ref={chRefs[i]} /><figcaption>Canal {c.toLowerCase()}</figcaption></figure>
              ))}
            </div>
          </Step>

          <Step n="3" title="Convolución: un filtro recorre la imagen" active>
            <p>
              Una neurona convolucional es una pequeña matriz de pesos (3 × 3) que se desliza por la imagen. En
              cada posición multiplica y suma los píxeles vecinos, y el resultado es un <strong>mapa de
              características</strong>. Elige un filtro para ver qué detecta:
            </p>
            <div className="chips">
              {KERNELS.map((k, i) => (
                <button key={k.name} type="button" className={i === kernelIdx ? 'active' : ''} onClick={() => setKernelIdx(i)}>
                  {k.name}
                </button>
              ))}
            </div>
            <div className="jr-row conv-row">
              <Kernel k={KERNELS[kernelIdx].k} />
              <span className="op">∗ imagen =</span>
              <figure><canvas ref={convRef} /><figcaption>Mapa de características</figcaption></figure>
            </div>
            <p className="jr-note">En ResNet34 estos pesos no los diseña nadie: se aprenden durante el entrenamiento, y la primera capa tiene 64 filtros.</p>
          </Step>

          <Step n="4" title="Activación ReLU y max pooling" active>
            <p>
              <strong>ReLU</strong> deja pasar solo los valores positivos (<code>max(0, x)</code>), es decir, las
              zonas donde el filtro &quot;se activó&quot;. Después, el <strong>max pooling</strong> toma el máximo de
              cada bloque de 2 × 2: reduce el tamaño a la mitad y conserva lo más importante.
            </p>
            <div className="jr-row three">
              <figure><canvas ref={reluRef} /><figcaption>Tras ReLU · 224 × 224</figcaption></figure>
              <span className="op">→</span>
              <figure><canvas ref={poolRef} className="pixel" /><figcaption>Max pooling · 112 × 112</figcaption></figure>
            </div>
          </Step>

          <Step n="5" title="Capas profundas: de bordes a estructuras" active>
            <p>
              ResNet34 apila 33 capas convolucionales en 4 bloques con <strong>conexiones residuales</strong>, que
              suman la entrada de cada bloque a su salida para que la señal no se pierda. A medida que se profundiza,
              la resolución baja y los canales aumentan: la red pasa de ver bordes a reconocer formas como el disco
              y la copa óptica, clave en el glaucoma.
            </p>
            <div className="stages">
              {STAGES.map((s, i) => (
                <div key={s.name} className="stage">
                  <canvas ref={(el) => { deepRefs.current[i] = el }} className="pixel" />
                  <strong>{s.name}</strong>
                  <span>{s.size} × {s.size} × {s.ch}</span>
                  <small>{s.what}</small>
                </div>
              ))}
            </div>
            <p className="jr-note">Ilustración: las miniaturas muestran la reducción de resolución aplicada al filtro elegido, no los canales internos reales del modelo.</p>
          </Step>

          <Step n="6" title="Global average pooling: 512 números" active={!!result}>
            <p>
              Cada uno de los 512 mapas de 7 × 7 se resume en su promedio. La imagen completa queda convertida en un
              vector de 512 números que describe qué patrones aparecieron.
            </p>
            <div className="vector" aria-hidden="true">
              {Array.from({ length: 48 }, (_, i) => (
                <span key={i} style={{ height: `${20 + Math.abs(Math.sin(i * 1.7 + (score ?? 0.5) * 9)) * 80}%` }} />
              ))}
            </div>
            <p className="jr-note">Ilustración de un vector de características.</p>
          </Step>

          <Step n="7" title="Clasificador: neurona final y sigmoide" active={!!result}>
            {result ? (
              <>
                <p>
                  Una última neurona combina los 512 valores con sus pesos y produce un número llamado
                  <strong> logit</strong>. La función <strong>sigmoide</strong> lo convierte en un puntaje entre 0 y 1.
                </p>
                <div className="formula">
                  <div><span>logit z</span><strong>{fmt(logit)}</strong></div>
                  <span className="op">→ σ(z) = 1 / (1 + e<sup>−z</sup>) →</span>
                  <div><span>puntaje</span><strong>{fmt(score, 4)}</strong></div>
                  <span className="op">{score >= threshold ? '≥' : '<'} umbral {fmt(threshold, 2)} →</span>
                  <div className={result.label === 1 ? 'is-alert' : 'is-healthy'}>
                    <span>clase</span><strong>{result.label === 1 ? 'Glaucoma' : 'Sano'}</strong>
                  </div>
                </div>
                <div className="sigmoid">
                  <svg viewBox="0 0 300 120" role="img" aria-label="Curva sigmoide con la posición del puntaje">
                    <line x1="0" y1={110 - threshold * 100} x2="300" y2={110 - threshold * 100} className="sg-th" />
                    <path className="sg-curve" d={Array.from({ length: 61 }, (_, i) => {
                      const z = -6 + i * 0.2
                      return `${i ? 'L' : 'M'}${i * 5},${110 - 100 / (1 + Math.exp(-z))}`
                    }).join(' ')} />
                    <circle className={`sg-dot ${result.label === 1 ? 'is-alert' : 'is-healthy'}`}
                      cx={Math.min(Math.max((logit + 6) * 25, 0), 300)} cy={110 - score * 100} r="6" />
                  </svg>
                </div>
              </>
            ) : <p className="jr-wait">Pulsa «Analizar imagen» para ver el cálculo real del modelo.</p>}
          </Step>

          <Step n="8" title="Grad-CAM: ¿dónde miró la red?" active={!!result}>
            {result ? (
              <>
                <p>
                  Grad-CAM calcula cuánto influyó cada zona del último bloque convolucional en el logit de glaucoma y lo
                  dibuja como un mapa de calor sobre la imagen original.
                </p>
                <div className="jr-row three">
                  <figure><img src={'data:image/png;base64,' + result.heatmap_png} alt="Mapa de calor Grad-CAM" /><figcaption>Mapa de calor</figcaption></figure>
                  <span className="op">+</span>
                  <figure><img src={'data:image/png;base64,' + result.overlay_png} alt="Grad-CAM superpuesto" /><figcaption>Superposición</figcaption></figure>
                </div>
              </>
            ) : <p className="jr-wait">Aparecerá cuando el modelo devuelva su resultado.</p>}
          </Step>
        </div>
      )}
    </section>
  )
}
