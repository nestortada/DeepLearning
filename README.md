# GlaucoAI

GlaucoAI es una demostración académica local que clasifica imágenes de fondo de ojo con ResNet18. La interfaz React muestra la clase, el puntaje experimental sin calibrar y un mapa Grad-CAM. El proyecto incluye una CNN de referencia, experimentos históricos de hiperparámetros y un protocolo reproducible de evaluación interna con PAPILA.

> **Aviso:** proyecto académico/educativo. No es un instrumento de diagnóstico clínico.

## Contenido
- [Dataset](#dataset)
- [Preparación reproducible de PAPILA](#preparación-reproducible-de-papila)
- [Carga de PAPILA y CUDA](#carga-de-papila-y-cuda)
- [Entrenamiento de la CNN e inferencia local](#entrenamiento-de-la-cnn-e-inferencia-local)
- [Transfer Learning con ResNet18](#transfer-learning-con-resnet18)
- [Evaluación interna congelada de PAPILA](#evaluación-interna-congelada-de-papila)
- [Grad-CAM de ResNet18 en validación](#grad-cam-de-resnet18-en-validación)
- [Instalación y uso en Windows](#instalación-y-uso-en-windows)
- [Docker, Vercel y Render](#docker-vercel-y-render)
- [Experimentos históricos del notebook](#experimentos-históricos-del-notebook)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Limitaciones](#limitaciones)

## Dataset
PAPILA reúne imágenes de ambos ojos de 244 pacientes, datos clínicos y contornos de disco y copa. La tabla siguiente describe la partición del notebook histórico. El manifiesto congelado de la evaluación reproducible usa 274 imágenes de entrenamiento, 62 de validación y 84 de test. [Artículo de PAPILA](https://doi.org/10.1038/s41597-022-01388-1)

- 488 imágenes en total; se excluye la clase *sospechoso*, quedan **420 imágenes**.
- 210 pacientes, con ambos ojos (OD y OS).
- Distribución: 333 sanos (79 %) y 87 glaucoma (21 %). **Desbalanceado.**
- Resolución original 2576 × 1934 px, redimensionada a **96 × 96 px**.
- Partición **estratificada por paciente** (semilla 42), para que los dos ojos de un mismo paciente no estén en conjuntos distintos:

| Split | Imágenes | % glaucoma |
|-------|----------|------------|
| Train | 272 | 21.0 % |
| Val   | 64  | 20.3 % |
| Test  | 84  | 20.2 % |

## Preparación reproducible de PAPILA

Esta etapa usa la [copia de PAPILA utilizada por el notebook](https://github.com/valentinaache/glaucomaPGUSab/tree/41f0a0718252dbed301adf90976b67688d918551), fijada al commit `41f0a0718252dbed301adf90976b67688d918551`. Consúltese también el [artículo de PAPILA](https://doi.org/10.1038/s41597-022-01388-1) y la [publicación oficial en Figshare](https://figshare.com/articles/dataset/PAPILA/14798004), que lista la versión 2 y la licencia GPL 3.0+. La corrida de verificación de esta etapa se hizo sobre la copia fijada de GitHub.

Desde la raíz del proyecto, con Python 3.11 o posterior, coloca los datos en `data/papila/`. Esa ruta debe contener `FundusImages/` y `ClinicalData/`, con `patient_data_od.xlsx` y `patient_data_os.xlsx` dentro de esta última carpeta.

```bash
python -m pip install -r requirements-data.txt
python -m src.prepare_papila --dataset-dir data/papila --output data/papila_manifest.csv --seed 42
python -m unittest discover -s tests -v
```

El comando indexa y decodifica los JPEG, y los une con las etiquetas por paciente y ojo. En la copia fijada, la planilla `os` contiene ambos ojos con lateralidad explícita. La planilla `od` contiene únicamente ojos derechos, con lateralidad implícita. Se concilian las coincidencias entre planillas y se rechazan los duplicados dentro de una planilla, las contradicciones y los registros sin pareja. El manifiesto conserva las clases 0 y 1, con una fila por ojo elegible y las columnas `image_id`, `patient_id`, `eye`, `label`, `relative_path` y `split`. Las rutas son relativas a `--dataset-dir`.

La partición usa la semilla **42** y agrupa a cada paciente por la combinación de diagnósticos de sus ojos elegibles. Así, los pacientes con un ojo sano y otro con glaucoma forman un estrato propio y cada ojo conserva su etiqueta. Dentro de cada estrato se ordenan los identificadores, se barajan con la semilla y se asignan pacientes a `train`, `val` y `test` en proporción aproximada de **65/15/20**, usando redondeo por restos mayores. Los empates se resuelven en el orden `train`, `val`, `test`. Los estratos pequeños pueden quedar ausentes de algún conjunto.

Se verificó la decodificación de **488 imágenes**, se conciliaron **244** etiquetas derechas redundantes y se excluyeron **68** ojos sospechosos. El manifiesto contiene **420 imágenes** de **210 pacientes**, con cero pacientes compartidos entre conjuntos.

| Split | Pacientes | Imágenes | Sanos (0) | Glaucoma (1) |
|-------|----------:|---------:|----------:|-------------:|
| Train | 137 | 274 | 217 | 57 |
| Val   | 31  | 62  | 49  | 13 |
| Test  | 42  | 84  | 67  | 17 |

Los conteos pasan de 272/64/84 en el experimento histórico a 274/62/84 con el nuevo método. Coincidir en el tamaño de prueba no implica que los pacientes sean los mismos. Dos ejecuciones sobre la misma fuente con semilla 42 produjeron CSV idénticos, con SHA-256 `c82776614b94d458e42e56b8f66270e1fbe0be9b4a80a741e2cb6923449ef5cd`. Los resultados de las secciones siguientes corresponden a la partición histórica.

## Carga de PAPILA y CUDA

El entorno verificado usa Windows, Python `3.14.6`, PyTorch `2.14.1+cu130` y torchvision `0.29.1+cu130`. CUDA `13.0` se verificó en una NVIDIA GeForce RTX 4050 Laptop GPU de 6 GB con el controlador `621.01`. En la `.venv` existente, ejecuta desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para reproducir ese entorno en una copia nueva del repositorio, usa Python 3.14, un controlador NVIDIA compatible y el índice oficial `cu130`. Estos comandos son para una instalación nueva. [Instalación de PyTorch](https://pytorch.org/get-started/locally/)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

Usa el CSV de la etapa anterior. `make_loaders` conserva sus asignaciones y valida que ningún paciente cruce entre conjuntos. Los valores iniciales son `batch_size=16`, `num_workers=0` e imágenes RGB de `96 × 96`. La media y la desviación por canal se calculan en una pasada por `train` sin *augmentation*. La decodificación se hace bajo demanda en CPU y el volteo horizontal aleatorio se aplica solo en `train`. El argumento `seed` controla el orden de carga y `torch.manual_seed(42)` inicializa también el generador usado por el *augmentation* con `num_workers=0`.

```python
from pathlib import Path

import torch

from src.data import make_loaders
from src.device import select_device

torch.manual_seed(42)
device = select_device()
loaders = make_loaders(Path("data/papila"), Path("data/papila_manifest.csv"), device=device)
images, labels = device.move_batch(next(iter(loaders.train)))
```

El cargador real de prueba se habilita con `include_test=True` para la evaluación final. Con CUDA, los lotes usan memoria fijada y se transfieren con `non_blocking=True`. `device.autocast()` y `device.make_grad_scaler()` preparan AMP en FP16 y quedan deshabilitados en CPU. Al implementar la CNN entrenable con AMP, usa logits y `BCEWithLogitsLoss`, y aplica la sigmoide al obtener puntajes de predicción. [Documentación de DataLoader](https://docs.pytorch.org/docs/2.14/data.html) · [Documentación de AMP](https://docs.pytorch.org/docs/2.14/amp.html)

## Entrenamiento de la CNN e inferencia local

La referencia usa la CNN histórica con salida de logits, Adam a `0.001`, BCE con logits, semilla `42`, lote `16`, imágenes RGB de `96 × 96`, `num_workers=0`, volteo horizontal en train, dropout y weight decay en `0`. Se detiene tras cinco épocas consecutivas sin mejora estricta de la pérdida de validación, hasta un máximo de 40. La época seleccionada es la de menor pérdida de validación. El umbral exploratorio de `0.5` se usa para las métricas y la clase predicha. PR-AUC es el área trapezoidal de la curva precisión–recall. ROC-AUC y PR-AUC quedan vacías cuando falta una clase en validación.

Usa el manifiesto existente y una carpeta nueva por corrida. El bucle carga exclusivamente imágenes de train y val. En CUDA, el entrenamiento usa AMP FP16 y registra pasos omitidos por desbordamiento de gradientes. La validación y la inferencia usan FP32. El comando fija las semillas y activa los algoritmos deterministas de PyTorch. Esta corrida usa la partición nueva y AMP, por lo que sus métricas de validación corresponden a un experimento distinto del histórico. [Reproducibilidad de PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-smoke --epochs 1
.\.venv\Scripts\python.exe -m src.train --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-reference
```

Cada corrida escribe `history.csv`, `loss.png`, `summary.json` y `best.pt` en su carpeta. El resumen registra el máximo de memoria asignada y reservada por PyTorch en CUDA. El checkpoint contiene pesos, época seleccionada, configuración, hash del manifiesto y normalización de train. Git excluye `models/runs/`. Consulta `python -m src.train --help` para ajustar los parámetros, y usa `--cpu` o `--no-amp` cuando corresponda.

```powershell
.\.venv\Scripts\python.exe -m src.predict --checkpoint models/runs/papila-reference/best.pt --image path/to/fundus.jpg
```

La salida de inferencia es un puntaje académico de glaucoma sin calibrar. La aplicación de la sigmoide convierte el logit al rango `[0, 1]`.

## Transfer Learning con ResNet18

La ResNet18 usa `torchvision 0.29.1+cu130` y los pesos oficiales `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`). El backbone y sus estadísticas de BatchNorm quedan congelados. Solo se ajusta la cabeza lineal de 512 entradas y un logit, con 513 parámetros entrenables. [Pesos y normalización de Torchvision](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.resnet18.html)

La entrada RGB se ajusta dentro de `224 × 224` con interpolación bilineal, conservación de aspecto y padding negro centrado. Una imagen PAPILA de `2576 × 1934` ocupa `224 × 168`, con 28 píxeles de padding arriba y abajo. Se escala a `[0, 1]` y se aplica la media `(0.485, 0.456, 0.406)` y desviación `(0.229, 0.224, 0.225)` de ImageNet. Se sustituye el recorte central de las transformaciones oficiales por padding para conservar el campo visual. El único aumento es el volteo horizontal con probabilidad `0.5` en train.

El entrenamiento usa el mismo manifiesto, `batch_size=16`, `num_workers=0`, Adam a `0.001`, semilla `42`, BCE con logits y AMP FP16 en CUDA. Se restaura la época de menor pérdida de validación con paciencia de cinco y máximo de 30 épocas. La validación usa FP32. Desde la raíz, con la `.venv` de CUDA y las dependencias de entrenamiento instaladas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --architecture resnet18 --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-resnet18-smoke --epochs 1
.\.venv\Scripts\python.exe -m src.train --architecture resnet18 --dataset-dir data/papila --manifest data/papila_manifest.csv
.\.venv\Scripts\python.exe -m src.compare --dataset-dir data/papila --manifest data/papila_manifest.csv
.\.venv\Scripts\python.exe -m src.predict --checkpoint models/runs/papila-resnet18/best.pt --image path/to/fundus.jpg
```

La comparación recarga ambos checkpoints en las mismas imágenes de validación y usa `0.50` como referencia. Guarda `comparison.json`, `curves.png`, `predictions.csv` y `thresholds.csv` en `models/runs/papila-resnet18/comparison/`. Average Precision (AP) pondera la precisión por cada incremento de recall. El área trapezoidal PR-AUC se registra como una métrica separada. La pérdida BCE es comparable sobre esas etiquetas, con la reserva de que cada modelo usa su propio preprocesamiento. Los tiempos totales incluyen preparación y guardado, y dependen del número de épocas ejecutadas. [Definición de Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

El análisis exploratorio usa una rejilla de `0.00` a `1.00` en pasos de `0.05`. Entre los umbrales con especificidad de al menos 80 %, maximiza sensibilidad y resuelve empates por mayor especificidad y luego mayor umbral. El punto elegido queda en el informe, separado del umbral `0.50` del checkpoint. Con 13 positivos, una sola imagen mueve la sensibilidad 7,7 puntos porcentuales. Las métricas de estos umbrales usan el mismo conjunto que los seleccionó. Una futura evaluación del test de PAPILA sería interna y exploratoria debido a su consulta histórica.

Un experimento posterior de ajuste parcial podría descongelar `layer4` con un learning rate inferior al de la cabeza y conservar el protocolo de validación agrupado por paciente. La corrida actual ajusta exclusivamente la cabeza y reporta puntajes académicos sin calibrar.

## Evaluación interna congelada de PAPILA

`configs/papila-internal-test.json` fija checkpoints, preprocesamiento, dependencias, umbrales e identidades SHA-256 del manifiesto y las imágenes de cada split. ResNet18, época 18, se evalúa a `0.35`. La CNN de referencia, época 8, se evalúa a `0.50`. Su umbral exploratorio de validación `0.55` se conserva como antecedente. Los resultados de validación en los umbrales aplicados se copiaron del informe de comparación previo, identificado por su hash.

Resultados ya registrados en el informe interno, en los mismos umbrales congelados. La validación sirvió para comparar modelos y fijar decisiones. El test interno tiene carácter exploratorio por su consulta histórica.

| Partición | Modelo | Umbral | Exactitud | Sensibilidad | Especificidad | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Validación (62) | CNN | 0,50 | 79,03 % | 38,46 % | 89,80 % | 0,7394 |
| Validación (62) | ResNet18 | 0,35 | 88,71 % | 69,23 % | 93,88 % | 0,8587 |
| Test interno (84) | CNN | 0,50 | 79,76 % | 47,06 % | 88,06 % | 0,7155 |
| Test interno (84) | ResNet18 | 0,35 | 82,14 % | 35,29 % | 94,03 % | 0,7954 |

El preflight verifica bytes y particiones y prueba ambos modelos con tensores sintéticos. La confirmación explícita ejecuta una pasada por checkpoint en test, en FP32 y modo inferencia. Comprueba que los parámetros y buffers conservan su hash. La carpeta local `models/runs/papila-internal-test/` se reserva antes de abrir el primer JPEG de test y su existencia bloquea otra ejecución. Guarda la configuración, métricas, predicciones por ojo agrupadas también por paciente, matrices, curvas e información de reproducibilidad. Los artefactos quedan fuera de Git.

Los intervalos exploratorios del 95 % usan 5.000 remuestreos percentiles por paciente con semilla 42. Cada extracción conserva ambos ojos y el mismo remuestreo para los dos modelos. Las réplicas con una métrica indefinida se omiten solo para esa métrica y se registra su cantidad. ROC-AUC y AP requieren ambas clases, sensibilidad y F requieren positivos reales, especificidad requiere negativos reales y precisión requiere predicciones positivas. [Bootstrap con datos oculares correlacionados](https://pmc.ncbi.nlm.nih.gov/articles/PMC7500131/)

PAPILA test se consultó en experimentos históricos. Esta evaluación es interna y exploratoria. Las decisiones de modelo, umbral y preprocesamiento se fijaron en validación antes de ejecutarla. Los resultados describen puntajes académicos sin calibrar.

## Grad-CAM de ResNet18 en validación

La capa objetivo es `model.backbone.layer4[-1]`. Para cada imagen se deriva el logit de glaucoma respecto a las activaciones `512 × 7 × 7`. Se promedian espacialmente los gradientes, se ponderan los canales y se aplica ReLU. El mapa se interpola a `224 × 224`, se recorta con la geometría exacta del letterbox y se normaliza dentro del contenido antes de reescalarlo al tamaño nativo. Cada mapa tiene su propia escala relativa. [Método Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.explain_validation --dataset-dir path/to/PAPILA --manifest path/to/manifest-c.csv
```

El comando verifica el manifiesto, el inventario de validación y los hashes de los checkpoints disponibles. Clasifica las imágenes de validación en CPU con el umbral congelado `0.35` y toma el primer `image_id` de cada categoría en orden lexicográfico. Luego genera Grad-CAM en CPU y, si hay CUDA, repite la comprobación en GPU con autocast y TF32 desactivados para la medición. `--cpu` limita la ejecución a CPU. `--alpha` controla la opacidad máxima, con valor predeterminado `0.55`.

La carpeta local `models/runs/papila-grad-cam/` contiene las figuras, los mapas nativos `.npy` y `report.json` con los IDs, las categorías ausentes, los puntajes, las mediciones y los hashes de entrada y salida. La variante `--cpu` usa `models/runs/papila-grad-cam-cpu/`. Cada variante requiere que su carpeta de salida aún no exista. `models/runs/` está excluida de Git. La medición por imagen incluye preprocesamiento, cálculo, interpolación y transferencia del mapa a CPU. La decodificación JPEG y el guardado de archivos se realizan aparte. El pico de CUDA registra memoria asignada por PyTorch, incluyendo el modelo residente.

Para una integración futura, `src.explain.explain_image(model, image, checkpoint, threshold=0.35)` acepta una imagen Pillow y el modelo en `eval()` cargado por `load_checkpoint`. Devuelve el puntaje académico sin calibrar, la clase, la caja del contenido y un tensor FP32 en CPU indexado como `(y, x)` en la imagen original. Ejecuta esta función fuera de `torch.inference_mode()`. `src.explain_visualization.overlay_png(image, result.heatmap, alpha=0.55)` devuelve los bytes PNG a resolución nativa.

`summarize_regions` acepta máscaras booleanas alineadas y devuelve la fracción de área, la fracción de masa Grad-CAM y su cociente. El informe incluye dos referencias geométricas para orientar la inspección de artefactos, la franja exterior del 10 % por eje y los píxeles originales con máximo RGB ≤ 20. Para evaluar disco, vasos o artefactos específicos hacen falta máscaras verificadas. PAPILA publica contornos de disco y copa de dos expertos como coordenadas de la imagen original, enlazables por ID y ojo. Para una comparación posterior se pueden rasterizar por experto y revisar visualmente su registro con cada JPEG antes de medir solapamiento. [Artículo de PAPILA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184612/)

La cuadrícula de `7 × 7` solo permite interpretar patrones espaciales gruesos. Las figuras muestran evidencia positiva relativa para el logit de glaucoma incluso en predicciones negativas. Cuatro ejemplos permiten formular hipótesis visuales. Las superposiciones por sí solas carecen de alcance causal y diagnóstico.

## Instalación y uso en Windows

La demo usa React 19 compilado con Vite 8, FastAPI sobre Uvicorn y PyTorch/torchvision para inferencia y Grad-CAM. FastAPI sirve la interfaz, sus recursos estáticos y `/api/*` desde `127.0.0.1`, en un solo proceso. El navegador envía la imagen a `/api/analyze` en el mismo origen. Nunito se sirve desde el build local y la política de contenido de la interfaz restringe scripts, estilos y fuentes al mismo origen. La demo arranca con el modelo cargado y verificado, y conserva las rutas API sin prefijo para clientes anteriores.

### Instalación inicial

Usa Windows de 64 bits, Python 3.14 y Node.js 24 para compilar. La demo funciona en CPU. Para CUDA, usa una GPU NVIDIA con controlador compatible y las ruedas `cu130`. La configuración se verificó con Intel Core i5-12450H y NVIDIA RTX 4050 Laptop de 6 GB. Node.js se usa para compilar el frontend. [Ruedas oficiales de PyTorch 2.14.1](https://pytorch.org/get-started/previous-versions/#v2141)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
```

Para una instalación solo CPU, usa `https://download.pytorch.org/whl/cpu` como índice en el comando de PyTorch. El repositorio incluye el checkpoint de la demo en `backend/predict/best.pt`. Su SHA-256 es `721699061b8f0489b77d3f2ce19bccd0d6468a396fc315d1d112ef41c154b761`. Para preparar otra máquina, clona el repositorio e instala las dependencias.

### Arranque con un comando

```powershell
.\.venv\Scripts\python.exe -m backend.demo
```

Desde la raíz, ejecuta el comando y abre [GlaucoAI](http://127.0.0.1:8000/) cuando Uvicorn anuncie la dirección. El comando requiere el build del frontend y verifica el checkpoint antes de abrir el puerto. Añade `--cpu` para forzar CPU o `--port 8001` para elegir otro puerto local. El proceso usa las dependencias ya instaladas. Para detenerlo, pulsa `Ctrl+C`.

| Ruta de FastAPI | Respuesta |
|---|---|
| `GET /api/health` | Estado del modelo, dispositivo y umbral. En desarrollo devuelve `503` si el modelo no pudo cargarse. |
| `POST /api/predict` | Puntaje experimental en `[0, 1]`, etiqueta, clase y umbral `0.35`. |
| `POST /api/explain` | PNG del mapa y de la superposición en Base64, capa objetivo e indicador de señal. |
| `POST /api/analyze` | Predicción y explicación con una sola pasada del modelo. |

Los POST reciben bytes con `Content-Type: image/jpeg` o `image/png`. El límite es 10 MiB, 6 megapíxeles, 4096 píxeles por lado y 15 segundos para recibir el cuerpo. Se valida la decodificación y se normaliza la orientación EXIF. Hasta dos cargas pueden estar en procesamiento o espera con el cuerpo en memoria. Un solo hilo ejecuta el modelo, aísla sus hooks y reutiliza la memoria temporal de CUDA. Los PNG de respuesta tienen hasta 1024 píxeles en su lado más largo. Las imágenes se procesan en memoria, el registro de acceso se deshabilita en la demo y las respuestas de análisis usan `Cache-Control: no-store`.

### Desarrollo y verificación

En desarrollo, ejecuta los siguientes comandos en terminales separadas. Vite reenvía `/api/` al backend local. Usa `npm.cmd --prefix frontend run build` para actualizar el build distribuible después de cambiar el frontend.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --strictPort
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
```

Para reproducir la medición, usa el manifiesto congelado y una copia local de PAPILA. Añade `--cpu` para medir en CPU.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_demo --dataset-dir path/to/PAPILA --manifest path/to/manifest-c.csv --image-id RET013OD --image-id RET033OD --samples 12
```

El script verifica el manifiesto y los pesos. Mide la API ASGI en proceso después de calentar el modelo, salvo la carga inicial. Las cifras de respuesta incluyen decodificación y serialización. Los promedios corresponden a 12 solicitudes por fase en el equipo de referencia.

| Dispositivo | Carga inicial | Inferencia FP32 | Grad-CAM FP32 | Respuesta `/api/analyze` | Respuesta con dos clientes |
|---|---:|---:|---:|---:|---:|
| CPU, 8 hilos | 152 ms | 35 ms | 51 ms | 282 ms | 541 ms |
| RTX 4050 Laptop | 731 ms | 19 ms | 29 ms | 243 ms | 468 ms |

En CUDA, el pico fue de 106,46 MiB asignados y 132 MiB reservados. Después de las tandas consecutiva y concurrente quedaron 58,94 MiB asignados, igual que tras el calentamiento. Los 132 MiB reservados incluyen la caché del asignador de PyTorch.

Auditoría npm del 8 de noviembre de 2026. Producción reporta cero hallazgos. Desarrollo reporta cuatro entradas moderadas asociadas al mismo aviso de `minimatch` 10.2.6. `eslint` y `@eslint/config-array` requieren la rama 10 y `@eslint/js` tiene a `eslint` como par opcional. El parche publicado, 11.0.1, queda fuera de esos rangos. Queda pendiente una versión compatible de la cadena de lint. [Aviso GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249)

## Docker, Vercel y Render

### Contenedor local

Con Docker Desktop en modo de contenedores Linux, ejecuta desde la raíz en PowerShell:

```powershell
docker build --platform linux/amd64 -t glaucoai:local .
docker run --rm -p 127.0.0.1:8000:8000 glaucoai:local
```

Abre [GlaucoAI](http://127.0.0.1:8000/). El contenedor compila React, sirve la aplicación y ejecuta PyTorch en CPU. La imagen incluye el checkpoint de la demo y verifica su SHA-256 al arrancar.

### Backend en Render

1. Crea un Blueprint con `render.yaml` en la rama del repositorio que contenga estos archivos. El servicio `glaucoai-api` usa Docker en CPU, plan gratuito `free` (0,1 CPU y 512 MB de RAM) y verificación en `/api/health`. [Planes de Render](https://render.com/docs/compute-plans)
2. Render construye la imagen con `backend/predict/best.pt` incluido. Para desplegar un commit nuevo, inicia el despliegue manualmente desde el servicio.
3. Abre `/api/health` en la URL HTTPS de Render y espera a que devuelva JSON con `model_ready: true`. Render Free suspende el servicio tras 15 minutos de inactividad y el siguiente arranque en frío puede tardar más de un minuto. [Límites del plan gratuito](https://render.com/docs/free)

### Frontend en Vercel

1. Importa el mismo repositorio y selecciona Node.js 24. Puedes usar **`frontend`** como **Root Directory** o dejar la raíz del repositorio. En ambos casos, `vercel.json` instala las dependencias de compilación de Vite y publica el contenido de `frontend/dist`.
2. En **Settings → Environment Variables**, configura `VITE_API_URL` con el origen HTTPS de Render, por ejemplo `https://glaucoai-api.onrender.com`. Selecciona **Production** y vuelve a desplegar si cambias el valor. [Variables de entorno en Vercel](https://vercel.com/docs/environment-variables/managing-environment-variables)
3. En **Deployment Protection**, verifica que el dominio de producción sea público y comparte esa URL. Prueba la demo con imágenes sintéticas. Usa datos médicos únicamente tras revisar los permisos y el tratamiento de datos en ambos proveedores. [Protección de despliegues en Vercel](https://vercel.com/docs/deployment-protection)

## Experimentos históricos del notebook

### Metodología histórica
1. **Modelo base:** CNN con 2 bloques Conv2D (32 y 64 filtros, 3×3) + ReLU + MaxPool, capa densa de 128 unidades y salida sigmoide. Pérdida BCE y optimizador Adam (lr = 1e-3).
2. **Diagnóstico:** el modelo base presenta sobreajuste (la pérdida de entrenamiento baja mientras la de validación se mantiene fluctuante).
3. **Barridos de hiperparámetros**, uno a la vez:
   - Regularización L2 (weight decay): {0, 1e-5, 1e-4, 1e-3, 1e-2}
   - Dropout: {0.0, 0.2, 0.3, 0.5, 0.7}
   - Data augmentation: Sin / Leve / Moderada / Fuerte
   - Early stopping (paciencia): {3, 5, 10}
   - Learning rate y optimizador (SGD+momentum, RMSprop, Adam), con un barrido de lr propio para cada uno
4. **Diseño factorial completo:** 4 factores binarios × 3 optimizadores = 48 configuraciones.
5. **Selección por Val Loss.** El conjunto de test se usa solo para el reporte final.

### Resultados históricos
Métricas sobre el conjunto de **test (84 imágenes)**, clase positiva = glaucoma, umbral 0.5:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|---------:|----------:|-------:|---:|--------:|
| Baseline (sin regularización) | 0.798 | 0.500 | 0.529 | 0.514 | 0.777 |
| Mejor learning rate (5e-4)    | 0.774 | 0.450 | 0.529 | 0.487 | **0.841** |
| **Combinado (modelo final)**  | **0.857** | **0.667** | **0.588** | **0.625** | 0.817 |

Matriz de confusión del modelo final: 10 de 17 glaucomas detectados (7 falsos negativos) y 5 falsos positivos.

Se prioriza el **recall**, porque un falso negativo (glaucoma no detectado) es más costoso que un falso positivo.

### CNN final del notebook histórico
- Arquitectura: CNN de la sección *Metodología*, entrada 96 × 96 × 3.
- Optimizador: Adam, lr = 1e-3.
- Regularización: augmentation Leve (flip horizontal, rotación ±10°, jitter de brillo y contraste), early stopping con paciencia 5. Sin L2 ni Dropout.
- Salida: puntaje de glaucoma en [0, 1] y clase según umbral.

## Estructura del repositorio
```
├── notebooks/        # experimentos (barridos, factorial, métricas)
├── src/              # datos, modelo, entrenamiento y evaluación
├── models/           # checkpoints y resultados locales
├── configs/          # protocolo y hashes congelados
├── backend/          # FastAPI, servicio del build e inicio local
├── frontend/         # aplicación React y build generado por Vite
├── scripts/          # medición reproducible de la demo
├── tests/            # pruebas de modelos, datos e integración
└── README.md
```

## Limitaciones
- El conjunto de test tiene solo 84 imágenes (17 con glaucoma). Una diferencia de un caso cambia el recall aproximadamente 6 puntos, así que las comparaciones entre modelos son orientativas.
- Un único split. No se usó validación cruzada.
- La CNN histórica usa 96 × 96 px. ResNet18 usa 224 × 224 px y su Grad-CAM parte de una cuadrícula de 7 × 7.
- Los puntajes son experimentales, pueden producir falsos positivos y negativos, y carecen de calibración y validación clínica. Los ejemplos Grad-CAM son ilustrativos.

## Licencia y créditos
PAPILA se cita como Kovalyk et al., *Scientific Data* 9, 291 (2022). Figshare indica GPL 3.0+ para su versión 2. La fuente Nunito se distribuye con su aviso SIL OFL 1.1. [Artículo de PAPILA](https://doi.org/10.1038/s41597-022-01388-1) · [Depósito y licencia de PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004) · [Licencia de Nunito](frontend/public/OFL-Nunito.txt)
