# Optimización de GlaucoAI en la RTX 4050

## Implementación de ResNet34

GlaucoAI usa como modelo principal el finalista 3 de la campaña experimental, `resnet34-lr-low-s43`. La decisión de implementación prioriza los 13 de 17 glaucomas detectados con cinco falsos positivos en la comparación interna. La versión anterior, ResNet18, detectó 6 de 17 y tuvo cuatro falsos positivos. La decisión se tomó después de revisar el test exploratorio. La selección por validación de la campaña y sus resultados se conservan en el informe original. [Campaña experimental final](campana-final-papila.md#evaluación-interna-con-selección-bloqueada).

El archivo servido es [`backend/predict/resnet34.pt`](../backend/predict/resnet34.pt), copiado byte a byte de `models/runs/papila-final-20261110/confirmation/resnet34-lr-low-s43/fit/best.pt`. Ambos archivos tienen SHA-256 `7ad8935d2aeb82b5e95f849e2659259e90a11f71ae58cbe8dce5f2e0d2813f49`. Los pesos y buffers de BatchNorm corresponden al checkpoint original de época 4. El umbral **`0.48704302310943604`** quedó fijado exclusivamente en validación antes de evaluar el test. El servidor clasifica como glaucoma cuando `sigmoid(logit) >= umbral`. [Configuración de servicio](../configs/glaucoai-models.json).

| Configuración original | Valor verificado |
|---|---|
| Arquitectura | ResNet34 de Torchvision con bloques 3–4–6–3, cabeza lineal de 512 entradas y un logit, dropout 0,0. |
| Inicialización | `ResNet34_Weights.IMAGENET1K_V1`. |
| Ajuste fino | Cabeza y `layer4`, con BatchNorm activa en `layer4` durante el entrenamiento. `tune_blocks=1`, `tune_last_block=false`, `freeze_batch_norm=false`, `progressive_head_epochs=0`. |
| Optimizador | Adam a `0.0001` en todos los parámetros entrenables. `backbone_learning_rate=0.0` indica tasa común. Planificador constante, `weight_decay=0.0`. El campo `momentum=0.9` se registra en la configuración y se aplica solo al elegir SGD. |
| Pérdida | BCE con logits y peso positivo `3.807017543859649` (217/57). `focal_loss=false`, `label_smoothing=0.0`. |
| Datos | PAPILA binario con separación por paciente, 274 ojos de entrenamiento y 62 de validación. Semilla 43, lote y lote efectivo 16, `accumulation_steps=1`, cero workers, caché activada y workers persistentes desactivados. |
| Entrada | RGB, letterbox bilineal de 224 × 224 con relleno negro centrado. Escala [0, 1], media ImageNet (0,485, 0,456, 0,406), desviación (0,229, 0,224, 0,225). Solo volteo horizontal durante entrenamiento. |
| Precisión y dispositivo | CUDA en la RTX 4050 Laptop GPU, FP32, sin AMP, memoria de lotes fijada. PyTorch 2.14.1+cu130 y Torchvision 0.29.1+cu130. |
| Parada y checkpoint | Máximo de 15 épocas, paciencia 5, 9 épocas ejecutadas. Época 4 por mínima BCE sin ponderar de validación (0,2927056474070395). El umbral 0,5 del bloque de entrenamiento se usó para las métricas por época. |
| Parámetros | 21.285.185 totales, 13.114.881 entrenables durante la corrida. En servicio, todos los módulos están en modo evaluación. |

| Partición y modelo | Umbral | VP / FN | VN / FP | Sensibilidad | Especificidad | F2 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validación · ResNet34 | 0,48704302310943604 | 11 / 2 | 46 / 3 | 84,6 % | 93,9 % | 0,833 | 0,898 |
| Validación · ResNet18 de respaldo | 0,35 | 9 / 4 | 46 / 3 | 69,2 % | 93,9 % | 0,703 | 0,859 |
| Test interno · ResNet34 | 0,48704302310943604 | 13 / 4 | 62 / 5 | 76,5 % | 92,5 % | 0,756 | 0,873 |
| Test interno · ResNet18 de respaldo | 0,35 | 6 / 11 | 63 / 4 | 35,3 % | 94,0 % | 0,385 | 0,795 |

El test contiene 84 ojos de 42 pacientes y se había consultado en campañas previas. La sensibilidad de ResNet34 quedó por debajo del objetivo exploratorio de 85 %. Su intervalo de confianza exploratorio al 95 % por paciente fue de 50,0 % a 100 %, dada la cantidad de casos. Los resultados son internos y los puntajes de glaucoma carecen de calibración clínica. La confirmación requiere una cohorte independiente.

FastAPI restaura la ResNet34 de Torchvision con su cabeza de un logit y carga estrictamente los tensores originales. `/api/health`, `/api/predict` y `/api/analyze` identifican el modelo cargado. La respuesta de análisis incluye Grad-CAM del logit de glaucoma sobre `backbone.layer4[-1]`, alineado con el campo de la imagen al deshacer el letterbox. React muestra la arquitectura y el umbral de la respuesta recibida.

En las comprobaciones locales, `/api/predict` reprodujo las 62 decisiones archivadas de validación en CPU y en la RTX 4050. La diferencia máxima de puntaje frente al archivo fue de aproximadamente `2.56e-6` en CPU y `5.96e-8` en CUDA. `/api/analyze` y `/api/explain` generaron PNG válidos para un ojo sano y uno con glaucoma en ambos dispositivos. Chrome comprobó el flujo de carga y el nombre del modelo, el umbral exacto y las imágenes para ResNet34 y el respaldo. Pasaron 77 pruebas de Python, el lint y el build de React.

Desde la raíz del proyecto y con las dependencias instaladas, compila el frontend e inicia la demo:

```powershell
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m backend.demo
```

Abre [GlaucoAI](http://127.0.0.1:8000/). Añade `--cpu` para forzar inferencia en CPU. Para una predicción individual puedes ejecutar:

```powershell
.\.venv\Scripts\python.exe -m src.predict --checkpoint backend/predict/resnet34.pt --image ruta/imagen.jpg --threshold '0.48704302310943604'
```

ResNet18 queda como respaldo experimental en [`backend/predict/best.pt`](../backend/predict/best.pt), con SHA-256 `721699061b8f0489b77d3f2ce19bccd0d6468a396fc315d1d112ef41c154b761` y umbral `0.35`. Detén la demo y define `$env:GLAUCOAI_MODEL = 'resnet18'` antes de reiniciar para usar el respaldo. Quita la variable con `Remove-Item Env:GLAUCOAI_MODEL` antes de volver a iniciar ResNet34.

Las secciones siguientes documentan los experimentos históricos anteriores a la implementación.

## Fundamentos de clase

Los nombres entre paréntesis corresponden a los archivos WebVTT guardados localmente en `transcripciones/`.

| Archivo | Minuto aproximado | Aplicación al proyecto |
| --- | --- | --- |
| (3) | 00:32–00:50, 01:08–01:12 | Convoluciones, filtros, max pooling y dropout. La tasa de aprendizaje controla el tamaño del paso. El lote determina cuántos ejemplos contribuyen al gradiente. |
| (2) | 00:13–00:15, 00:37–00:55, 01:22–01:24 | Test fijo, CNN en PyTorch, entropía cruzada, optimizador y tasa de aprendizaje, envío de los lotes al dispositivo y aumento de datos. |
| (1) | 00:42–00:58, 01:14–01:19 | Transferencia desde un modelo preentrenado. El profesor muestra Inception, menciona VGG16 y ResNet50, y descongela bloques seleccionados manteniendo BatchNorm congelada. Se aplica esa técnica al último bloque de la ResNet18 existente. |
| (4) | 00:38–00:40, 01:04–01:11, 01:33–01:35 | Pérdida y gradientes, ajuste de lote y tasa de aprendizaje, y diagnóstico de sobreajuste o subajuste con curvas de entrenamiento y validación. |
| Sin número | 00:19–00:24 | YOLO para detección con cajas, junto a SAM para segmentación en la demostración. El repositorio carece de un clasificador YOLO y PAPILA se usa aquí con etiquetas por ojo. |

La pérdida binaria con logits es la variante ya usada por el proyecto para sus dos clases. Adam es el optimizador del entrenador reproducible. SGD con momentum, RMSprop, L2 y parada temprana aparecen en el notebook histórico. Las transcripciones permiten atribuir al curso el papel del optimizador, la entropía cruzada y el seguimiento de pérdida y aciertos. Para este proyecto también se usan sensibilidad, especificidad, F1, F2, ROC-AUC y AP, cuya enseñanza no pude confirmar en los subtítulos.

## Datos y referencias

PAPILA tiene 488 retinografías de 244 pacientes en el inventario de origen. Tras excluir ojos sospechosos, quedan 420 imágenes de 210 pacientes. Se verificaron los hashes del manifiesto y de cada inventario contra la configuración congelada. El lector comprobó que los dos ojos del mismo paciente quedan en un mismo conjunto. La comparación SHA-256 de las 420 imágenes encontró cero duplicados exactos. La auditoría de bytes no permite detectar versiones recortadas o reexportadas de una misma imagen.

| Conjunto | Pacientes | Ojos sanos | Ojos con glaucoma |
| --- | ---: | ---: | ---: |
| Entrenamiento | 137 | 217 | 57 |
| Validación | 31 | 49 | 13 |
| Prueba interna | 42 | 67 | 17 |

El notebook histórico usa otra asignación de pacientes, por lo que sus métricas sirven como antecedente. Las referencias comparables son los checkpoints congelados de la CNN y de ResNet18. La CNN tiene dos bloques convolucionales de 32 y 64 canales, max pooling y una capa densa de 128 unidades. Usa RGB a 96 × 96 y normalización calculada solo en entrenamiento. La ResNet18 usa bloques residuales y una cabeza lineal de un logit. Conserva la proporción al ajustar a 224 × 224 con bandas negras y normaliza con estadísticas de ImageNet. Ambas referencias usan Adam y volteo horizontal solo en entrenamiento.

En prueba interna, la CNN detectó 8 de 17 ojos con glaucoma a umbral 0,50, mientras que la ResNet18 detectó 6 de 17 a umbral 0,35. La ResNet18 tuvo mayor ROC-AUC (0,795 frente a 0,715) y especificidad (94,0 % frente a 88,1 %). El contraste muestra por qué la calidad del ordenamiento y el desempeño a un umbral concreto deben reportarse por separado.

## Protocolo de comparación

- Se usa el manifiesto congelado con semilla de partición 42. El entrenamiento lee únicamente train y validación. Se registran hardware, configuración y hash de cada checkpoint.
- Se conserva el checkpoint de menor pérdida de validación de cada corrida. Para comparar checkpoints se prueban umbrales de 0 a 1 en pasos de 0,05. Se maximiza sensibilidad con especificidad de al menos 80 %. Los desempates favorecen mayor especificidad y luego mayor umbral.
- La comparación de candidatos prioriza sensibilidad y F2 frente a la referencia evaluada con la misma regla. ROC-AUC y AP complementan las métricas al umbral. Con 13 positivos en validación, un solo ojo equivale a 7,7 puntos de sensibilidad. La prueba interna se reserva para el candidato ya elegido y su umbral fijado.
- Los pesos y resultados anteriores se preservan. Las métricas son exploratorias porque el test de PAPILA se ha consultado históricamente. El puntaje de la app es académico y sin calibrar.

La ronda inicial usó la pérdida binaria existente. El seguimiento con autorización para probar técnicas adicionales se documenta al final.

[Publicación de PAPILA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184612/) · [Documentación de precisión mixta de PyTorch](https://docs.pytorch.org/docs/2.15/amp.html)

## Entorno y pilotos

Windows detectó una NVIDIA GeForce RTX 4050 Laptop GPU con 6141 MiB y controlador 621.01. El Python global tiene PyTorch `2.14.0+cpu`. La ejecución se hizo desde esta carpeta con el intérprete de la otra copia local del repositorio, `C:\Users\nesto\Proyectos\DeepLearning\.venv\Scripts\python.exe`, que tiene PyTorch `2.14.1+cu130` y torchvision `0.29.1+cu130`. Una operación de tensor en `cuda:0` confirmó el acceso real a la GPU. CUDA 13.0 corresponde a la compilación de PyTorch. El controlador reporta compatibilidad CUDA 13.5.

| Piloto ResNet18, 2 épocas | Lote | Workers | Precisión | s/época | Pasos omitidos | VRAM reservada por PyTorch |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Cabeza | 16 | 0 | AMP | 13,83 | 2 | 316 MiB |
| Cabeza | 16 | 2 | AMP | 18,84 | 2 | 316 MiB |
| Cabeza | 32 | 2 | AMP | 17,97 | 1 | 508 MiB |
| Cabeza | 16 | 2 | FP32 | 16,87 | 0 | 268 MiB |
| Último bloque | 16 | 2 | AMP | 18,30 | 2 | 450 MiB |

Los tiempos abarcan el entrenamiento y la validación de cada época. La comparación con dos workers favoreció FP32 y el piloto con cero workers favoreció un cargador sin procesos secundarios. El tamaño de lote se mantuvo en 16. En CUDA, el cargador fija memoria y usa transferencias no bloqueantes. Se dejaron dos hilos para OpenMP y MKL en las corridas de esta campaña.

## Reproducción

Desde la raíz del repositorio en PowerShell, ajusta `$python` y `$dataset` a tu entorno. La raíz del dataset debe contener las imágenes de la copia fijada de PAPILA y el manifiesto congelado. Usa una carpeta de salida nueva por corrida.

```powershell
$python = 'C:/Users/nesto/Proyectos/DeepLearning/.venv/Scripts/python.exe'
$dataset = (Resolve-Path 'data/papila').Path
$campaign = 'models/runs/papila-opt-repro'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$common = @('--dataset-dir', $dataset, '--manifest', "$dataset/manifest-c.csv")
$training = $common + @('--epochs', '15', '--no-amp', '--seed', '42')
& $python -m src.train @training --architecture resnet18 --learning-rate 0.0003 --tune-last-block --run-dir "$campaign/resnet-layer4"
& $python -m src.train @training --architecture papila_cnn --dropout 0.3 --run-dir "$campaign/cnn-dropout"
& $python -m src.train @training --architecture resnet18 --learning-rate 0.0003 --run-dir "$campaign/resnet-head"
$reference = 'C:/Users/nesto/Proyectos/DeepLearning/models/runs'
& $python -m src.train @common --architecture resnet18 --epochs 10 --no-amp --tune-last-block `
    --learning-rate 0.00001 --init-checkpoint "$reference/papila-resnet18/best.pt" --run-dir "$campaign/resnet-layer4-desde-cabeza"
& $python -m scripts.evaluate_optimization @common --campaign $campaign `
    --run CNN "$reference/papila-reference" --run ResNet18 "$reference/papila-resnet18" `
    --run CNN-dropout "$campaign/cnn-dropout" --run ResNet18-cabeza "$campaign/resnet-head" `
    --run ResNet18-layer4 "$campaign/resnet-layer4" `
    --run ResNet18-desde-cabeza "$campaign/resnet-layer4-desde-cabeza"
```

El evaluador guarda `validation.json` con los pesos identificados por SHA-256, los umbrales y el candidato seleccionado. Un candidato tiene que superar a la ResNet18 de referencia en sensibilidad y F2 aplicando la misma regla de validación. Con un candidato fijado, `--test` lee su configuración de validación, reserva una carpeta de salida y evalúa el test congelado una vez junto a las referencias. Los tiempos de inferencia de este evaluador miden el pase del modelo en CUDA, con lote de uno, sin decodificación, subida HTTP ni Grad-CAM.

```powershell
& $python -m scripts.evaluate_optimization @common --campaign $campaign --test
```

## Resultados del 9 de noviembre de 2026

Las cuatro corridas nuevas usaron Adam, BCE con logits, lote 16, cero workers, semilla 42, volteo horizontal y FP32. El entrenamiento y la validación se ejecutaron en CUDA. La memoria reservada de la tabla corresponde a PyTorch. Los archivos de telemetría de la campaña registraron picos de 100 % de utilización, 617 MiB de uso global y 59,4 °C al incluir los pilotos. La utilización media de las muestras válidas durante las cuatro corridas fue 11,3 %. El patrón de ráfagas es compatible con esperas de la carga de datos. Haría falta un perfil para identificar cuánto tiempo se dedica a decodificar JPEG.

| Corrida | Tasa | Épocas ejecutadas | Época elegida | Pérdida de validación | Tiempo total | s/época | VRAM reservada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN con dropout 0,30 | 0,001 | 15 | 11 | 0,4502 | 185,4 s | 11,72 | 286 MiB |
| ResNet18, cabeza congelando el backbone | 0,0003 | 15 | 15 | 0,4103 | 198,7 s | 13,13 | 268 MiB |
| ResNet18, cabeza y `layer4` desde ImageNet | 0,0003 | 11 | 6 | 0,3894 | 142,1 s | 12,81 | 422 MiB |
| ResNet18, `layer4` desde la cabeza entrenada | 0,00001 | 7 | 2 | 0,3519 | 90,4 s | 12,77 | 422 MiB |

La cabeza sola ajustó 513 parámetros. Al descongelar `layer4` se ajustaron 8 389 121 parámetros, conservando fijos los parámetros y estadísticas de BatchNorm. La variante desde la cabeza entrenada mostró sobreajuste temprano. Su pérdida de entrenamiento siguió bajando hasta 0,2316 y la de validación volvió a subir después de la época 2. La variante desde ImageNet se detuvo en la época 11 tras cinco épocas sin mejora. La cabeza sola seguía reduciendo la pérdida al llegar al límite de 15 épocas.

### Validación con el mismo procedimiento de umbral

| Modelo | Umbral | Exactitud | Precisión | Sensibilidad | Especificidad | F1 | F2 | ROC-AUC | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN original | 0,55 | 82,3 % | 62,5 % | 38,5 % | 93,9 % | 0,476 | 0,417 | 0,739 | 8 |
| CNN con dropout | 0,35 | 82,3 % | 58,3 % | 53,8 % | 89,8 % | 0,560 | 0,547 | 0,739 | 6 |
| ResNet18 original | 0,35 | 88,7 % | 75,0 % | 69,2 % | 93,9 % | 0,720 | 0,703 | 0,859 | 4 |
| ResNet18, cabeza a 0,0003 | 0,25 | 82,3 % | 56,3 % | 69,2 % | 85,7 % | 0,621 | 0,662 | 0,879 | 4 |
| **ResNet18, `layer4` desde ImageNet** | **0,10** | **82,3 %** | **55,0 %** | **84,6 %** | **81,6 %** | **0,667** | **0,764** | **0,846** | **2** |
| ResNet18, `layer4` desde la cabeza entrenada | 0,25 | 82,3 % | 56,3 % | 69,2 % | 85,7 % | 0,621 | 0,662 | 0,841 | 4 |

La variante de `layer4` desde ImageNet fue la única que mejoró tanto sensibilidad como F2 frente a la ResNet18 original con la misma regla de umbral. El checkpoint de la época 6 y el umbral 0,10 quedaron fijados antes de puntuar el test.

### Prueba interna con el umbral fijado

| Modelo | Umbral | Exactitud | Precisión | Sensibilidad | Especificidad | F1 | F2 | ROC-AUC | VP / FN | VN / FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN original | 0,50 | 79,8 % | 50,0 % | 47,1 % | 88,1 % | 0,485 | 0,476 | 0,716 | 8 / 9 | 59 / 8 |
| ResNet18 original | 0,35 | 82,1 % | 60,0 % | 35,3 % | 94,0 % | 0,444 | 0,385 | 0,795 | 6 / 11 | 63 / 4 |
| ResNet18 con ajuste fino | 0,10 | 73,8 % | 41,4 % | 70,6 % | 74,6 % | 0,522 | 0,619 | 0,834 | 12 / 5 | 50 / 17 |

El candidato detectó seis casos adicionales y produjo trece falsos positivos adicionales frente a la demo. Su especificidad de 74,6 % quedó por debajo del mínimo operativo exploratorio de 80 % y en esa etapa se conservó la ResNet18 original. El candidato queda disponible como artefacto experimental. Su mejor ordenamiento en este test interno y su mayor sensibilidad requieren confirmación independiente. La campaña usa una semilla de entrenamiento y un test con 17 positivos y dos ojos correlacionados por paciente.

La pasada de inferencia CUDA de lote uno promedió 0,34 ms para la CNN, 2,91 ms para la ResNet18 original y 4,39 ms para el candidato durante la prueba interna. Son mediciones puntuales del pase del modelo. Las dos ResNet18 tienen la misma arquitectura de inferencia y estos tiempos no aíslan el efecto de las frecuencias de la GPU ni del orden de medición.

Los pesos, curvas de pérdida, matrices de confusión, curvas ROC y precisión–recall, predicciones por ojo y telemetría quedan en `models/runs/papila-opt-20261109/`. Los resultados y la copia local verificada de PAPILA en `data/papila/` están excluidos de Git.

### Compatibilidad con la demo

La prueba local cargó el candidato en FastAPI mediante una especificación temporal con su SHA-256 y umbral. Con dos retinografías de validación se comprobó el frontend compilado, el estado del modelo, la igualdad entre los puntajes de predicción y análisis, y la validez de los dos PNG de Grad-CAM. El benchmark local en CPU con un hilo y 12 solicitudes por fase registró 93,45 ms de respuesta media de predicción para la demo y 94,63 ms para el candidato. La respuesta media de análisis con Grad-CAM fue de 281,43 y 268,17 ms, respectivamente. El archivo `backend-cpu.json` conserva los resultados. Esta medición usa el equipo local. Render sirve inferencia en CPU y Vercel sirve la interfaz React.

Pasaron las 66 pruebas de Python, el lint de React y la compilación de Vite. Se verificó el SHA-256 del checkpoint de la demo al finalizar.

## Seguimiento con funciones de pérdida adicionales

Se probaron dos objetivos de entrenamiento fuera de las técnicas identificadas en los subtítulos. Se conservó como control la corrida `resnet-layer4` desde ImageNet, con el mismo manifiesto y los mismos JPEG verificados, semilla 42, lote 16, FP32, Adam a 0,0003, volteo horizontal, hasta 15 épocas y paciencia de cinco. Solo cambió la función de pérdida de entrenamiento. El checkpoint se eligió por BCE sin ponderar en validación, también para las variantes. Las pérdidas de entrenamiento y validación pueden tener escalas distintas.

- **BCE ponderada.** El término de cada glaucoma real pesa 217/57, aproximadamente 3,81 veces el término de un ojo sano. El cociente se calculó exclusivamente en entrenamiento. Esto cambia el gradiente y puede desplazar los puntajes. La selección del umbral es una decisión posterior. [BCEWithLogitsLoss en PyTorch](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html)
- **Focal loss.** Con `gamma=2` y sin ponderación adicional por clase, el factor `(1 − p_t)^2` reduce el peso de ejemplos ya bien clasificados. `p_t` es el puntaje asignado a la etiqueta verdadera. Esto cambia el peso según la dificultad de cada ojo, independientemente de su clase. [Artículo original de focal loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

La comparación incluye la CNN y la ResNet18 de la demo como referencias, además del control directo con `layer4`. Para todos los modelos se reaplica el umbral de validación con especificidad de al menos 80 %. El control ya tiene una evaluación interna archivada. Solo se reabre el test si una variante nueva se selecciona por encima del control con la misma regla. El test seguiría siendo exploratorio por sus consultas previas.

```powershell
$campaign = 'models/runs/papila-loss-repro'
$lossTrain = $common + @('--architecture', 'resnet18', '--tune-last-block', '--learning-rate', '0.0003', '--epochs', '15', '--no-amp')
& $python -m src.train @lossTrain --positive-weight 3.807017543859649 --run-dir "$campaign/bce-ponderada"
& $python -m src.train @lossTrain --focal-loss --run-dir "$campaign/focal"
& $python -m scripts.evaluate_optimization @common --campaign $campaign `
    --run CNN "$reference/papila-reference" --run ResNet18 "$reference/papila-resnet18" `
    --run Control-layer4 'models/runs/papila-opt-20261109/resnet-layer4' `
    --run BCE-ponderada "$campaign/bce-ponderada" --run Focal "$campaign/focal"
```

### Resultados de las pérdidas adicionales

| Variante | Épocas / mejor época | Tiempo total en CUDA | Umbral de validación | VP / FN | FP | Especificidad | F2 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control `layer4`, BCE | 11 / 6 | 142,12 s | 0,10 | 11 / 2 | 9 | 81,6 % | 0,764 | 0,846 |
| BCE ponderada | 15 / 12 | 190,98 s | 0,40 | 11 / 2 | 7 | 85,7 % | 0,786 | 0,843 |
| Focal loss | 11 / 6 | 146,61 s | 0,35 | 7 / 6 | 6 | 87,8 % | 0,538 | 0,856 |

Ambas variantes reservaron hasta 422 MiB en PyTorch. La telemetría conjunta registró picos de 99 % de utilización, 530 MiB de uso global y 61,6 °C. BCE ponderada conservó la sensibilidad de validación y redujo dos falsos positivos. Focal loss obtuvo una ROC-AUC algo mayor que el control en validación, pero detectó cuatro casos menos en el punto operativo y quedó fuera del test.

La BCE ponderada se evaluó en el test congelado con el umbral 0,40 fijado en validación. La fila del control con BCE normal corresponde a la evaluación anterior sobre el mismo manifiesto.

| Modelo | Umbral | Exactitud | Precisión | Sensibilidad | Especificidad | F1 | F2 | ROC-AUC | VP / FN | VN / FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 de la demo | 0,35 | 82,1 % | 60,0 % | 35,3 % | 94,0 % | 0,444 | 0,385 | 0,795 | 6 / 11 | 63 / 4 |
| Control `layer4`, BCE | 0,10 | 73,8 % | 41,4 % | 70,6 % | 74,6 % | 0,522 | 0,619 | 0,834 | 12 / 5 | 50 / 17 |
| `layer4` con BCE ponderada | 0,40 | 82,1 % | 55,6 % | 58,8 % | 88,1 % | 0,571 | 0,581 | 0,836 | 10 / 7 | 59 / 8 |

La BCE ponderada superó el mínimo exploratorio de 80 % de especificidad en el test. Frente al control con BCE normal, produjo nueve falsos positivos menos a cambio de dos casos adicionales de glaucoma sin detectar. Frente a la demo detectó cuatro casos adicionales y produjo cuatro falsos positivos más. La diferencia de ROC-AUC frente al control fue de 0,002. Los resultados corresponden a una semilla y a un test interno consultado en rondas anteriores, con 17 positivos.

Los checkpoints, métricas, curvas y telemetría de esta ronda están en `models/runs/papila-loss-20261109/`. El checkpoint de BCE ponderada queda como candidato experimental con umbral 0,40. La demo conserva su checkpoint publicado.

## Barrido del peso positivo con tres semillas

El barrido compara BCE normal y pesos positivos de 2,00 y 3,81, con las semillas 42, 43 y 44. Todas las corridas usan ResNet18 con `layer4` ajustable, BatchNorm congelada, Adam a 0,0003, lote 16 y FP32. Se reutilizan dos checkpoints de la semilla 42 y se entrenan siete corridas nuevas. Para cada checkpoint se fija el umbral en validación con el mínimo de 80 % de especificidad. Se comparan la sensibilidad media y su rango entre semillas, junto con especificidad, F2 y ROC-AUC.

Para cada peso se prueba también el promedio de los tres puntajes por ojo y se aplica la misma regla de umbral. El promedio necesita tres modelos en inferencia. Esta comparación exploratoria usa los mismos 62 ojos de validación para elegir el umbral y medir su desempeño. [Artículo sobre ensambles de redes](https://arxiv.org/abs/1612.01474)

Las corridas nuevas usan `--cache-preprocessed` con cero workers. La caché guarda en RAM el tensor tras redimensionar cada imagen. El volteo aleatorio y la normalización se siguen aplicando al acceder al ejemplo. Se verificó igualdad exacta de todos los lotes de entrenamiento y validación con y sin caché durante dos épocas. Los tensores de ambas particiones a 224 × 224 ocupan unos 193 MiB en RAM, más el espacio de trabajo de PyTorch.

```powershell
$campaign = 'models/runs/papila-weights-repro'
$sweepTrain = $common + @('--architecture', 'resnet18', '--tune-last-block', '--learning-rate', '0.0003', '--epochs', '15', '--no-amp', '--cache-preprocessed')
$runs = @(
    '--run', 'CNN', "$reference/papila-reference",
    '--run', 'ResNet18', "$reference/papila-resnet18",
    '--run', 'w1-s42', 'models/runs/papila-opt-20261109/resnet-layer4',
    '--run', 'w3p81-s42', 'models/runs/papila-loss-20261109/bce-ponderada'
)
foreach ($seed in 42,43,44) {
    foreach ($variant in @(@{ Tag='1'; Weight='1' }, @{ Tag='2'; Weight='2' }, @{ Tag='3p81'; Weight='3.80701754385965' })) {
        if ($seed -eq 42 -and $variant.Tag -ne '2') { continue }
        $run = "w$($variant.Tag)-s$seed"
        & $python -m src.train @sweepTrain --seed $seed --positive-weight $variant.Weight --run-dir "$campaign/$run"
        if ($LASTEXITCODE -ne 0) { throw "Falló $run" }
        $runs += @('--run', $run, "$campaign/$run")
    }
}
& $python -m scripts.evaluate_optimization @common --campaign $campaign @runs
& $python -m scripts.summarize_weight_sweep --campaign $campaign
```

| Peso positivo | Sensibilidad media (rango) | Especificidad media | F2 medio | ROC-AUC media | Promedio de tres modelos, VP / FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,00 | 82,1 % (76,9–84,6 %) | 81,6 % | 0,744 | 0,830 | 10 / 8 |
| 2,00 | 76,9 % (61,5–84,6 %) | 84,4 % | 0,715 | 0,844 | 11 / 9 |
| 3,81 | 84,6 % (84,6–84,6 %) | 83,7 % | 0,775 | 0,848 | 11 / 9 |

El peso 3,81 detectó 11 de los 13 ojos con glaucoma en cada semilla y obtuvo el mayor F2 medio. Sus umbrales de validación fueron 0,40, 0,15 y 0,35 para las semillas 42, 43 y 44. El umbral debe acompañar al checkpoint que lo originó. El promedio con peso 3,81 detectó los mismos 11 casos, con nueve falsos positivos frente a los siete del checkpoint de la semilla 42. Este checkpoint sigue siendo el candidato de referencia por su resultado en validación y el test interno archivado. El barrido compara el mismo conjunto de validación en tres entrenamientos. Los resultados no estiman el desempeño en una población nueva.

En una comparación con peso 2,00 y semilla 42, las tres primeras épocas tuvieron métricas idénticas con y sin caché. El tiempo total de entrenamiento y validación de esas épocas bajó de 43,77 a 17,14 s. La primera época llena la caché. El promedio de las épocas 2 y 3 bajó de 14,90 a 0,71 s. Pasaron las 67 pruebas, incluida la que comprueba que el volteo se sigue aplicando en cada acceso.

El resumen, los siete checkpoints nuevos y las predicciones de validación se encuentran en `models/runs/papila-weights-20261109/`. Los dos checkpoints reutilizados conservan sus rutas originales en el resumen.
