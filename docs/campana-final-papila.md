# Campaña experimental final de GlaucoAI

La campaña compara métodos de clasificación por ojo para aumentar la sensibilidad sin perder especificidad. El objetivo exploratorio es alcanzar al menos 85 % en ambas tasas. La recomendación considera estabilidad entre semillas, resultados internos, coste y compatibilidad con la aplicación académica.

**Decisión posterior de implementación.** GlaucoAI usa el finalista 3, ResNet34 a 0,0001, desde el checkpoint y umbral fijados en esta campaña. Las conclusiones siguientes corresponden al protocolo experimental original. [Implementación y respaldo](optimizacion-papila.md#implementación-de-resnet34).

**Recomendación final.** Conservar la ResNet18 publicada como versión académica vigente. EfficientNet-B0 ganó el criterio de validación por un margen estrecho entre grupos y obtuvo 76,5 % de sensibilidad con 67,2 % de especificidad en test interno. La ResNet34 finalista a 0,0001 obtuvo 76,5 % y 92,5 %. Su resultado queda como candidato para una confirmación independiente. Cambiar la selección después de observar ese test introduciría otra decisión guiada por la cohorte ya consultada.

## Auditoría y procedencia

PAPILA contiene 488 retinografías de 244 pacientes. El manifiesto binario congelado excluye 68 imágenes sospechosas y conserva ambos ojos de 210 pacientes. Se verificaron el SHA-256 del manifiesto, los tres inventarios congelados de JPEG, la ausencia de duplicados de bytes y la asignación única de cada paciente. La comparación de hashes detecta duplicados exactos. El dataset público incluye contornos de disco y copa de dos expertos. [Artículo de PAPILA](https://www.nature.com/articles/s41597-022-01388-1) · [Auditoría local](../models/runs/papila-final-20261110/audit.json).

| Partición | Pacientes | Ojos sanos | Ojos con glaucoma |
|---|---:|---:|---:|
| Entrenamiento | 137 | 217 | 57 |
| Validación | 31 | 49 | 13 |
| Test interno | 42 | 67 | 17 |

Los cuatro checkpoints de referencia coincidieron con sus hashes. Se volvieron a calcular matrices, sensibilidad, especificidad y ROC-AUC desde las predicciones históricas guardadas. Se verificaron las 35 corridas históricas y que el piloto FP32 con caché tardó 15,80 s frente a 62,75 s sin caché, con CSV de predicciones idénticos. El checkpoint publicado de ResNet18 tiene SHA-256 `721699061b8f0489b77d3f2ce19bccd0d6468a396fc315d1d112ef41c154b761`. [Verificación de antecedentes](../models/runs/papila-final-20261110/historical-audit.json) · [Campaña anterior](investigacion-papila.md).

Los cargadores de entrenamiento solo construyen train y validación. El preprocesamiento común conserva el aspecto con ajuste RGB a 224 × 224, relleno negro y normalización ImageNet. Las transformaciones aleatorias se aplican únicamente a train y después de la caché. La API conserva su punto de entrada, su checkpoint publicado y su contrato de predicción y Grad-CAM. El candidato se verifica mediante una instancia local con un cargador explícito.

## Plan y selección

- **Presupuesto.** Hasta 64 entrenamientos secuenciales en la RTX 4050 de 6 GB, con un máximo de 15 épocas y paciencia inicial 5. La búsqueda está en [papila-final.json](../configs/papila-final.json). Se conserva configuración, semilla, log, historia, checkpoint, métricas y recursos por intento.
- **Arquitectura y ajuste.** ResNet18, ResNet34 y EfficientNet-B0 parten de pesos oficiales ImageNet verificados por SHA-256. Para cada una se comparan cabeza sola, última etapa, dos etapas, ajuste progresivo con tres épocas de cabeza y BatchNorm activa en la última etapa. En ResNet se usan `layer4` y `layer3`; en EfficientNet, las etapas MBConv finales y la convolución terminal. Se parte de Adam a 0,0003, lote 16, BCE ponderada 217/57 y volteo horizontal.
- **Cribado por etapas.** Pasan dos arquitecturas por el mejor ajuste de validación en semilla 42. En cada una se compara BCE normal, peso 2, peso 217/57 y focal con $\gamma=2$ sin peso de clase. Sobre la mejor pérdida por arquitectura se contrastan tasas fijas y diferenciadas, Adam, AdamW, SGD con momentum, dos planificadores, lotes 8 y 32 y acumulación 16 × 2. El mejor resultado del cribado se usa como ancla de regularización y aumento. Si el ancla usa focal, el suavizado se prueba sobre su mejor referencia BCE.
- **Aumento y regularización.** El control usa volteo horizontal. Se prueban ausencia de volteo, brillo o contraste ±10 %, rotación ±5° con margen para conservar el campo, zoom hacia afuera entre 0,92 y 1,0, dropout 0,20 en la cabeza, penalización 0,001, suavizado 0,05 y paciencia 3. Se descartaron recorte y zoom hacia adentro por el riesgo de suprimir la papila o la retina peripapilar.
- **Checkpoint, umbral y finalistas.** Cada corrida selecciona la menor BCE sin ponderar de validación, en FP32 IEEE. Se examinan los puntos medios entre puntajes distintos y los extremos 0 y 1. El umbral maximiza el mínimo entre sensibilidad y especificidad. Desempata por su media, sensibilidad y umbral. Se prioriza alcanzar ambas metas. El ranking de cribado usa cumplimiento, mínimo, media, F2 y ROC-AUC. Las tres mejores configuraciones se completan con semillas 43 y 44. El ranking de grupos prioriza número de semillas con 85/85, medias del mínimo, exactitud balanceada, F2 y ROC-AUC, dispersión y latencia. El checkpoint operativo es el más cercano a las medianas del mínimo y de la exactitud balanceada en el grupo ganador. La semilla menor resuelve un empate.
- **Separación de resultados.** El archivo de bloqueo fija representantes, controles de arquitectura, candidato previo, referencia ponderada y demo con sus hashes y umbrales antes de puntuar test. Los intervalos percentiles al 95 % se calculan con 4.000 remuestras por paciente, manteniendo ambos ojos y usando los mismos pacientes para cada comparación pareada. En validación se condicionan al modelo y umbral ya elegidos. Las tres semillas varían inicialización, orden y aumentos sobre los mismos pacientes.
- **Hardware.** Se usa CUDA con operaciones deterministas, memoria fijada y transferencia no bloqueante. Las corridas usan cero workers y caché compartida de píxeles preprocesados dentro de la campaña. Se comparan FP32, FP16 y BF16 con la misma receta durante cinco épocas después de fijar la selección de validación. Los pilotos anteriores verificaron caché, pinning y workers persistentes. Se registran uso global de GPU por `nvidia-smi`, picos asignado y reservado por PyTorch, RAM residente del proceso por muestreo, tiempos, parámetros y latencia FP32 de lote 1.

Con 13 positivos en validación, 85 % exige 12/13 aciertos. En test exige 15/17 y al menos 57/67 negativos correctos. Precisión es $VP/(VP+FP)$, sensibilidad es $VP/(VP+FN)$ y especificidad es $VN/(VN+FP)$. F2 da mayor peso a la sensibilidad. ROC-AUC y AP usan los logits y se informan aparte del punto operativo. Los puntajes sigmoides son experimentales y carecen de calibración.

## Cribado de validación

Los tres bloques siguientes usan semilla 42. Cada umbral se obtuvo con la misma regla de validación. Los CSV y JSON conservan la configuración y todas las métricas, incluso en las corridas descartadas. [Experimentos CSV](../models/runs/papila-final-20261110/experiments.csv) · [Experimentos JSON](../models/runs/papila-final-20261110/experiments.json).

### Arquitectura y ajuste fino

Cada celda muestra verdaderos positivos de 13, verdaderos negativos de 49 y ROC-AUC. Todos los modelos usan BCE con peso 3,81 en esta etapa. [Selección de arquitecturas](../models/runs/papila-final-20261110/architecture-selection.json).

| Ajuste fino | ResNet18 | ResNet34 | EfficientNet-B0 |
|---|---:|---:|---:|
| Última etapa, BN congelada | 11/41 · 0,849 | 11/40 · 0,882 | 11/41 · 0,876 |
| Cabeza sola | 10/39 · 0,871 | 9/38 · 0,827 | 11/42 · 0,923 |
| Dos etapas finales | 11/42 · 0,889 | 9/29 · 0,656 | 11/41 · 0,882 |
| Progresivo | 10/44 · 0,893 | 10/45 · 0,871 | 11/42 · 0,896 |
| Última etapa, BN activa | 11/38 · 0,881 | **12/43 · 0,932** | 11/42 · 0,854 |

Avanzaron ResNet34 con BatchNorm activa en `layer4` y EfficientNet-B0 con cabeza sola. El mejor ajuste de ResNet18 fue de dos etapas. ResNet34 obtuvo su peor resultado al ajustar dos etapas. Frente a la última etapa con BN congelada, activar BN en EfficientNet-B0 obtuvo un verdadero negativo adicional y la misma sensibilidad, con menor ROC-AUC. En ResNet18 produjo tres falsos positivos adicionales en esa misma comparación.

### Pérdidas con ajuste fino fijo por arquitectura

| Arquitectura y pérdida | VP / FN | VN / FP | Sensibilidad | Especificidad | Precisión | F1 | F2 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet34 · BCE 3,81 | 12 / 1 | 43 / 6 | 92,3 % | 87,8 % | 66,7 % | 0,774 | 0,857 | 0,932 |
| ResNet34 · BCE normal | 11 / 2 | 42 / 7 | 84,6 % | 85,7 % | 61,1 % | 0,710 | 0,786 | 0,920 |
| ResNet34 · BCE 2 | 12 / 1 | 42 / 7 | 92,3 % | 85,7 % | 63,2 % | 0,750 | 0,845 | 0,925 |
| ResNet34 · focal | 11 / 2 | 43 / 6 | 84,6 % | 87,8 % | 64,7 % | 0,733 | 0,797 | 0,923 |
| EfficientNet-B0 · BCE 3,81 | 11 / 2 | 42 / 7 | 84,6 % | 85,7 % | 61,1 % | 0,710 | 0,786 | 0,923 |
| EfficientNet-B0 · BCE normal | 11 / 2 | 42 / 7 | 84,6 % | 85,7 % | 61,1 % | 0,710 | 0,786 | **0,934** |
| EfficientNet-B0 · BCE 2 | 11 / 2 | 42 / 7 | 84,6 % | 85,7 % | 61,1 % | 0,710 | 0,786 | 0,923 |
| EfficientNet-B0 · focal | 11 / 2 | 43 / 6 | 84,6 % | 87,8 % | 64,7 % | 0,733 | 0,797 | 0,911 |

BCE 3,81 avanzó en ResNet34. Focal avanzó en EfficientNet-B0 por un falso positivo menos con la misma sensibilidad. La BCE normal de EfficientNet ordenó mejor la validación que ambas variantes ponderadas y focal. Las tres referencias de EfficientNet con cabeza y tasa iguales se fijaron antes de test para comprobar si esa ventaja se trasladaba a la cohorte interna.

### Optimizadores, tasas, lotes y regularización

- **Tasas.** Adam a 0,0001 fue el mejor de los valores evaluados en semilla 42 para ambas arquitecturas finalistas. ResNet34 pasó de 12 VP y 43 VN a 12 VP y 46 VN. EfficientNet-B0 con focal pasó de 11 VP y 43 VN a 12 VP y 44 VN. La tasa diferenciada de ResNet34 obtuvo 11 VP y 46 VN.
- **Optimizadores.** AdamW con penalización 0,0001 obtuvo un VN más que Adam con el mismo L2 en ResNet34, con la misma sensibilidad. Adam sin penalización y tasa 0,0001 obtuvo el mejor equilibrio. En EfficientNet-B0, SGD con momentum a 0,0003 ganó un VP frente al Adam base a la misma tasa. Adam a 0,0001 mantuvo ese VP y añadió un VN. El SGD a 0,003 redujo especificidad en ResNet34 y sensibilidad en EfficientNet-B0.
- **Planificadores y lotes.** Ninguno superó a la mejor tasa fija. En ResNet34 el lote 32 acertó dos VN más que acumular dos microlotes de 16 con el mismo número de VP. La BN activa calcula estadísticas por microlote y puede contribuir a la diferencia. EfficientNet-B0 con cabeza sola obtuvo los mismos conteos con ambos esquemas. La latencia y el pico de memoria se reportan por corrida.
- **Regularización y aumento.** En ResNet34 a 0,0001, el control de volteo obtuvo 12 VP y 46 VN. Rotación mantuvo los 12 VP y cayó a 38 VN. Brillo, contraste, zoom hacia afuera, ausencia de volteo, dropout, mayor penalización y suavizado quedaron en 11 VP. Reducir paciencia a 3 eligió otro checkpoint y redujo el balance. Esta evidencia corresponde a un ancla y una semilla.

## Confirmación por semillas y modelo seleccionado

| Configuración | Semilla 42 VP/VN | Semilla 43 VP/VN | Semilla 44 VP/VN | Sensibilidad media ± DE | Especificidad media ± DE | Media del mínimo | F2 medio | ROC-AUC media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0, cabeza, focal, Adam 0,0001 | 12/44 | 11/41 | 11/42 | 87,2 ± 4,4 % | 86,4 ± 3,1 % | **86,03 %** | 0,810 | 0,886 |
| ResNet34, última etapa y BN activa, BCE 3,81, Adam 0,0003 | 12/43 | 11/44 | 11/42 | 87,2 ± 4,4 % | 87,8 ± 2,0 % | 85,66 % | 0,817 | 0,896 |
| ResNet34, última etapa y BN activa, BCE 3,81, Adam 0,0001 | 12/46 | 11/46 | 10/42 | 84,6 ± 7,7 % | 91,2 ± 4,7 % | 84,62 % | 0,818 | 0,906 |

Cada receta llegó a 85/85 en una de tres semillas, siempre la 42. ResNet34 a 0,0001 tuvo la mayor dispersión en sensibilidad. EfficientNet-B0 ganó por 0,37 puntos porcentuales en la media del mínimo frente a ResNet34 con BN activa y tasa 0,0003. Esa ResNet34 obtuvo mejores medias de especificidad, F2 y ROC-AUC. La ventaja de EfficientNet depende del criterio de equilibrio fijado y es pequeña frente a la dispersión de las semillas. [Resultados por semilla](../models/runs/papila-final-20261110/seed_comparison.csv) · [Medias y desviaciones](../models/runs/papila-final-20261110/seed_summary.csv).

Se fijó la semilla **44**, época **15**, del grupo ganador por cercanía a sus medianas. Ajusta **1.281** de **4.008.829** parámetros, solo la cabeza, con backbone y BN congelados, focal $\gamma=2$, Adam a **0,0001**, lote 16, FP32 y volteo horizontal. El umbral **0,35721901059150696** obtuvo 11 VP, 2 FN, 42 VN y 7 FP en validación. La sensibilidad fue 84,6 %, la especificidad 85,7 %, F1 0,710, F2 0,786 y ROC-AUC 0,871. El resultado medio del grupo y el punto operativo del checkpoint son estimaciones distintas. [Checkpoint](../models/runs/papila-final-20261110/confirmation/efficientnet_b0-lr-low-s44/fit/best.pt) · [Curvas de entrenamiento](../models/runs/papila-final-20261110/confirmation/efficientnet_b0-lr-low-s44/fit/loss.png) · [Métricas por época](../models/runs/papila-final-20261110/confirmation/efficientnet_b0-lr-low-s44/fit/validation.png) · [Umbrales evaluados](../models/runs/papila-final-20261110/confirmation/efficientnet_b0-lr-low-s44/validation/thresholds.csv).

## Coste y optimización de hardware

Se completaron 62 de un máximo de 64 intentos sin fallos de entrenamiento. Sumaron 649,8 s dentro del entrenador. Ese acumulado excluye evaluación, intervalos y Grad-CAM. El máximo asignado por PyTorch fue 576 MiB y el máximo de RAM residente observado fue 1.748 MiB. La caché compartida mantuvo aproximadamente 193 MiB de píxeles de train y validación. La primera corrida llenó la caché. Los tiempos siguientes corresponden al proceso ya activo.

| Control de última etapa | Parámetros | Tiempo total | VRAM pico asignada | Uso GPU medio | Inferencia mediana, lote 1 |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 11.177.025 | 31,67 s | 313 MiB | 25,3 % | 5,78 ms |
| ResNet34 | 21.285.185 | 11,85 s | 408 MiB | 71,5 % | 11,11 ms |
| EfficientNet-B0 | 4.008.829 | 11,47 s | 263 MiB | 44,7 % | 21,16 ms |

La latencia de lote uno mide el pase FP32 sincronizado sin decodificación. EfficientNet-B0 tuvo menos parámetros y menor memoria, con mayor latencia de inferencia que las ResNet en estos controles. El tiempo total incluye inicialización y gráficos y depende de la cantidad de épocas.

| Piloto de la receta ganadora, cinco épocas | Total | Media por época | VRAM asignada | RAM residente máxima | Uso GPU medio | Pasos omitidos |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 3,81 s | 0,64 s | 249 MiB | 832 MiB | 57,9 % | 0 |
| FP16 | 7,10 s | 1,30 s | 249 MiB | 1.320 MiB | 19,4 % | 0 |
| BF16 | 3,60 s | 0,61 s | 249 MiB | 1.596 MiB | 32,6 % | 0 |

Los tres pilotos obtuvieron las mismas decisiones de validación en el punto operativo y ROC-AUC 0,771, con un horizonte más corto que el entrenamiento final. La diferencia de 0,21 s entre FP32 y BF16 en una ejecución no justifica cambiar la precisión del candidato. Con ResNet34, la acumulación 16 × 2 redujo el pico asignado de 511 a 457 MiB frente al lote 32. Con EfficientNet-B0 redujo el pico de 419 a 249 MiB y conservó los conteos de validación. La receta elegida cabe holgadamente con lote 16 y usa una actualización por lote.

## Evaluación interna con selección bloqueada

Antes de puntuar el test se fijaron los tres representantes finalistas, los controles de arquitectura, las tres pérdidas de EfficientNet-B0 con cabeza sola y tasa 0,0003, y las referencias anteriores. Los puntos operativos se seleccionaron en validación. [Bloqueo de checkpoints y umbrales](../models/runs/papila-final-20261110/test-lock.json) · [Comparación CSV de ambas particiones](../models/runs/papila-final-20261110/final_comparison.csv) · [Métricas e intervalos JSON](../models/runs/papila-final-20261110/metrics.json).

| Modelo, test interno | VP / FN | VN / FP | Sensibilidad | Especificidad | Precisión | F1 | F2 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidato seleccionado, EfficientNet-B0 cabeza y focal 0,0001 | 13 / 4 | 45 / 22 | 76,5 % | 67,2 % | 37,1 % | 0,500 | 0,631 | 0,824 |
| Finalista 2, ResNet34 BN activa a 0,0003 | 11 / 6 | 61 / 6 | 64,7 % | 91,0 % | 64,7 % | 0,647 | 0,647 | 0,894 |
| Finalista 3, ResNet34 BN activa a 0,0001 | 13 / 4 | 62 / 5 | 76,5 % | 92,5 % | 72,2 % | 0,743 | 0,756 | 0,873 |
| Control ResNet18, última etapa | 12 / 5 | 57 / 10 | 70,6 % | 85,1 % | 54,5 % | 0,615 | 0,667 | 0,838 |
| Control ResNet34, última etapa | 12 / 5 | 53 / 14 | 70,6 % | 79,1 % | 46,2 % | 0,558 | 0,638 | 0,865 |
| Control EfficientNet-B0, última etapa | 13 / 4 | 58 / 9 | 76,5 % | 86,6 % | 59,1 % | 0,667 | 0,722 | 0,881 |
| Candidato previo EfficientNet-B0 | 12 / 5 | 57 / 10 | 70,6 % | 85,1 % | 54,5 % | 0,615 | 0,667 | 0,860 |
| ResNet18 ponderada histórica | 10 / 7 | 59 / 8 | 58,8 % | 88,1 % | 55,6 % | 0,571 | 0,581 | 0,836 |
| Demo publicada | 6 / 11 | 63 / 4 | 35,3 % | 94,0 % | 60,0 % | 0,444 | 0,385 | 0,795 |

El candidato ganó un VP frente al candidato anterior y produjo doce FP adicionales. La sensibilidad mejoró 5,9 puntos porcentuales y la especificidad cayó 17,9 puntos. El IC pareado al 95 % de esa diferencia en especificidad fue [−28,8, −8,5] puntos. El IC de la diferencia en ROC-AUC fue [−0,107, 0,022]. Entre los 12 modelos evaluados, la mayor sensibilidad en test fue 76,5 %. Estas cifras sustentan conservar la versión vigente.

| Candidato e intervalo por paciente | Validación | Test interno |
|---|---:|---:|
| Sensibilidad, IC 95 % | 84,6 % [66,7, 100] | 76,5 % [50,0, 100] |
| Especificidad, IC 95 % | 85,7 % [72,5, 96,2] | 67,2 % [54,3, 79,4] |
| ROC-AUC, IC 95 % | 0,871 [0,741, 0,981] | 0,824 [0,674, 0,961] |
| F2, IC 95 % | 0,786 [0,556, 0,926] | 0,631 [0,389, 0,794] |

La ResNet34 a 0,0001 tuvo el mejor equilibrio puntual en test entre los tres finalistas fijados. Su sensibilidad fue inferior a la meta, con IC 95 % [50,0, 100], y su especificidad tuvo IC [86,4, 98,4]. Las tres repeticiones de esa receta ya habían mostrado dispersión elevada. Su resultado motiva una nueva evaluación independiente de la receta fijada. La regla de validación conserva el significado del resultado aun cuando su ganador pierda en el test exploratorio.

### ¿Mejoró BCE normal en EfficientNet-B0?

La comparación controlada de pérdidas mantuvo cabeza sola, Adam 0,0003, semilla 42 y el resto de parámetros.

| Pérdida de EfficientNet-B0, test | VP / FN | VN / FP | Sensibilidad | Especificidad | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BCE normal | 12 / 5 | 52 / 15 | 70,6 % | 77,6 % | 0,837 |
| BCE 3,81 | 12 / 5 | 53 / 14 | 70,6 % | 79,1 % | 0,830 |
| Focal | 12 / 5 | 55 / 12 | 70,6 % | 82,1 % | 0,866 |

BCE normal mejoró ROC-AUC en 0,007 frente a BCE 3,81, con IC pareado [−0,004, 0,021]. Produjo un FP adicional con la misma sensibilidad. En este punto operativo y esta semilla, focal mejoró especificidad y ordenamiento frente a ambas BCE. La focal ganadora del cribado, a otra tasa y semilla representativa, tuvo peor especificidad en test. La pérdida y la receta completa deben interpretarse juntas. [Curvas ROC y precisión–recall de pérdidas](../models/runs/papila-final-20261110/test/loss-roc-pr.png).

[Matriz del candidato](../models/runs/papila-final-20261110/test/candidate/confusion.png) · [Curvas de los finalistas](../models/runs/papila-final-20261110/test/comparison-roc-pr.png) · [Matriz de ResNet34 a 0,0001](../models/runs/papila-final-20261110/test/finalist-3/confusion.png).

## Análisis de errores, pacientes y Grad-CAM

Se revisaron todos los errores del candidato, los ojos con desacuerdo entre arquitecturas y tres controles unánimes por clase. Fueron 20 ojos de validación y 38 de test, con cuatro mapas por ojo. Las 11 hojas muestran original, candidato, ResNet18, ResNet34 y EfficientNet-B0 de control. Los contornos oficiales se extrajeron de un ZIP cuyo MD5 coincide con Figshare. Los 146 JPEG de ambas particiones coinciden byte a byte con los originales del ZIP. Se usaron 584 contornos de dos expertos, disponibles para disco y copa. Esa información se reservó para el análisis posterior a la evaluación. [Procedencia de contornos](../models/runs/papila-final-20261110/official-contours/source.json) · [Resumen cuantitativo](../models/runs/papila-final-20261110/error-analysis/summary.json).

- **Falsos negativos.** En test fueron `RET021OS`, `RET026OD`, `RET026OS` y `RET088OS`, de tres pacientes. Los cuatro también fueron FN de los tres controles. El ojo derecho de 021 fue VP, mientras ambos ojos de 026 fueron FN. El mapa del candidato se concentró sobre la región del disco en tres de los cuatro FN. `RET088OS` produjo mapa uniforme sin señal positiva tras ReLU. En validación, los dos FN fueron `RET025OD` y `RET265OS`. Ver el disco en un mapa puede coexistir con un logit insuficiente para la etiqueta verdadera.
- **Falsos positivos.** Los 22 FP de test correspondieron a 17 pacientes. Incluyen errores bilaterales en 149, 165, 229, 254 y 277. Los cuatro modelos marcaron como glaucoma ambos ojos sanos de 229, y coincidieron en FP en `RET135OD` y `RET153OD`. En `RET034OS` y `RET036OS`, el candidato produjo FP y los tres controles acertaron. Esos patrones muestran errores compartidos y desacuerdos dependientes del modelo.
- **Calidad visual.** Las imágenes de validación de 033 y 175 muestran exposición reducida y sombras. En toda la validación, la mediana de luminancia fue 0,136 en FP frente a 0,243 en VN. En test las medianas fueron 0,240 y 0,265. La varianza laplaciana mediana fue 0,000219 en FP y 0,000281 en VN. El cálculo se hizo en miniaturas de tamaño fijo, excluyendo píxeles oscuros. Son descriptores de intensidad y detalle. Los errores también incluyeron imágenes iluminadas, como `RET190OS`, y la comparación descriptiva admite confusión por anatomía y adquisición.
- **Mapas y patrones accidentales.** En los FP del candidato, la mediana de masa de Grad-CAM dividida por área relativa del disco fue 9,71. En los FN con señal fue 13,71. Las medianas del borde exterior fueron aproximadamente 0,002 y 0,0. Las hojas de ResNet18 y ResNet34 de control muestran máximos periféricos en ejemplos como `RET004OS`, `RET026OS`, `RET135OS` y `RET282OS`. Eso motiva investigar el uso de bordes y fondos, aunque el mapa aislado no establece causalidad. La concentración sobre disco o copa tampoco establece patología ni calidad diagnóstica.
- **Agrupación por paciente.** Los 26 errores de test afectaron a 20 de 42 pacientes, seis con ambos ojos errados. En validación, nueve errores afectaron a siete de 31 pacientes, dos con ambos ojos errados. El remuestreo por paciente conserva esa dependencia y evita tratar los dos ojos como observaciones independientes.

[Errores por ojo de test](../models/runs/papila-final-20261110/error-analysis/test/errors.csv) · [Resultados por paciente](../models/runs/papila-final-20261110/error-analysis/test/patients.csv) · [Mapa y contornos de los FN](../models/runs/papila-final-20261110/error-analysis/test/gradcam-001.png) · [Errores bilaterales compartidos](../models/runs/papila-final-20261110/error-analysis/test/gradcam-025.png) · [Descriptores y regiones por mapa](../models/runs/papila-final-20261110/error-analysis/test/gradcam.csv).

## Integración y conclusiones

Se verificaron carga y hash del candidato, RGB y normalización, punto operativo, sigmoide, clases y Grad-CAM a través de `/predict`, `/explain` y `/analyze`. Se sirvió el frontend compilado en las pruebas locales. CPU y CUDA coincidieron en las dos imágenes de validación verificadas dentro de la tolerancia absoluta de $10^{-4}$, con PNG válidos. [Verificación de FastAPI y frontend](../models/runs/papila-final-20261110/backend-verification.json).

| Solicitud local, media de seis llamadas | Demo CPU | Candidato CPU | Demo CUDA | Candidato CUDA |
|---|---:|---:|---:|---:|
| Predicción | 93 ms | 69 ms | 45 ms | 55 ms |
| Análisis con Grad-CAM | 309 ms | 251 ms | 211 ms | 233 ms |

Las mejoras de código incorporadas son configuraciones experimentales de AdamW y SGD, dropout en cabeza con restauración compatible, suavizado BCE, acumulación ponderada por el tamaño real del grupo, caché compartida y evaluaciones bloqueadas con intervalos agrupados. La versión publicada conserva el checkpoint verificado al inicio.

Pasaron 74 pruebas Python, el lint y la compilación del frontend. Se verificaron los 62 hashes de checkpoints de campaña, los cinco consolidados, las 12 matrices de cada partición, las 11 hojas Grad-CAM y las referencias locales del informe.

La campaña deja las siguientes respuestas.

| Pregunta | Conclusión sustentada |
|---|---|
| Mejor arquitectura y ajuste | EfficientNet-B0 con cabeza sola ganó la media del mínimo en validación. La diferencia frente a ResNet34 fue pequeña y perdió especificidad en test. ResNet34 con BN activa es el finalista para confirmación externa. |
| Mejor pérdida | BCE 3,81 fue mejor en el cribado de ResNet34. Focal fue mejor en el punto operativo de EfficientNet-B0 con cabeza sola. BCE normal elevó algo el AUC de esa comparación y produjo un FP adicional en test. |
| Mejor optimizador y tasa | Adam a 0,0001 ofreció el mejor cribado por arquitectura. Las pruebas de AdamW y SGD fueron de una semilla. |
| Aumentos y regularización | El volteo fue el control más equilibrado en el ancla evaluada. Los aumentos adicionales, dropout, mayor penalización, suavizado y paciencia 3 redujeron el balance. |
| Umbral | 0,357219 para el representante de EfficientNet-B0, seleccionado exclusivamente en validación. Los otros checkpoints conservan sus propios umbrales bloqueados. |
| Reproducibilidad | Tres semillas por finalista, con medias, desviaciones y elección de representante central. Cada receta solo cumplió 85/85 en una semilla. |
| Modelo para GlaucoAI | Conservar la demo vigente. Guardar los tres finalistas como experimentales y evaluar la ResNet34 a 0,0001 en pacientes independientes antes de considerar sustitución. |

El test interno ha tenido múltiples consultas en campañas anteriores. Esta búsqueda agrega muchos modelos y decisiones sobre 31 pacientes de validación. Las semillas no aportan pacientes nuevos. Los intervalos agrupados describen la variabilidad de esas cohortes bajo modelos fijados, con independencia externa pendiente. También quedaron pendientes repeticiones del rendimiento de hardware y confirmación de pérdidas y optimizadores descartados con más semillas. La resolución 224 puede perder detalles finos y los mapas 7 × 7 son espacialmente gruesos. Esas limitaciones restringen el uso a investigación académica.

## Sustentación académica

### Evidencia de clase y técnicas ya presentes

| Evidencia | Uso en esta campaña |
|---|---|
| `Deep Learning para Vision (1).vtt`, 00:44:39, 00:58:00 y 01:14:19–01:17:56 | Transfer Learning, familia ResNet, ajuste por bloques y BatchNorm congelada. |
| `Deep Learning para Vision (2).vtt`, 00:13–00:15, 00:37–00:55 y 01:22:45 | Test fijo, entrenamiento en PyTorch, entropía cruzada y aumento. La discusión menciona iluminación y rotación. |
| `Deep Learning para Vision (3).vtt`, 00:32–00:50 y 01:08–01:12 | CNN, filtros, pooling, dropout y efecto de tasa y lote. |
| `Deep Learning para Vision (4).vtt`, 00:38–00:40, 01:04–01:11 y 01:33–01:35 | Pérdida, gradiente y detección de sobreajuste con curvas de train y validación. |
| `Deep Learning para Vision.vtt`, 00:19–00:24 | YOLO y SAM en una demostración de detección y segmentación. |

El notebook histórico usa Adam, SGD con momentum, penalización L2 y parada temprana. Esa evidencia acredita su uso en el proyecto. Las transcripciones acreditan los conceptos de optimización y regularización indicados en la tabla. ResNet18 y ResNet34 aprenden una función residual $F(x)$ sumada a $x$ o su proyección. EfficientNet-B0 sustituye esos bloques por etapas MBConv, con expansión, convolución por canal, proyección y atención por canal. La familia escala profundidad, anchura y resolución con un coeficiente común. B0 reduce parámetros, aunque la latencia depende de los kernels. [ResNet](https://arxiv.org/abs/1512.03385) · [EfficientNet](https://arxiv.org/abs/1905.11946).

### Técnicas cuya enseñanza no se confirmó

| Técnica | Fundamento, uso y compromiso |
|---|---|
| BCE ponderada y focal | Para $p=\sigma(z)$ y etiqueta $y$, BCE ponderada es $-[wy\log p+(1-y)\log(1-p)]$. $217/57$ equilibra los totales de clases de train y 2 prueba una intervención menor. Focal reduce el aporte de ejemplos fáciles con $-(1-p_t)^2\log p_t$. El primero actúa por clase y el segundo por dificultad. Ambos pueden cambiar la distribución de puntajes y focal puede aumentar la influencia de etiquetas ambiguas. [Focal Loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html). |
| AdamW | Adam mantiene momentos del gradiente y adapta el tamaño de paso por parámetro. AdamW aplica el encogimiento $\theta\leftarrow(1-\eta\lambda)\theta$ fuera del cálculo de momentos, mientras Adam con L2 añade $\lambda\theta$ al gradiente. Se comparan ambos con el mismo $\lambda=10^{-4}$ y con Adam sin penalización. El coeficiente necesita ajuste y no garantiza mejorar con pocos datos. [Artículo de AdamW](https://arxiv.org/abs/1711.05101). |
| Tasas por grupo y planificadores | Las tasas diferenciadas dejan avanzar la cabeza más rápido y conservan más los pesos transferidos. `ReduceLROnPlateau` divide la tasa entre dos tras dos épocas sin mejora apreciable de BCE. Coseno usa $\eta_t=\eta_0[1+\cos(\pi t/T)]/2$ con $T=15$. El primero responde al ruido de validación y el segundo depende del horizonte. Se registra la tasa aplicada por época. [CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) · [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). |
| Suavizado binario | Mezcla la etiqueta con la uniforme mediante $y'=(1-\epsilon)y+\epsilon/2$. Se aplica al objetivo BCE de entrenamiento y deja intactas las etiquetas de evaluación. Puede reducir logits extremos. También puede reducir separación o interactuar con el peso de clase. [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567). |
| Acumulación, AMP y caché | Se promedia el gradiente por número real de imágenes en el grupo de microlotes, incluyendo el último grupo parcial. Reduce el pico de VRAM frente al lote equivalente, con diferencias numéricas y de BatchNorm si está activa. AMP usa operaciones elegibles en FP16 o BF16 y FP32 para las restantes. FP16 necesita escalado y puede omitir pasos. BF16 conserva el rango del exponente de FP32 con menor mantisa. La caché conserva geometría y tensor inicial, y repite aumentos estocásticos en cada época. [AMP](https://docs.pytorch.org/docs/stable/amp.html). |
| Grad-CAM y contornos | El peso de un canal es el promedio espacial de $\partial z/\partial A_k$. La suma ponderada con ReLU localiza evidencia para el logit positivo. Se compara la fracción de masa del mapa con la fracción de área en disco, copa y regiones de control. Una cuadrícula pequeña produce mapas gruesos y el alineamiento espacial no demuestra causalidad ni patología. [Grad-CAM](https://arxiv.org/abs/1610.02391). |
| Umbral equilibrado y bootstrap agrupado | Maximizar el mínimo penaliza la tasa más baja. El umbral depende de una cohorte pequeña y del checkpoint elegido. Remuestrear por paciente conserva la dependencia entre ojos. El área ROC bootstrap usa pares positivo-negativo con medio crédito para empates. Los intervalos de la cohorte exploratoria no incorporan el sesgo de reutilizar validación. |

## Reproducción local

Usa una carpeta de campaña nueva. Se requiere la copia local verificada de PAPILA y los checkpoints históricos. El entorno usado fue Python 3.14.6, PyTorch 2.14.1+cu130 y Torchvision 0.29.1+cu130. Descarga el ZIP oficial v2 de Figshare con ID 35013982 para el análisis de contornos. Su MD5 es `7d73fc5157d44b221a62ed5a4baf3779`. [Entorno de ejecución](../models/runs/papila-final-20261110/environment.json) · [Descarga oficial de PAPILA v2](https://ndownloader.figshare.com/files/35013982).

```powershell
$glaucoPython = 'C:/Users/nesto/Proyectos/DeepLearning/.venv/Scripts/python.exe'
$glaucoCampaign = 'models/runs/papila-final-repro'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
& $glaucoPython -m scripts.evaluate_research audit --campaign $glaucoCampaign
& $glaucoPython -m scripts.train_final --campaign $glaucoCampaign
& $glaucoPython -m scripts.evaluate_final lock --campaign $glaucoCampaign
& $glaucoPython -m scripts.evaluate_final test --campaign $glaucoCampaign
& $glaucoPython -m scripts.verify_research_backend --campaign $glaucoCampaign
& $glaucoPython -m scripts.extract_final_contours --archive 'RUTA/PAPILA.zip' --output "$glaucoCampaign/official-contours"
& $glaucoPython -m scripts.analyze_final_errors --campaign $glaucoCampaign --contours "$glaucoCampaign/official-contours"
& $glaucoPython -m unittest discover -s tests -q
```

Si las predicciones de test ya existen y se interrumpe el informe, `python -m scripts.evaluate_final finalize --campaign RUTA` reconstruye el consolidado y los intervalos desde los CSV. Para una predicción individual experimental, `python -m src.predict --checkpoint RUTA --image JPEG --threshold 0.35721901059150696` usa el punto operativo del candidato. Los resultados, pesos, predicciones y figuras quedan en la carpeta local de campaña, excluida de Git.
