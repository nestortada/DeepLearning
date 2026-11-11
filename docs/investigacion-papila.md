# GlaucoAI · Investigación con PAPILA

Campaña iniciada el 10 de noviembre de 2026 en la RTX 4050 Laptop de 6 GB. La unidad de predicción es el ojo y la unidad de separación y remuestreo es el paciente. PAPILA contiene retinografías centradas en la papila y etiquetas sana, glaucoma y sospechosa. El protocolo binario congelado excluye la tercera categoría. [Publicación de PAPILA](https://www.nature.com/articles/s41597-022-01388-1).

**Recomendación para GlaucoAI.** Conservar la ResNet18 publicada y guardar EfficientNet-B0 con tasas diferenciadas como candidato experimental. El candidato obtuvo 84,6 % de sensibilidad y 87,8 % de especificidad en validación. En test interno obtuvo 70,6 % y 85,1 %, respectivamente. En ese test igualó las tasas de la ResNet18 reentrenada y mejoró ROC-AUC de 0,838 a 0,860. La sensibilidad quedó por debajo del objetivo de 85 % en ambas particiones.

## Auditoría previa

| Partición | Pacientes | Ojos sanos | Ojos con glaucoma |
|---|---:|---:|---:|
| Entrenamiento | 137 | 217 | 57 |
| Validación | 31 | 49 | 13 |
| Test interno | 42 | 67 | 17 |

El manifiesto local `data/papila/manifest-c.csv` tiene SHA-256 `c82776614b94d458e42e56b8f66270e1fbe0be9b4a80a741e2cb6923449ef5cd`. Se verificaron los tres inventarios de JPEG contra la configuración congelada. Los 210 pacientes aportan dos ojos cada uno y tienen una sola asignación. Se comprobaron 420 hashes de imagen distintos. Esta comprobación detecta duplicados de bytes. Los recortes o reexportaciones de una misma captura requerirían una comprobación visual o perceptual adicional.

El notebook histórico usa otra asignación de pacientes. Los resultados de esta campaña se calculan con el manifiesto congelado y se comparan con checkpoints asociados a ese mismo hash. El test congelado ya se había consultado para la demo y para experimentos de ajuste fino y pérdidas. Una mejora en ese test es exploratoria. La confirmación requiere otro conjunto de pacientes, idealmente con un centro o dispositivo de adquisición independiente.

El cargador valida IDs, lateralidad, rutas y pertenencia del paciente. En entrenamiento solo instancia cargadores de train y validación. La rama de transferencia usa RGB, ajuste proporcional a 224 × 224 con relleno negro y normalización fija de ImageNet. Se verificó que los 420 JPEG son RGB y su orientación EXIF efectiva es 1. El volteo horizontal se aplica solo en entrenamiento. La caché conserva la salida determinista anterior al aumento aleatorio y a la normalización. Se estiman 193 MiB de tensores FP32 para 336 ojos en caché, más los gastos del proceso. La caché local se usa con cero workers para evitar duplicaciones por proceso.

El protocolo anterior elegía la menor BCE de validación y evaluaba una grilla de umbrales a intervalos de 0,05 con un mínimo de especificidad del 80 %. El umbral 0,40 era el punto operativo archivado de la ResNet18 ponderada. La demo publicada tiene su propio umbral de 0,35. Esta campaña conserva esos puntos para los comparadores históricos y calcula el punto del candidato exclusivamente con validación.

## Protocolo ejecutado

1. Comparación de ResNet18, ResNet34 y EfficientNet-B0 con semillas 42, 43 y 44. Cada modelo parte de su checkpoint ImageNet oficial, verificado por SHA-256. Todos usan la misma entrada, Adam, tasa 0,0003, lote 16, FP32 con modo IEEE explícito para convolución y multiplicación, BCE con peso positivo exacto `217/57`, cabeza de un logit y último bloque ajustable con BatchNorm congelada. Para EfficientNet-B0 se ajustan la última etapa MBConv y la convolución final. Se sustituyó su clasificador por una cabeza lineal, igual que en las ResNet.
2. Exploración en la mejor arquitectura de la comparación anterior. Cada variante cambia un factor del control y usa semilla 42. Se ajustan solo la cabeza, dos etapas finales, una etapa tras tres épocas de cabeza, o BatchNorm de la etapa ajustable. Se prueban dos tasas fijas, tasas por grupo, dos planificadores, dos aumentos y dos pesos alternativos de BCE. Las dos variantes con mejor criterio de validación se repiten con semillas 43 y 44. Las interacciones entre cambios quedan fuera de este diseño.
3. Todos los entrenamientos de calidad tienen un máximo de 15 épocas y paciencia de cinco. Se guarda el checkpoint con la menor BCE **sin ponderar** de validación para aplicar el mismo criterio a las pérdidas. Las curvas muestran la pérdida de entrenamiento de cada configuración y la BCE común de validación. Las variantes con peso distinto optimizan pérdidas en escalas diferentes.
4. Se evalúan los puntos medios entre puntajes de validación consecutivos distintos, más los extremos 0 y 1, usando la regla `puntaje ≥ umbral`. Primero se favorece cumplir ambos objetivos de 85 %. Luego se maximiza `min(sensibilidad, especificidad)`, seguido por la media de las dos tasas y la sensibilidad. El umbral más alto resuelve el último empate. La fila fija de 0,40 se guarda como referencia. En validación, los objetivos exigen al menos 12/13 ojos positivos y 42/49 negativos acertados.
5. Para ordenar configuraciones con tres semillas, primero se cuenta cuántas cumplen ambos objetivos. Se desempata con la media del mínimo, la media de la exactitud balanceada, la sensibilidad media y ROC-AUC media. En la configuración ganadora se fija el checkpoint individual que mejor equilibra las tasas de validación y se usa la BCE común como desempate posterior. Tres semillas estiman sensibilidad a la inicialización, al orden de lotes y al aumento en un mismo conjunto de pacientes. La selección entre múltiples variantes y umbrales en 31 pacientes puede producir estimaciones optimistas.
6. El fichero `test-lock.json` fija candidato, comparadores, hashes y umbrales. El evaluador comprueba esa decisión y los JPEG antes de puntuar el test. Guarda predicciones por ojo, matrices, ROC y precisión–recall. El bootstrap pareado remuestrea pacientes completos para comparar al candidato con la ResNet18 ponderada histórica y con la demo, conservando la correlación entre ambos ojos.
7. Los pilotos de rendimiento usan tres épocas de ResNet18, semilla 42 y una sola variación por corrida. La época inicial incluye llenado de caché o arranque de workers. Se reportan esa época y la media de las siguientes. `nvidia-smi` mide utilización global de la GPU a intervalos de 500 ms y PyTorch mide picos de memoria asignada y reservada. La latencia de lote 1 mide el pase FP32 sincronizado. Las solicitudes de FastAPI se cronometran por separado e incluyen decodificación y, en `/analyze`, Grad-CAM y PNG.

Precisión significa `VP/(VP+FP)`, sensibilidad `VP/(VP+FN)` y especificidad `VN/(VN+FP)`. F1 pondera por igual precisión y sensibilidad. F2 pondera cuatro veces la sensibilidad en la media armónica ponderada. ROC-AUC y AP se calculan sobre los logits. AP promedia la precisión ponderada por los incrementos de sensibilidad. El área trapezoidal PR-AUC también se informa como columna propia. Los puntajes son sigmoides de los logits y carecen de calibración probabilística.

## Resultados de arquitectura en validación

Media y desviación estándar muestral sobre tres semillas. Cada checkpoint tiene un umbral de validación propio. [Métricas por corrida](../models/runs/papila-research-20261110/architectures-validation/runs.csv) · [Estabilidad por arquitectura](../models/runs/papila-research-20261110/architectures-validation/variants.csv).

| Arquitectura | Sensibilidad | Especificidad | Media del mínimo | F2 | ROC-AUC | Semillas 85/85 |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 84,6 ± 0,0 % | 84,4 ± 1,2 % | 84,0 % | 0,778 | 0,868 | 0/3 |
| ResNet18 | 82,1 ± 4,4 % | 81,0 ± 3,1 % | 79,4 % | 0,741 | 0,834 | 0/3 |
| ResNet34 | 82,1 ± 4,4 % | 80,3 ± 2,4 % | 78,7 % | 0,737 | 0,860 | 0/3 |

EfficientNet-B0 detectó 11/13 ojos positivos en las tres semillas y la especificidad osciló entre 41/49 y 42/49. Mejoró 4,6 puntos porcentuales en la media del mínimo y 0,035 de ROC-AUC frente a la ResNet18 reentrenada. ResNet34 mejoró el ordenamiento frente a ResNet18 y obtuvo un mínimo medio algo menor en el punto operativo. La comparación de ajustes continuó con EfficientNet-B0.

| Arquitectura | Parámetros totales | Entrenables | Entrenamiento medio | Primera época media | Época posterior media | Pico asignado medio | Uso GPU medio | Mediana de inferencia media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 4.008.829 | 1.122.865 | 23,4 s | 16,43 s | 0,71 s | 264 MiB | 25,2 % | 9,44 ms |
| ResNet18 | 11.177.025 | 8.389.121 | 27,8 s | 16,85 s | 0,74 s | 313 MiB | 29,7 % | 3,83 ms |
| ResNet34 | 21.285.185 | 13.107.713 | 28,2 s | 16,26 s | 0,88 s | 407 MiB | 37,9 % | 6,20 ms |

La primera época incluye decodificación y almacenamiento en RAM. El tiempo total también depende de la parada temprana y las figuras que guarda el entrenador. EfficientNet-B0 usó menos parámetros y memoria, con mayor latencia FP32 de lote uno en estas mediciones. La latencia es la media de tres medianas y puede variar con frecuencia de la GPU, kernels y carga del sistema.

## Ajustes controlados de EfficientNet-B0

Los siguientes valores son de validación con la semilla 42. Las catorce configuraciones de la tabla, incluido el control, acertaron 11/13 positivos (84,6 %). La especificidad distingue puntos que conservan esa sensibilidad. [Resultados completos del cribado](../models/runs/papila-research-20261110/screen-validation/runs.csv).

| Cambio respecto al control | Época elegida | Umbral | VN/49 | F2 | ROC-AUC | BCE validación |
|---|---:|---:|---:|---:|---:|---:|
| Control, etapa final y BN congelada, BCE 3,81 | 7 | 0,2026 | 41 | 0,775 | 0,876 | 0,316 |
| Solo cabeza | 12 | 0,4501 | 42 | 0,786 | 0,923 | 0,478 |
| Dos etapas finales | 7 | 0,2930 | 41 | 0,775 | 0,882 | 0,329 |
| Progresivo, cabeza durante tres épocas | 7 | 0,3278 | 42 | 0,786 | 0,896 | 0,319 |
| BN activa en la etapa ajustada | 5 | 0,2683 | 42 | 0,786 | 0,854 | 0,357 |
| Tasa fija 0,0001 | 9 | 0,2340 | 40 | 0,764 | 0,873 | 0,339 |
| Tasa fija 0,001 | 6 | 0,1092 | 39 | 0,753 | 0,876 | 0,324 |
| Tasa 0,0003 cabeza y 0,00003 backbone | 8 | 0,4157 | 43 | 0,797 | 0,911 | 0,342 |
| ReduceLROnPlateau | 2 | 0,2808 | 43 | 0,797 | 0,900 | 0,330 |
| CosineAnnealingLR | 7 | 0,3149 | 43 | 0,797 | 0,879 | 0,322 |
| Brillo y contraste | 7 | 0,1713 | 40 | 0,764 | 0,876 | 0,317 |
| Rotación y escala conservadora | 7 | 0,1292 | 40 | 0,764 | 0,867 | 0,332 |
| BCE normal | 4 | 0,2534 | 43 | 0,797 | 0,885 | 0,306 |
| BCE con peso 2 | 4 | 0,3614 | 43 | 0,797 | 0,882 | 0,315 |

- Para ajuste fino, el progresivo y la cabeza sola ganaron un verdadero negativo frente al control. La cabeza ordenó mejor los puntajes y tuvo la mayor BCE de validación. Descongelar una segunda etapa no mejoró el punto operativo. Activar BatchNorm recuperó un negativo con menor ROC-AUC.
- Con tasa global fija, 0,0003 conservó más verdaderos negativos que 0,0001 y 0,001. Las tasas diferenciadas y los dos planificadores ganaron dos verdaderos negativos. Se repitieron tasas diferenciadas y `ReduceLROnPlateau` por mayor ROC-AUC dentro del empate operativo. En la semilla 42, el mejor checkpoint de `ReduceLROnPlateau` se guardó en época 2, antes de la primera reducción. Su puntaje de esa semilla no demuestra un efecto del planificador.
- El volteo horizontal del control conservó un verdadero negativo más que cada aumento adicional. Rotación y ajuste fotométrico quedan descartados para esta configuración. El aumento se aplicó después de la caché, conservando una nueva muestra aleatoria por acceso.
- BCE común y peso 2 mejoraron dos verdaderos negativos en la semilla 42 de EfficientNet-B0. BCE común obtuvo la menor pérdida de validación y mayor ROC-AUC que peso 2. El resultado de una semilla todavía no establece que el peso óptimo de EfficientNet sea 1 o 2. El barrido archivado de ResNet18 aporta evidencia complementaria sobre el peso en esa arquitectura.

### Confirmación con tres semillas en validación

| Configuración EfficientNet-B0 | Sensibilidad | Especificidad | Media del mínimo | F2 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tasas diferenciadas | 84,6 ± 0,0 % | 85,7 ± 2,0 % | 84,3 % | 0,786 | 0,887 |
| Control a tasa fija | 84,6 ± 0,0 % | 84,4 ± 1,2 % | 84,0 % | 0,778 | 0,868 |
| ReduceLROnPlateau | 84,6 ± 0,0 % | 83,7 ± 5,4 % | 82,3 % | 0,775 | 0,879 |

Las tasas diferenciadas ganaron por 0,3 puntos en la media del mínimo frente al control EfficientNet-B0 y mejoraron la especificidad media en 1,4 puntos. La sensibilidad se mantuvo en 11/13 para las tres semillas y ninguna alcanzó el doble objetivo. El resultado favorece la combinación de tasas dentro de esta cohorte. La dispersión en especificidad de `ReduceLROnPlateau` fue mayor y su media quedó por debajo del control. [Métricas de confirmación](../models/runs/papila-research-20261110/confirm-validation/variants.csv).

El candidato fijado es EfficientNet-B0 de semilla 42, época 8, con última etapa MBConv y convolución final ajustadas, BatchNorm congelada, BCE con peso 217/57, tasa 0,0003 en la cabeza y 0,00003 en el backbone. Conserva solo el volteo horizontal y usa FP32. Su umbral de validación es **0,41565512120723724**. El checkpoint obtuvo 11 VP, 2 FN, 43 VN y 6 FP, con sensibilidad 84,6 %, especificidad 87,8 %, precisión 64,7 %, F1 0,733, F2 0,797 y ROC-AUC 0,911. La BCE de validación cayó de 0,600 en época 1 a 0,342 en la época elegida y terminó en 0,350. [Checkpoint](../models/runs/papila-research-20261110/screen/lr-discriminative-s42/best.pt) · [Barrido de umbrales](../models/runs/papila-research-20261110/screen-validation/lr-discriminative-s42/thresholds.csv) · [Curvas de entrenamiento](../models/runs/papila-research-20261110/screen/lr-discriminative-s42/loss.png).

En validación, usar 0,40 en el mismo candidato dejó 11 VP y produjo 7 FP, frente a 6 FP con el umbral elegido. Ambos puntos empatan en el mínimo de sensibilidad y especificidad. El umbral 0,415655 gana el desempate por exactitud balanceada de 86,2 % frente a 85,2 %. [Predicciones y métricas con el umbral fijado](../models/runs/papila-research-20261110/locked-validation/candidate/metrics.json).

### Referencias archivadas y reproducción de métricas

Se recalcularon las métricas de los nueve checkpoints del barrido de peso de ResNet18 con sus umbrales originales. Los resultados coincidieron con el archivo en sensibilidad, especificidad, F2 y ROC-AUC para las tres semillas de cada peso. La regla original exigía 80 % de especificidad. [Verificación de las corridas históricas](../models/runs/papila-research-20261110/archive-validation/metrics.json).

| Peso positivo en ResNet18 | Sensibilidad media | Especificidad media | F2 medio | ROC-AUC media |
|---|---:|---:|---:|---:|
| 1 | 82,1 % | 81,6 % | 0,744 | 0,830 |
| 2 | 76,9 % | 84,4 % | 0,715 | 0,844 |
| 217/57 | 84,6 % | 83,7 % | 0,775 | 0,848 |

El peso 217/57 obtuvo el mejor F2 y la mayor sensibilidad media en ResNet18. Con la nueva regla de umbral, el checkpoint focal archivado detectó 10/13 positivos y 42/49 sanos (76,9 % y 85,7 %), con ROC-AUC 0,856 y F2 0,725. La ResNet18 ponderada archivada detectó 11/13 positivos y 42/49 sanos a 0,40. [Validación de focal loss](../models/runs/papila-research-20261110/locked-validation/focal-archived/metrics.json).

Se repitió además el entrenamiento de la ResNet18 ponderada de semilla 42. Coincidieron exactamente los tensores del mejor checkpoint, la época 12, la BCE de validación de 0,39374318430500643 y el historial común al quitar tiempos. Esa corrida usó la política por defecto de PyTorch, que permite TF32 para convolución en esta instalación. TF32 reduce la precisión de los operandos y acumula en FP32. La comparación nueva fijó convolución y multiplicación en modo IEEE. Su ResNet18 de semilla 42 seleccionó la época 15. [Comprobación tensor por tensor](../models/runs/papila-research-20261110/historical-reproduction/comparison.json) · [Semántica CUDA de PyTorch](https://docs.pytorch.org/docs/2.15/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

## Evaluación final en test interno

Se evaluaron 84 ojos de 42 pacientes después de fijar el candidato, las referencias, los hashes y los umbrales. La meta requería 15 VP de 17 y 57 VN de 67. El candidato alcanzó el requisito de especificidad y quedó tres verdaderos positivos por debajo del mínimo de sensibilidad. [Archivo de selección](../models/runs/papila-research-20261110/test-lock.json) · [Resultados por modelo](../models/runs/papila-research-20261110/test/metrics.csv).

| Modelo | Umbral | VP / FN | VN / FP | Sensibilidad | Especificidad | Precisión | F1 | F2 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0, tasas diferenciadas | 0,415655 | 12 / 5 | 57 / 10 | 70,6 % | 85,1 % | 54,5 % | 0,615 | 0,667 | 0,860 | 0,710 |
| ResNet18 ponderada reentrenada | 0,137272 | 12 / 5 | 57 / 10 | 70,6 % | 85,1 % | 54,5 % | 0,615 | 0,667 | 0,838 | 0,663 |
| ResNet18 ponderada archivada | 0,40 | 10 / 7 | 59 / 8 | 58,8 % | 88,1 % | 55,6 % | 0,571 | 0,581 | 0,836 | 0,634 |
| ResNet18 de la demo | 0,35 | 6 / 11 | 63 / 4 | 35,3 % | 94,0 % | 60,0 % | 0,444 | 0,385 | 0,795 | 0,569 |

Frente a la demo, EfficientNet-B0 detectó seis ojos adicionales con glaucoma y produjo seis falsos positivos adicionales. La diferencia fue de +35,3 puntos porcentuales en sensibilidad y −9,0 en especificidad. Frente a la ResNet18 ponderada archivada detectó dos casos adicionales y produjo dos falsos positivos adicionales (+11,8 y −3,0 puntos, respectivamente). Frente a la ResNet18 reentrenada con el mismo protocolo, obtuvo el mismo conteo de VP, FN, VN y FP, con +0,022 en ROC-AUC y +0,047 en AP. Mejorar el ordenamiento de puntajes aquí se tradujo en el mismo resultado agregado en el punto operativo fijado.

| Diferencia del candidato | Sensibilidad, IC 95 % | Especificidad, IC 95 % | ROC-AUC, IC 95 % |
|---|---:|---:|---:|
| Frente a ResNet18 ponderada archivada | +11,8 pp [0,0; 28,6] | −3,0 pp [−12,3; 4,7] | +0,025 [−0,042; 0,097] |
| Frente a la demo | +35,3 pp [12,5; 58,8] | −9,0 pp [−19,4; 0,0] | +0,065 [−0,016; 0,152] |

Los intervalos incluyen cero para las tres diferencias frente a la ponderada archivada. El intervalo de sensibilidad frente a la demo es positivo en el test interno. El historial de consultas del test y el número de pacientes limitan la generalización de este resultado. Se corrigió un error al registrar las estimaciones de diferencia del bootstrap y se completó el informe a partir de las predicciones ya guardadas, sin otra inferencia de test. [Intervalos pareados por paciente](../models/runs/papila-research-20261110/test/patient-bootstrap.json).

[Matriz del candidato](../models/runs/papila-research-20261110/test/candidate/confusion.png) · [ROC y precisión–recall de test](../models/runs/papila-research-20261110/test/roc-pr.png) · [Predicciones del candidato por ojo](../models/runs/papila-research-20261110/test/candidate/predictions.csv).

## Pilotos de GPU y canal de datos

Ocho corridas de ResNet18 con semilla 42 y tres épocas en CUDA. El control fue FP32 con caché, lote 16, cero workers, memoria fijada y transferencia no bloqueante. Cada fila cambia un factor, salvo los dos pilotos de workers, que requieren desactivar la caché y se comparan con el piloto sin caché. [Métricas, memoria reservada, latencia y utilización por corrida](../models/runs/papila-research-20261110/pilots-validation/runs.csv).

| Variante | Total | Primera época | Media épocas 2–3 | Pico asignado | Uso GPU medio | Pasos omitidos |
|---|---:|---:|---:|---:|---:|---:|
| FP32 con caché | 15,80 s | 13,15 s | 0,66 s | 313 MiB | 11,7 % | 0 |
| FP16 con caché | 16,09 s | 14,40 s | 0,58 s | 312 MiB | 17,6 % | 4 |
| BF16 con caché | 20,11 s | 17,92 s | 0,68 s | 313 MiB | 11,6 % | 0 |
| Sin memoria fijada ni transferencia no bloqueante | 23,12 s | 20,59 s | 0,78 s | 313 MiB | 12,4 % | 0 |
| Lote 32 | 23,39 s | 21,05 s | 0,63 s | 419 MiB | 10,1 % | 0 |
| Sin caché, cero workers | 62,75 s | 21,09 s | 20,33 s | 313 MiB | 8,2 % | 0 |
| Sin caché, dos workers | 61,29 s | 19,13 s | 20,64 s | 313 MiB | 5,5 % | 0 |
| Sin caché, dos workers persistentes | 50,70 s | 29,16 s | 10,34 s | 313 MiB | 8,3 % | 0 |

La caché redujo el total de 62,75 a 15,80 s en el par FP32 con cero workers, aproximadamente cuatro veces en este piloto. Los historiales coincidieron exactamente al quitar el tiempo y las 62 predicciones de validación fueron idénticas. Desactivar memoria fijada y transferencias no bloqueantes produjo también las mismas predicciones. Persistir los workers ayudó después del arranque al trabajar sin caché, con mayor tiempo total que la caché local. El cambio de workers modificó la secuencia aleatoria de aumento y produjo historiales distintos. Las mediciones incluyen el llenado de caché, la carga de workers y las figuras. Las diferencias pequeñas entre opciones con caché requieren repeticiones de rendimiento.

FP16 omitió cuatro actualizaciones y el tiempo total quedó cerca de FP32. BF16 funcionó sin omitir pasos y tardó más en el piloto. Se mantuvo FP32 para calidad. El lote 32 elevó el pico asignado y redujo el número de actualizaciones por época. Se mantuvo lote 16. Los pilotos ocuparon hasta 419 MiB de memoria asignada por PyTorch y hasta 690 MiB de reserva en el proceso. La reserva puede incluir bloques retenidos por el asignador entre corridas. La memoria observada dejó margen para entrenar sin acumulación de gradientes en la RTX 4050.

## Integración local de FastAPI

Se cargaron la demo y el candidato en instancias locales con el mismo frontend compilado. Con un ojo sano y uno con glaucoma de validación se verificaron el hash del checkpoint, dispositivo, preprocesamiento, puntajes, clases, umbral, y las rutas `/predict`, `/explain` y `/analyze`. CPU y CUDA coincidieron en las decisiones y los puntajes dentro de una tolerancia absoluta de $10^{-4}$. En las dos imágenes, Grad-CAM generó PNG válidos con señal. La demo usó `backbone.layer4[-1]` y el candidato `backbone.features[-1]`. [Verificación por ruta y dispositivo](../models/runs/papila-research-20261110/backend-verification.json).

| Modelo | Dispositivo | Predicción media | Análisis con Grad-CAM medio |
|---|---|---:|---:|
| Demo | CPU, 2 hilos | 116 ms | 391 ms |
| EfficientNet-B0 | CPU, 2 hilos | 106 ms | 548 ms |
| Demo | CUDA | 65 ms | 377 ms |
| EfficientNet-B0 | CUDA | 107 ms | 463 ms |

Se midieron seis solicitudes locales por ruta y condición con `TestClient`. Estos tiempos incluyen el procesamiento de la imagen y la serialización de la respuesta. La latencia de análisis del candidato fue mayor en ambos dispositivos en esta medición.

Pasaron 70 pruebas Python, el lint de React y la compilación de Vite. El SHA-256 de `backend/predict/best.pt` coincidió con la configuración publicada y con la auditoría inicial. Se verificaron los 34 directorios de la campaña y la reproducción histórica adicional, los checkpoints fijados, la lectura de las figuras PNG y las 84 filas de predicción de cada modelo en test.

## Fundamento académico

### Técnicas de clase

| Evidencia local | Técnica aplicada |
|---|---|
| `Deep Learning para Vision (1).vtt` · 00:44:39, 00:58:00 y 01:14:19–01:17:56 | Transferencia, cambio de cabeza, familia ResNet, descongelación por bloques y BatchNorm congelada. |
| `Deep Learning para Vision (2).vtt` · 00:13–00:15, 00:37–00:55 y 01:22:45 | Test fijo, entropía cruzada, PyTorch, optimizador, dispositivo y aumento. Se menciona iluminación y rotación. |
| `Deep Learning para Vision (3).vtt` · 00:32–00:50 y 01:08–01:12 | Convoluciones, max pooling, dropout, tamaño de lote y tasa de aprendizaje. |
| `Deep Learning para Vision (4).vtt` · 00:38–00:40, 01:04–01:11 y 01:33–01:35 | Pérdida y gradientes, análisis de las curvas de entrenamiento y validación, y ajuste de tasa y lote. |
| `Deep Learning para Vision.vtt` · 00:19–00:24 | Detección con YOLO y segmentación con SAM en la demostración. Esta investigación trabaja con clasificación por ojo. |

ResNet18 y ResNet34 son variantes de la familia de conexiones residuales vista en clase. El bloque aprende una función residual $F(x)$ que se suma a la ruta corta $x$ o a su proyección. ResNet34 añade bloques y capacidad a cambio de más cómputo y parámetros. El presupuesto, la resolución y el criterio de selección permiten comparar las variantes en PAPILA. [Artículo original de ResNet](https://arxiv.org/abs/1512.03385).

### Aviso sobre técnicas nuevas

La revisión de los cinco archivos no permitió confirmar que se enseñaran las siguientes técnicas. Las filas remiten a los experimentos de este informe o a los resultados archivados que se volvieron a puntuar.

| Técnica | Fundamento, implementación y compromiso |
|---|---|
| EfficientNet-B0 | Su bloque MBConv expande canales, aplica convolución espacial por canal y vuelve a proyectarlos. Squeeze-and-excitation repondera los canales a partir del promedio espacial. La familia escala profundidad, anchura y resolución como potencias de un factor común. B0 es el modelo base de la familia. Frente a ResNet, su menor número de parámetros puede favorecer memoria y almacenamiento. La latencia también depende de los kernels y de la cantidad de operaciones pequeñas. Se carga con Torchvision, con cabeza lineal y el mismo preprocesamiento de PAPILA. [Artículo original](https://arxiv.org/abs/1905.11946) · [Torchvision](https://docs.pytorch.org/vision/stable/models/efficientnet.html). |
| BCE ponderada | Para etiqueta $y$, logit $z$, $p=\sigma(z)$ y peso $w$, la pérdida es $-[w y \log p+(1-y)\log(1-p)]$. La rama positiva del gradiente es $w(p-1)$, frente a $p-1$ con BCE común. Se usa `BCEWithLogitsLoss(pos_weight=...)` por estabilidad numérica, con pesos 1, 2 y 217/57 calculado solo en train. Un peso mayor da más influencia a los casos de glaucoma. Puede desplazar puntajes y aumentar falsas alarmas a un umbral fijo. [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html). |
| Tasas diferenciadas | En Adam la tasa $\eta$ escala el paso calculado con los momentos del gradiente. Dos grupos permiten mover la cabeza nueva a 0,0003 y conservar más las características transferidas usando 0,00003 en el backbone. Esto puede reducir el sobreajuste o dejar corto el ajuste de dominio. La variante de clase con tasa global se usa como control. |
| Planificadores de tasa | `ReduceLROnPlateau` reduce la tasa a la mitad tras dos épocas consecutivas sin mejora apreciable de BCE de validación. Puede responder al estancamiento, con sensibilidad al ruido de un conjunto pequeño. `CosineAnnealingLR` usa $\eta_t=\eta_0[1+\cos(\pi t/T)]/2$ con $T=15$ y sin reinicios. Su descenso es suave y depende del presupuesto fijado. Ambos se comparan con tasa constante y registran la tasa de cada época. [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) · [CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html). |
| Ajuste progresivo y BatchNorm activa | El ajuste por bloques se vio en clase. Aquí se prueba una programación concreta de tres épocas de cabeza antes de habilitar el bloque final, manteniendo la selección por BCE. Otra variante aprende los parámetros afines y actualiza estadísticas solo en el bloque habilitado. Normaliza como $\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$. Puede adaptar estadísticas de ImageNet a PAPILA o hacerlas inestables con lotes pequeños. |
| Búsqueda de umbral y métricas | La sigmoide $1/(1+e^{-z})$ ordena los puntajes. Cambiar el umbral cambia las dos tasas en sentidos opuestos. Maximizar su mínimo penaliza al modelo por su tasa más baja y conserva la meta simétrica. Se calculan los puntos alcanzables con validación. Con 13 positivos el criterio es discreto y el umbral puede ser sensible a un solo ojo. |
| Precisión mixta y canal de datos | AMP deja que PyTorch ejecute operaciones elegibles con FP16 o BF16 y mantiene operaciones sensibles en FP32. FP16 tiene rango más estrecho y se usa con escalado dinámico de gradientes. BF16 conserva el rango de exponente de FP32 con menos precisión de mantisa. La evaluación se mantiene en FP32. La memoria fijada permite encolar transferencias no bloqueantes hacia CUDA y workers persistentes evitan reiniciar procesos por época. La caché evita repetir JPEG y geometría a cambio de RAM. Los pilotos separan el efecto de estas opciones. [AMP de PyTorch](https://docs.pytorch.org/docs/stable/amp.html) · [DataLoader](https://docs.pytorch.org/docs/stable/data.html). |
| Grad-CAM | Calcula $\alpha_k=\operatorname{media}_{i,j}(\partial z/\partial A^k_{ij})$ y aplica ReLU a $\sum_k \alpha_k A^k$. Destaca evidencia positiva de glaucoma a baja resolución espacial. Se usa la última salida residual o la última convolución de EfficientNet y se proyecta al área original después de quitar las bandas. El mapa puede ser uniforme y su utilidad depende de las características que el modelo haya aprendido. [Artículo original](https://arxiv.org/abs/1610.02391). |
| Bootstrap por paciente | Se extraen 42 pacientes con reemplazo y se incluyen sus dos ojos. Los mismos pacientes remuestreados se aplican a candidato y comparador. Los percentiles 2,5 y 97,5 de 2.000 diferencias estiman el intervalo pareado. Cuantifica incertidumbre del test con la correlación por paciente y tiene resolución limitada por el tamaño de la cohorte. |

### Decisiones sobre técnicas descartadas

- Se probó rotación de ±5° con reducción de escala a 0,92 para que la imagen completa permanezca dentro del cuadro, y brillo y contraste con factores de 0,90 a 1,10. El recorte y el zoom hacia adentro se descartaron porque pueden excluir parte de la papila, el borde neurorretiniano o la retina peripapilar. La rotación y la modificación fotométrica pueden suavizar detalles o cambiar contraste, respectivamente. Su conveniencia se decide con las corridas controladas.
- Focal loss multiplica la BCE de cada ojo por $(1-p_t)^2$ para reducir el peso de ejemplos fáciles. El experimento archivado con semilla 42 perdió sensibilidad en el punto operativo. Se vuelve a puntuar ese checkpoint en validación como antecedente y se conserva el presupuesto de nuevos entrenamientos para pesos BCE. El factor de dificultad puede magnificar el efecto de etiquetas ambiguas o imágenes atípicas. [Artículo original de focal loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).
- Label smoothing se descartó por la falta de una hipótesis independiente sobre ruido de etiqueta en la cohorte binaria y porque introduciría simultáneamente otro cambio en la separación de los puntajes. Las imágenes sospechosas ya se excluyen por el protocolo PAPILA congelado.

## Reproducción

Desde la raíz del repositorio, usa el intérprete de PyTorch con CUDA. La corrida utilizó el entorno local de la otra copia del repositorio. `--dataset-dir` y `--manifest` permiten apuntar a una copia verificada de PAPILA. Para repetir una campaña completa, usa un directorio de salida nuevo. El cierre de la comparación requiere también el checkpoint ponderado archivado.

```powershell
$glaucoPython = 'C:/Users/nesto/Proyectos/DeepLearning/.venv/Scripts/python.exe'
$glaucoCampaign = 'models/runs/papila-research-repro'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
& $glaucoPython -m scripts.evaluate_research audit --campaign $glaucoCampaign
foreach ($glaucoPhase in 'architectures','screen','confirm','pilots') {
    & $glaucoPython -m scripts.train_research $glaucoPhase --campaign $glaucoCampaign
    if ($LASTEXITCODE -ne 0) { throw "Entrenamiento incompleto en $glaucoPhase" }
    & $glaucoPython -m scripts.evaluate_research $glaucoPhase --campaign $glaucoCampaign
    if ($LASTEXITCODE -ne 0) { throw "Evaluación incompleta en $glaucoPhase" }
}
& $glaucoPython -m scripts.evaluate_research archive --campaign $glaucoCampaign
& $glaucoPython -m scripts.evaluate_research lock --campaign $glaucoCampaign
& $glaucoPython -m scripts.verify_research_backend --campaign $glaucoCampaign
& $glaucoPython -m scripts.evaluate_research test --campaign $glaucoCampaign
& $glaucoPython -m unittest discover -s tests -q
```

Para repetir el entrenamiento ponderado del archivo con la política de precisión por defecto de PyTorch, el comando independiente usa el mismo peso y mantiene la caché local.

```powershell
& $glaucoPython -m src.train --architecture resnet18 --tune-last-block --learning-rate 0.0003 `
    --positive-weight 3.80701754385965 --epochs 15 --no-amp --cache-preprocessed `
    --dataset-dir data/papila --manifest data/papila/manifest-c.csv --run-dir "$glaucoCampaign/historical-reproduction"
```

La API de evaluación se construye con `create_app(model_loader=...)` y verifica el hash antes de cargar el checkpoint. Para inferencia local individual, `python -m src.predict --checkpoint RUTA --image JPEG --threshold UMBRAL` permite usar el punto fijado. El valor se toma de `test-lock.json`.

Si se interrumpe el cálculo de intervalos después de guardarse todas las predicciones de test, `python -m scripts.evaluate_research finalize-test --campaign RUTA` completa los archivos a partir de los CSV. Los checkpoints, los CSV, los JSON, la telemetría y las figuras quedan en `models/runs/papila-research-20261110/`, una ruta local excluida de Git.
