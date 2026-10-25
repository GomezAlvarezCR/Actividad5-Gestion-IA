# Actividad 5 — Gestión de datos, entrenamiento y MLflow

## Objetivo
Clasificar casos del conjunto público Wisconsin Diagnostic Breast Cancer y comparar **regresión logística** frente a **random forest** bajo un protocolo reproducible, con búsqueda de hiperparámetros, validación cruzada y seguimiento mediante MLflow.

## Estructura
- `datos/datos_org/wdbc_original_v1.csv`: datos originales importados de scikit-learn.
- `datos/datos_limp/wdbc_clean_v1.csv`: nombres estandarizados, duplicados eliminados y objetivo recodificado.
- `fuentes/datos_prep.py`: adquisición, validación y visualizaciones de datos.
- `fuentes/train.py`: división estratificada, GridSearchCV, evaluación y registro MLflow.
- `fuentes/entrena.ipynb`: cuaderno de Google Colab (añadir el `.ipynb` usado en la ejecución).
- `evidencias/`: resultados y gráficas. Incorporar `mlflow_runs.csv`, `mlflow_details.txt` y capturas de la ejecución original.
- `mlflow.db` y `mlartifacts/`: base local y modelos registrados (copiar desde Colab si se incluyen en entrega; no se generan artificialmente).
- `Reporte_Tecnico_Actividad5.pdf`: síntesis del análisis y recomendaciones MLOps.
- `CHANGELOG.md`: registro del versionado y decisiones.

## Dataset
- Fuente: `sklearn.datasets.load_breast_cancer`, basado en Wisconsin Diagnostic Breast Cancer (UCI).
- 569 registros; 30 variables numéricas; sin faltantes ni duplicados en la fuente.
- Variable original `target_original`: **0 = maligno; 1 = benigno**.
- Variable transformada `target_maligno`: **1 = maligno; 0 = benigno**.
- Datos originales y limpios en archivos separados; SHA-256 registrado en `evidencias/quality_report.json`.
- Los valores extremos son señales por verificar, no errores demostrados. Se conservan todas las filas.

## Reproducción
Desde el directorio `Actividad5` con Python 3.11+:

```bash
python -m pip install -r requirements.txt
python fuentes/datos_prep.py
python fuentes/train.py
```

Para probar el modelo sin registro MLflow: `python fuentes/train.py --no-mlflow`.

**Advertencia:** ejecutar `train.py` nuevamente genera dos runs adicionales en MLflow. Para consultar los originales use la base `mlflow.db` exportada desde Colab.

## Protocolo experimental
- División *estratificada* 80/20 con `random_state=42` (455 entrenamiento / 114 prueba).
- Validación cruzada estratificada de 5 particiones, `shuffle=True`, semilla 42.
- Selección de hiperparámetros con `GridSearchCV`, optimizando ROC-AUC.
- Regresión logística: `StandardScaler` en pipeline (evita fuga de datos) + búsqueda de `C` y `class_weight`.
- Random forest: búsqueda de `n_estimators`, `max_depth` y `min_samples_leaf`.
- Métricas de prueba: accuracy, precision, recall, F1 y ROC-AUC; CV: promedio y dispersión de ROC-AUC.

## Resultados de ejecución en Colab (25/oct/2026)
| Modelo | Accuracy | Precision | Recall (maligno) | F1 | ROC-AUC | CV AUC media ± DE |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9960 | 0.9959 ± 0.0046 |
| Random forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9940 | 0.9891 ± 0.0067 |

En la prueba, la regresión logística produjo **2 falsos negativos** frente a **4** del bosque aleatorio. Se selecciona la regresión logística por su menor tasa de omisión de casos malignos, con la reserva de que una sola partición de prueba no demuestra superioridad generalizable.

## Seguimiento MLflow
Experimento `Actividad5_Comparacion_Modelos`. Cada ejecución registra parámetros, métricas, identificación de dataset por SHA-256, artefactos y un modelo serializado. En Colab, la interfaz web falló por validación del encabezado `Host`; esto **no afectó** los runs `FINISHED` obtenidos mediante la API de tracking. Se adjunta evidencia exportada por API (y base/artefactos al integrar desde Colab).

## Limitaciones y ética
Proyecto **académico**; no apto para decisiones clínicas. El conjunto es pequeño y curado, la división es única, no hay prueba externa, ni análisis por subgrupos, ni evaluación de calibración. Para producción se requerirían validación prospectiva, monitoreo de deriva, auditoría, gobernanza de datos y control de accesos.

## Fuentes
- scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Repositorio UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- MLflow: https://mlflow.org/docs/latest/ml/tracking/
