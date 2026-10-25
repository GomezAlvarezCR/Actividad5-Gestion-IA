# CHANGELOG — Actividad 5

## v1.0.0 — 2026-10-25
- Se eligió el dataset público Wisconsin Diagnostic Breast Cancer, 569 filas y 30 predictores.
- Se separaron archivos fuente y limpios (`datos_org/`, `datos_limp/`) y se registraron hashes SHA-256.
- Se comprobó ausencia de nulos y duplicados; se estandarizaron nombres y se recodificó el objetivo a `1=maligno`.
- Se exploró desbalance de clases (357 benignos/212 malignos) y valores atípicos potenciales sin eliminarlos automáticamente.
- Se construyeron pipelines, partición estratificada 80/20 y GridSearchCV con 5 pliegues.
- Se compararon regresión logística y random forest con 7 métricas, matrices de confusión y curvas ROC.
- Se registraron parámetros, métricas y modelos en MLflow (SQLite); se descartaron runs duplicados de las pruebas de depuración.
- Se comprobó el bloqueo del dashboard en Colab por `Invalid Host header`; se exportó evidencia mediante la API de tracking.
- Se documentaron limitaciones, deuda técnica, ética y posibles mejoras MLOps.

## Próximas iteraciones propuestas (no implementadas)
- Validación externa en una cohorte independiente; pruebas por subgrupos y calibración.
- Control automático de calidad, CI y pruebas unitarias; evaluación repetida o bootstrap.
- Monitorización de cambios en distribuciones, rendimiento y falsos negativos tras despliegue.
