# Modelo de Predicción JPMorgan y Alicorp con Transformers

Evaluación comparativa de arquitecturas Transformer para la predicción del precio de cierre bursátil, aplicada a JPMorgan Chase (NYSE:JPM) y Alicorp (BVL:ALICORC1). El proyecto desarrolla tres niveles de madurez del modelo (General, Mejorado y una extensión de análisis de sentimiento) y los valida mediante un backtest walk-forward de 500 días bursátiles, comparándolos contra baselines naive y lineales con tres pruebas estadísticas de significancia.

**Autora:** Mirian Lucero Ancco Ancalla — Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)

## Contenido del repositorio

### `Notebooks/`

| Notebook | Descripción |
|---|---|
| `JPMorgan_Transformer_General.ipynb` | Réplica base del modelo Transformer-encoder aplicada a JPMorgan (Close + SMA_5, salida puntual). |
| `JPMorgan_Transformer_Mejorado.ipynb` | Variante reforzada: regresión cuantílica, ensemble de 5 semillas, baselines lineales (NLinear/DLinear) obligatorios, y validación estadística extendida (Wilcoxon, Diebold-Mariano, bootstrap por bloques). |
| `Alicorp_Transformer_General.ipynb` | Mismo modelo base aplicado a Alicorp; discontinuado tras el backtest inicial por ausencia de ventaja predictiva. |
| `JPMorgan_Analisis_Sentimiento_FinBERT.ipynb` | Extensión exploratoria: sentimiento de titulares de noticias (Finnhub + FinBERT) fusionado con la señal del modelo Mejorado. |

### `Datos/`

| Archivo | Contenido |
|---|---|
| `JPMorgan_Stock_Price_History.csv` | Histórico de precios de JPMorgan usado para entrenar y evaluar los modelos. |
| `Alicorp_ALICORC1_Historico_Precios.csv` | Histórico de precios de Alicorp. |
| `JPM_General_backtest_500d_resultados.csv` | Resultados del backtest walk-forward de 500 días — JPMorgan General. |
| `JPM_Mejorado_backtest_resultados.csv` | Resultados del backtest walk-forward de 500 días — JPMorgan Mejorado. |
| `Alicorp_General_backtest_1d_resultados.csv` / `Alicorp_General_backtest_h5_resultados.csv` | Resultados de backtest de Alicorp a 1 y 5 días. |
| `JPM_Sentimiento_titulares_log.csv` | Registro diario de titulares procesados y su sentimiento promedio (FinBERT). |
| `JPM_Sentimiento_fusion_log.csv` | Registro de la fusión entre la señal del modelo Mejorado y el sentimiento de noticias. |

### Documentos

- **`Informe_Final_JPMorgan.pdf`** — Informe de exposición completo: marco teórico, arquitectura, hiperparámetros, resultados de cada backtest, seguimiento en vivo fuera de muestra, validación con TradingView y con dinero real, y conclusiones.
- **`Paper_Transformer_JPMorgan.pdf`** — Paper académico en formato IEEE (resumen, trabajos relacionados, metodología, resultados, discusión, limitaciones, conclusión y referencias).

## Resultado principal

| Modelo | Precisión direccional (5 días) | Veredicto |
|---|---|---|
| Alicorp — General | 48.6% | Sin ventaja frente al baseline naive |
| JPMorgan — General | 54.2% | Sin ventaja frente al baseline naive |
| JPMorgan — Mejorado | 61.8% | Ventaja parcial, confirmada en el seguimiento en vivo fuera de muestra |

Ningún modelo demuestra una ventaja estadísticamente robusta y sin matices sobre la estrategia más simple posible (asumir que el precio de mañana será el de hoy). La variante Mejorado es la única que sostiene una ventaja pequeña pero consistente entre el backtest histórico y 12 jornadas reales de seguimiento fuera de muestra — el detalle completo está en el informe y el paper.

## Requisitos

Los notebooks están pensados para ejecutarse en Google Colab. Dependencias principales: `torch`, `pandas`, `numpy`, `scipy`, `transformers` (para el notebook de sentimiento).
