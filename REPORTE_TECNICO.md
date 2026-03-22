Aquí está el informe traducido y enriquecido:

---

# Reporte Técnico — Predicción del Precio de Solana (SOL-USD)
**Diplomado en Ciencia de Datos · ENES UNAM León 2025**
**GitHub:** https://github.com/[tu-usuario]/diplomado2025_entregable

---

## 1. Resumen

Este proyecto desarrolla un pipeline de Machine Learning para predecir el precio de cierre del día siguiente de Solana (SOL-USD), una de las criptomonedas más activamente negociadas en el mercado. Los datos históricos OHLCV fueron obtenidos de Yahoo Finance mediante la librería `yfinance`, cubriendo el período de enero 2025 a marzo 2026 (444 observaciones diarias).

Se entrenaron y evaluaron dos modelos de aprendizaje supervisado: Ridge Regression como baseline lineal y XGBoost como modelo principal. Todos los features fueron diseñados para ser invariantes a la escala (retornos y razones en lugar de precios absolutos) para prevenir el desplazamiento de distribución (*distribution shift*) entre diferentes regímenes de precio. La variable objetivo es el retorno porcentual del día siguiente, que posteriormente se reconstruye como precio absoluto para la evaluación.

**a. Tipo de solución:** Regresión supervisada — pronóstico de series de tiempo

**b. Métricas principales (conjunto de prueba):**

| Modelo | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression (baseline) | 3.56 | 4.81 | 3.76 | 0.9339 |
| XGBoost | 3.24 | 4.34 | 3.40 | 0.9461 |

**c. Impacto:** XGBoost alcanza un error porcentual absoluto medio de 3.40%, lo que significa que el precio predicho del día siguiente se encuentra dentro de ~$3.24 del precio real en promedio. Ambos modelos explican más del 93% de la varianza en el conjunto de prueba.

> **Contexto:** Un MAPE de 3.40% es competitivo para pronóstico de criptomonedas a un día usando únicamente features técnicos. La literatura reporta errores típicos de 3–8% para este tipo de modelos en activos de alta volatilidad como SOL.

---

## 2. Introducción

Los mercados de criptomonedas se caracterizan por alta volatilidad, operación las 24 horas los 7 días de la semana, y sensibilidad a eventos externos (noticias regulatorias, cambios macroeconómicos, sentimiento en redes sociales). Solana (SOL) es un protocolo blockchain de capa 1 que ha experimentado variaciones de precio significativas, oscilando entre aproximadamente $77 y $262 en el período estudiado.

Predecir el precio del día siguiente de una criptomoneda es un problema de regresión desafiante debido a:

- **No estacionariedad**: el nivel absoluto del precio cambia dramáticamente a lo largo de meses, haciendo que los features derivados de precios crudos sean poco confiables entre diferentes ventanas de tiempo.
- **Ruido**: los retornos diarios en crypto son altamente volátiles (~5% de desviación estándar diaria para SOL).
- **Datos limitados**: a diferencia de los mercados tradicionales con décadas de historia, SOL ha sido negociado activamente solo desde 2020.
- **Sin ancla fundamental**: a diferencia de las acciones, no existen utilidades ni dividendos que proporcionen un piso de valuación.

> **Contexto adicional:** Solana fue lanzada en 2020 por Anatoly Yakovenko con el objetivo de superar las limitaciones de escalabilidad de Ethereum. Su mecanismo de consenso *Proof of History* le permite procesar miles de transacciones por segundo, lo que la posiciona como una de las blockchains de mayor rendimiento. Esta característica técnica atrae tanto a desarrolladores como a especuladores, lo cual se refleja en su perfil de volatilidad — más pronunciado que Bitcoin o Ethereum — y hace que el problema de predicción sea especialmente interesante desde el punto de vista estadístico.

A pesar de estos desafíos, los features de análisis técnico (medias móviles, RSI, MACD, Bandas de Bollinger) han demostrado contener señal predictiva en tareas de pronóstico de corto horizonte. Este proyecto investiga si estas señales, cuando se normalizan adecuadamente, permiten a los modelos de Machine Learning generalizar a través de diferentes regímenes de precio.

---

## 3. Objetivos del Proyecto

### a. Objetivo General
Diseñar, implementar y evaluar un pipeline de Machine Learning que prediga el precio de cierre del día siguiente de Solana (SOL-USD) usando datos históricos de mercado, aplicando la metodología de ciencia de datos estudiada a lo largo del diplomado.

### b. Objetivos Particulares
1. Descargar y validar datos históricos OHLCV de SOL-USD desde Yahoo Finance.
2. Realizar un Análisis Exploratorio de Datos (EDA) para caracterizar la distribución de retornos, volatilidad, estacionariedad y estructura de autocorrelación de la serie.
3. Construir un conjunto de 20 features técnicos invariantes a la escala (lags de retornos, razones precio/MA, indicadores de momentum normalizados) que permanezcan estables entre regímenes de precio.
4. Entrenar y ajustar dos modelos de regresión — Ridge Regression y XGBoost — usando una división cronológica estricta de entrenamiento/validación/prueba para prevenir fuga de datos (*data leakage*).
5. Evaluar ambos modelos con MAE, RMSE, MAPE, R² y precisión direccional, y comparar su desempeño entre sí.

---

## 4. Alcance del Proyecto

### Incluye
- Predicción del precio diario (cierre del día siguiente) para SOL-USD.
- Ingeniería de features basada únicamente en el historial de precio y volumen (análisis técnico).
- Dos familias de modelos: regresión lineal (Ridge) y gradient boosting (XGBoost).
- Evaluación cronológica con una ventana de prueba fija (último 15% del dataset).
- Pipeline completamente reproducible en Jupyter notebooks publicado en GitHub.

### No Incluye
- Predicción en tiempo real o intradía.
- Análisis de sentimiento (Twitter, Reddit, noticias).
- Datos on-chain (actividad de wallets, volumen de transacciones).
- Optimización de portafolio o estrategia de trading automatizado.
- Otras criptomonedas o modelos multi-activo.

### Restricciones Técnicas
- El modelo se entrena sobre datos históricos; no se actualiza automáticamente con nuevos datos de mercado.
- Los modelos basados en árboles (XGBoost) no pueden extrapolar más allá del rango de precios visto durante el entrenamiento cuando se usan features de precio absoluto — por esto los features invariantes a la escala fueron esenciales.
- Con ~295 muestras de entrenamiento (70% de 422 filas), el dataset es pequeño para gradient boosting; agregar más historia (desde 2023) podría mejorar la generalización.

> **Contexto:** Esta restricción de tamaño de muestra es una limitación real y honesta del proyecto. En un contexto profesional, se compensaría usando datos de mayor frecuencia (horarios o de 4 horas), incorporando otros activos correlacionados (BTC, ETH) como features adicionales, o aplicando técnicas de data augmentation para series temporales.

---

## 5. Marco Teórico

### 5.1 Ridge Regression
Ridge Regression es una forma regularizada de Mínimos Cuadrados Ordinarios que añade un término de penalización L2 a la función de pérdida:

```
minimizar: ||y - Xβ||² + α||β||²
```

El parámetro de regularización `α` contrae los coeficientes hacia cero, reduciendo el sobreajuste cuando los features están correlacionados (como es común con los indicadores técnicos). Ridge requiere estandarización de features (media cero, varianza unitaria) porque la penalización es sensible a la escala. Sirve como baseline lineal: si un modelo no lineal no puede superar a Ridge, sugiere que la relación es aproximadamente lineal o que el modelo no lineal está sobreajustando.

### 5.2 XGBoost (Extreme Gradient Boosting)
XGBoost es un método de ensamble que construye un modelo aditivo de árboles de decisión, donde cada nuevo árbol corrige los residuos del ensamble anterior:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

donde `h_m` es el m-ésimo árbol y `η` es la tasa de aprendizaje. Los mecanismos clave de regularización incluyen:
- `max_depth`: limita la profundidad del árbol para evitar memorización.
- `min_child_weight`: peso mínimo de instancias en una hoja.
- `subsample` / `colsample_bytree`: submuestreo de filas y columnas (similar a Random Forest).
- `reg_alpha` / `reg_lambda`: regularización L1 y L2 sobre los puntajes de las hojas.
- **Early stopping**: el entrenamiento se detiene cuando el RMSE de validación deja de mejorar durante 30 rondas consecutivas.

A diferencia de los modelos lineales, XGBoost no requiere estandarización de features y puede capturar interacciones no lineales entre variables automáticamente.

> **Contexto:** XGBoost fue desarrollado por Tianqi Chen en 2016 y desde entonces ha dominado competencias de datos tabulares en Kaggle. Su ventaja frente a Random Forest en este tipo de problema es que el boosting corrige errores sistemáticos iterativamente, mientras que bagging (Random Forest) solo reduce varianza. Para series temporales financieras con patrones de momentum, esto resulta en mejor captura de tendencias de corto plazo.

### 5.3 Indicadores Técnicos
- **SMA / EMA**: las Medias Móviles Simple y Exponencial suavizan el ruido del precio. Su razón con el cierre actual (ej. `Close/SMA_7 - 1`) mide cuánto se ha desviado el precio de su tendencia reciente.
- **RSI (Índice de Fuerza Relativa)**: oscilador en [0, 100] que mide la velocidad y magnitud de los cambios de precio recientes. Valores por encima de 70 sugieren sobrecompra; por debajo de 30, sobreventa.
- **MACD**: diferencia entre EMA-12 y EMA-26, normalizada por el precio de cierre para eliminar dependencia de escala. La línea de señal es una EMA de 9 períodos del MACD.
- **Bandas de Bollinger**: bandas superior e inferior a ±2 desviaciones estándar de una SMA de 20 días. `BB_position` mapea el precio actual linealmente entre la banda inferior (0) y la superior (1).

### 5.4 Invarianza de Escala y Distribution Shift
Una decisión de diseño crítica en este proyecto es reemplazar features de precio absoluto por features relativos. Cuando un modelo de árboles se entrena a un precio de ~$200 y se evalúa a ~$90, las condiciones de las hojas (`Close_lag_1 < 150`) aprendidas durante el entrenamiento ya no aplican — el modelo nunca vio esos valores absolutos y no puede extrapolar. Convertir todos los features a retornos (%) y razones elimina este problema porque la distribución estadística de estos features es aproximadamente estable entre niveles de precio.

> **Contexto:** Este es uno de los errores más comunes al aplicar Machine Learning a series financieras y el hallazgo más importante del proyecto. El modelo inicial con features absolutos obtuvo R²=−2.33 en el conjunto de prueba — peor que simplemente predecir la media — a pesar de tener buen desempeño en validación. Detectar y corregir este tipo de falla requiere entender la diferencia entre *generalización temporal* y *generalización estadística*, un concepto clave en cualquier aplicación de ML a datos financieros.

---

## 6. Metodología

### Entorno de Hardware y Software
El proyecto fue desarrollado en una computadora personal con sistema operativo Linux. No se requirió GPU. El stack de software es:
- Python 3.12
- Jupyter Lab para desarrollo interactivo
- Librerías principales: `yfinance 0.2`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `ta`
- Gestión de entorno virtual con `uv`

### Paso 1 — Adquisición de Datos (`01_data_ingestion.ipynb`)
Los datos OHLCV diarios de SOL-USD fueron descargados programáticamente desde Yahoo Finance usando la librería `yfinance` para el período del 1 de enero de 2025 al 20 de marzo de 2026, obteniendo 444 observaciones. Las cinco variables son:
- **Open**: precio de apertura de la jornada (USD)
- **High**: precio máximo alcanzado durante el día (USD)
- **Low**: precio mínimo alcanzado durante el día (USD)
- **Close**: precio de cierre — la variable objetivo (USD)
- **Volume**: volumen total de transacciones en USD

Las verificaciones básicas de calidad confirmaron cero valores nulos y sin fechas duplicadas.

### Paso 2 — Análisis Exploratorio de Datos (`02_eda.ipynb`)
El EDA examinó:
- Tendencias de precio y volumen a lo largo del período completo
- Distribución de retornos (media, desviación estándar, asimetría, curtosis)
- Volatilidad rolling para identificar regímenes de alta turbulencia
- Prueba de Dickey-Fuller Aumentada (ADF) para estacionariedad — los retornos diarios son estacionarios; los precios absolutos no lo son
- Autocorrelación (ACF) y autocorrelación parcial (PACF) para seleccionar longitudes de lag significativas
- Patrones de estacionalidad mensual

### Paso 3 — Ingeniería de Features (`03_feature_engineering.ipynb`)
Se construyeron 20 features invariantes a la escala:

| Categoría | Features |
|---|---|
| Lags de retornos | `return_lag_1/2/3/5/7/14` — retorno diario (%) desplazado N días |
| Razones precio/MA | `Close_to_SMA7/21`, `Close_to_EMA12/26` — desviación % de la MA |
| Momentum | `RSI_14`, `MACD_pct`, `MACD_signal_pct` |
| Volatilidad | `BB_width`, `BB_position`, `volatility_21d` |
| Volumen | `Volume_change`, `Volume_ratio` |
| Calendario | `day_of_week`, `month` |

La variable objetivo es el **retorno porcentual del día siguiente**:
```
Target_t = (Close_{t+1} - Close_t) / Close_t × 100
```

La última fila se eliminó después de calcular el target (no hay cierre futuro disponible). El dataset final contiene 422 muestras × 20 features.

### Paso 4 — Modelado (`04_modeling.ipynb`)
El dataset fue dividido **cronológicamente** (sin mezcla aleatoria) en:
- **Entrenamiento**: 70% (~295 muestras) — usado para ajustar los modelos
- **Validación**: 15% (~63 muestras) — usado para early stopping de XGBoost
- **Prueba**: 15% (~63 muestras) — usado únicamente para evaluación final

`StandardScaler` fue ajustado exclusivamente en el conjunto de entrenamiento y aplicado a validación y prueba para prevenir fuga de datos. Ridge Regression fue entrenado sobre features estandarizados; XGBoost sobre features crudos (la estandarización no es requerida para modelos de árboles).

Los retornos predichos fueron convertidos de vuelta a precios para la evaluación:
```
precio_predicho_{t+1} = Close_t × (1 + retorno_predicho / 100)
```

### Paso 5 — Evaluación (`05_evaluation.ipynb`)
Ambos modelos fueron evaluados en el conjunto de prueba usando cuatro métricas de regresión (MAE, RMSE, MAPE, R²) y una métrica de precisión direccional (¿predice el modelo correctamente si el precio sube o baja al día siguiente?).

---

## 7. Resultados

### Métricas de Regresión (conjunto de prueba)

| Modelo | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression | 3.56 | 4.81 | 3.76 | 0.9339 |
| XGBoost | **3.24** | **4.34** | **3.40** | **0.9461** |

XGBoost supera a Ridge en las cuatro métricas. El error absoluto medio de $3.24 representa aproximadamente 3.4% del precio típico en el período de prueba (rango ~$95–145), lo cual es competitivo para un pronóstico de crypto a un día usando únicamente features técnicos.

Ambos modelos explican más del 93% de la varianza en el conjunto de prueba (R² > 0.93), indicando que el conjunto de features invariantes a la escala generaliza bien al período fuera de muestra.

### Precisión Direccional

| Modelo | Precisión Direccional |
|---|---|
| Ridge Regression | 46.9% |
| XGBoost | 48.4% |
| Baseline aleatorio | ~50% |

Ninguno de los modelos predice confiablemente la dirección del movimiento de precio del día siguiente. Esto es esperado y destaca una distinción importante: predecir el **nivel** del precio dentro de ±$3.24 (R²=0.94) no implica predecir correctamente la **dirección** de un movimiento diario de ±$2–5. La predicción direccional requiere que el error del modelo sea menor que el movimiento mismo — un problema más difícil. Este resultado es consistente con la Hipótesis de Mercados Eficientes en el horizonte corto.

### Hallazgo Principal
La implementación inicial de XGBoost entrenada con features de precio absoluto (lags de precio crudo, SMA, EMA) alcanzó R²=−2.33 en el conjunto de prueba a pesar de buen desempeño en validación — síntoma de distribution shift. La reingeniería de todos los features como retornos y razones resolvió completamente esta falla, llevando a XGBoost a R²=0.9461 y confirmando que **la invarianza de escala en los features es un prerequisito para generalizar modelos de árboles en diferentes regímenes de precio de criptomonedas**.

> **Contexto:** Este hallazgo tiene implicaciones prácticas más allá de este proyecto. Cualquier modelo de ML aplicado a activos financieros con alta variación de precio (cryptos, acciones de alto crecimiento, materias primas) debe considerar si sus features son estables bajo diferentes regímenes. La métrica R²=−2.33 es una señal de alarma clara: un R² negativo significa que el modelo es peor que simplemente predecir el promedio de los precios del conjunto de entrenamiento, lo que indica que el modelo aprendió patrones específicos del régimen de precios alto y falló completamente al evaluar en el régimen de precios bajo.

---

## 8. Referencias

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Analysis. https://doi.org/10.1145/2939672.2939785

2. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

3. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.

4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

5. Pandas Development Team. (2024). *pandas: Powerful Python data analysis toolkit* (v2.x). https://pandas.pydata.org

6. Aroussi, R. (2024). *yfinance: Download market data from Yahoo! Finance's API*. https://github.com/ranaroussi/yfinance

7. Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.

8. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

9. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/