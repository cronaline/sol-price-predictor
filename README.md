# Predicción del Precio de Solana (SOL-USD)
**Proyecto Final — Diplomado en Ciencia de Datos · ENES UNAM León 2025**

## Descripción
Modelo de Machine Learning para predecir el precio de cierre del día siguiente de Solana (SOL-USD) usando datos históricos de Yahoo Finance (2024-2025).

## Estructura del proyecto
```
solana-predictor/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # Descarga de datos desde Yahoo Finance
│   ├── 02_eda.ipynb                  # Análisis Exploratorio de Datos
│   ├── 03_feature_engineering.ipynb  # Indicadores técnicos y lags
│   ├── 04_modeling.ipynb             # Ridge Regression + XGBoost
│   └── 05_evaluation.ipynb           # Métricas y visualizaciones finales
├── data/
│   ├── raw/                          # Datos crudos y gráficas
│   └── processed/                    # Datos procesados y modelos
├── requirements.txt
└── README.md
```

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
Ejecutar los notebooks en orden (01 → 05) desde Jupyter:
```bash
jupyter notebook
```

## Modelos utilizados
- **Ridge Regression** (baseline)
- **XGBoost** (modelo principal)

## Features principales
- Lags del precio de cierre (1, 2, 3, 5, 7, 14 días)
- Medias móviles: SMA 7, SMA 21, EMA 12, EMA 26
- Indicadores técnicos: RSI 14, MACD, Bandas de Bollinger
- Volatilidad rolling, cambio en volumen, features de calendario

## Tecnologías
Python · yfinance · scikit-learn · XGBoost · pandas · matplotlib
