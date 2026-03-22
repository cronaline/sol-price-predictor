# Solana Price Prediction (SOL-USD)
**Final Project — Data Science Diploma · ENES UNAM León 2025**

## Description
Machine Learning model to predict the next-day closing price of Solana (SOL-USD) using historical Yahoo Finance data (2024-2025).

## Project Structure
```
solana-predictor/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # Data download from Yahoo Finance
│   ├── 02_eda.ipynb                  # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb  # Technical indicators and lags
│   ├── 04_modeling.ipynb             # Ridge Regression + XGBoost
│   └── 05_evaluation.ipynb           # Final metrics and visualizations
├── raw/                              # Raw data and charts
├── processed/                        # Processed data and models
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the notebooks in order (01 → 05) from Jupyter:
```bash
jupyter notebook
```

## Models
- **Ridge Regression** (baseline)
- **XGBoost** (main model)

## Key Features
- Closing price lags (1, 2, 3, 5, 7, 14 days)
- Moving averages: SMA 7, SMA 21, EMA 12, EMA 26
- Technical indicators: RSI 14, MACD, Bollinger Bands
- Rolling volatility, volume change, calendar features

## Technologies
Python · yfinance · scikit-learn · XGBoost · pandas · matplotlib
