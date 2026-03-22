# Solana Price Prediction (SOL-USD)
**Final Project — Data Science Diploma · ENES UNAM León 2025**

## Description
Machine Learning pipeline to predict the next-day closing price of Solana (SOL-USD) using historical Yahoo Finance data. Models are trained on scale-invariant features (returns, ratios) to avoid distribution shift across different price regimes.

## Project Structure
```
diplomado2025_entregable/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # Data download from Yahoo Finance
│   ├── 02_eda.ipynb                  # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb  # Scale-invariant features + return target
│   ├── 04_modeling.ipynb             # Ridge Regression + XGBoost + price reconstruction
│   └── 05_evaluation.ipynb           # Final metrics and visualizations
├── data/
│   ├── raw/                          # Raw CSV and charts (generated)
│   └── processed/                    # Processed data and saved models (generated)
├── requirements.txt
└── README.md
```

> `data/` is not committed — regenerate it by running the notebooks in order.

## Installation
```bash
pip install -r requirements.txt
```

Python 3.12 required. Optionally use the local virtual environment:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run the notebooks in order (01 → 05) from Jupyter:
```bash
jupyter lab
```

## Models
- **Ridge Regression** (baseline) — linear model, requires StandardScaler
- **XGBoost** (main model) — gradient boosting with early stopping on validation set

### Results (test set)
| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression | 3.56 | 4.81 | 3.76 | 0.9339 |
| XGBoost | 3.24 | 4.34 | 3.40 | 0.9461 |

## Feature Engineering
All features are **scale-invariant** to prevent distribution shift when price levels change between training and test periods:

| Category | Features |
|---|---|
| Return lags | `return_lag_1/2/3/5/7/14` — daily return (%) shifted by N days |
| Price/MA ratios | `Close_to_SMA7/21`, `Close_to_EMA12/26` — % deviation from moving average |
| Momentum | `RSI_14`, `MACD_pct`, `MACD_signal_pct` — MACD normalized by Close |
| Volatility | `BB_width`, `BB_position` (0=lower band, 1=upper band), `volatility_21d` |
| Volume | `Volume_change`, `Volume_ratio` (vs 7-day average) |
| Calendar | `day_of_week`, `month` |

**Target**: next-day return (%) via `Close.pct_change().shift(-1) * 100`. Predicted returns are converted back to prices using `price_t+1 = Close_t × (1 + return_pred / 100)`.

## Architecture Decisions
- **No data leakage**: train/val/test split is strictly chronological (70/15/15%). StandardScaler is fit only on train data.
- **Scale-invariant features**: absolute price features (lags, SMA, EMA, MACD) are replaced with returns and ratios so models generalize across price regimes.
- **Return-based target**: predicting % return instead of absolute price removes non-stationarity and allows XGBoost to extrapolate correctly.
- **XGBoost early stopping**: trained on raw (unscaled) features with `early_stopping_rounds=30` on the validation set.

## Technologies
Python · yfinance · scikit-learn · XGBoost · pandas · matplotlib · seaborn · ta
