# Technical Report — Solana Price Prediction (SOL-USD)
**Data Science Diploma · ENES UNAM León 2025**
**GitHub:** https://github.com/[your-username]/diplomado2025_entregable

---

## 1. Summary

This project develops a machine learning pipeline to predict the next-day closing price of Solana (SOL-USD), one of the most actively traded cryptocurrencies in the market. Historical OHLCV data was obtained from Yahoo Finance using the `yfinance` library, covering the period from January 2025 to March 2026 (444 daily observations).

Two supervised learning models were trained and evaluated: Ridge Regression as a linear baseline and XGBoost as the main model. All features were engineered to be scale-invariant (returns and ratios instead of absolute prices) to prevent distribution shift across different price regimes. The target variable is the next-day percentage return, which is then reconstructed into an absolute price for evaluation.

**a. Type of solution:** Supervised regression — time series forecasting

**b. Main metrics (test set):**

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression (baseline) | 3.56 | 4.81 | 3.76 | 0.9339 |
| XGBoost | 3.24 | 4.34 | 3.40 | 0.9461 |

**c. Impact:** XGBoost achieves a mean absolute percentage error of 3.40%, meaning the predicted next-day price is within ~$3.24 of the actual price on average. Both models explain more than 93% of the variance in the test set.

---

## 2. Introduction

Cryptocurrency markets are characterized by high volatility, 24/7 trading, and sensitivity to external events (regulatory news, macroeconomic changes, social media sentiment). Solana (SOL) is a layer-1 blockchain protocol that has experienced significant price swings, ranging from approximately $77 to $262 in the period studied.

Predicting the next-day price of a cryptocurrency is a challenging regression problem due to:

- **Non-stationarity**: the absolute price level changes dramatically over months, making features derived from raw prices unreliable across different time windows.
- **Noise**: daily returns in crypto are highly volatile (~5% daily standard deviation for SOL).
- **Limited data**: unlike traditional markets with decades of history, SOL has only been actively traded since 2020.
- **No fundamental anchor**: unlike stocks, there are no earnings or dividends to provide a valuation floor.

Despite these challenges, technical analysis features (moving averages, RSI, MACD, Bollinger Bands) have been shown to carry predictive signal in short-horizon forecasting tasks. This project investigates whether these signals, when properly normalized, allow machine learning models to generalize across different price regimes.

---

## 3. Project Objectives

### a. General Objective
Design, implement, and evaluate a machine learning pipeline that predicts the next-day closing price of Solana (SOL-USD) using historical market data, applying the data science methodology studied throughout the diploma.

### b. Specific Objectives
1. Download and validate historical OHLCV data for SOL-USD from Yahoo Finance.
2. Conduct an Exploratory Data Analysis (EDA) to characterize the return distribution, volatility, stationarity, and autocorrelation structure of the series.
3. Engineer a set of 20 scale-invariant technical features (return lags, price-to-MA ratios, normalized momentum indicators) that remain stable across price regimes.
4. Train and tune two regression models — Ridge Regression and XGBoost — using a strictly chronological train/validation/test split to prevent data leakage.
5. Evaluate both models using MAE, RMSE, MAPE, R², and directional accuracy, and compare their performance against each other.

---

## 4. Project Scope

### Included
- Daily price prediction (next-day close) for SOL-USD.
- Feature engineering based solely on price and volume history (technical analysis).
- Two model families: linear regression (Ridge) and gradient boosting (XGBoost).
- Chronological evaluation with a fixed test window (last 15% of the dataset).
- Full reproducible pipeline in Jupyter notebooks published on GitHub.

### Not Included
- Real-time or intraday prediction.
- Sentiment analysis (Twitter, Reddit, news).
- On-chain data (wallet activity, transaction volume).
- Portfolio optimization or automated trading strategy.
- Other cryptocurrencies or multi-asset models.

### Technical Constraints
- The model is re-trained on historical data; it does not update automatically with new market data.
- Tree-based models (XGBoost) cannot extrapolate beyond the price range seen during training when absolute price features are used — this is why scale-invariant features were essential.
- With ~295 training samples (70% of 422 rows), the dataset is small for gradient boosting; adding more historical data (2023 onwards) could further improve generalization.

---

## 5. Theoretical Framework

### 5.1 Ridge Regression
Ridge Regression is a regularized form of Ordinary Least Squares that adds an L2 penalty term to the loss function:

```
minimize: ||y - Xβ||² + α||β||²
```

The regularization parameter `α` shrinks coefficients toward zero, reducing overfitting when features are correlated (as is common with technical indicators). Ridge requires feature standardization (zero mean, unit variance) because the penalty is sensitive to scale. It serves as the linear baseline: if a non-linear model cannot beat Ridge, it suggests either that the relationship is approximately linear or that the non-linear model is overfitting.

### 5.2 XGBoost (Extreme Gradient Boosting)
XGBoost is an ensemble method that builds an additive model of decision trees, where each new tree corrects the residuals of the previous ensemble:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

where `h_m` is the m-th tree and `η` is the learning rate. Key regularization mechanisms include:
- `max_depth`: limits tree depth to prevent memorization.
- `min_child_weight`: minimum sum of instance weights in a leaf.
- `subsample` / `colsample_bytree`: row and column subsampling (similar to Random Forest).
- `reg_alpha` / `reg_lambda`: L1 and L2 weight regularization on leaf scores.
- **Early stopping**: training is halted when validation RMSE stops improving for 30 consecutive rounds.

Unlike linear models, XGBoost does not require feature scaling and can capture non-linear interactions between features automatically.

### 5.3 Technical Indicators
- **SMA / EMA**: Simple and Exponential Moving Averages smooth price noise. Their ratio to the current close (e.g., `Close/SMA_7 - 1`) measures how far price has deviated from its recent trend.
- **RSI (Relative Strength Index)**: oscillator in [0, 100] measuring the speed and magnitude of recent price changes. Values above 70 suggest overbought conditions; below 30 suggest oversold.
- **MACD**: difference between EMA-12 and EMA-26, normalized by the closing price to remove scale dependence. The signal line is a 9-period EMA of MACD.
- **Bollinger Bands**: upper and lower bands at ±2 standard deviations from a 20-day SMA. `BB_position` maps the current price linearly between the lower (0) and upper (1) band.

### 5.4 Scale-Invariance and Distribution Shift
A critical design decision in this project is replacing absolute price features with relative ones. When a tree model is trained at a price of ~$200 and evaluated at ~$90, the leaf conditions (`Close_lag_1 < 150`) that were learned during training no longer apply — the model has never seen those absolute values and cannot extrapolate. Converting all features to returns (%) and ratios eliminates this problem because the statistical distribution of these features is approximately stable across price levels.

---

## 6. Methodology

### Hardware and Software Environment
The project was developed on a personal computer running Linux. No GPU was required. The software stack is:
- Python 3.12
- Jupyter Lab for interactive development
- Key libraries: `yfinance 0.2`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `ta`

### Step 1 — Data Acquisition (`01_data_ingestion.ipynb`)
Daily OHLCV data for SOL-USD was downloaded programmatically from Yahoo Finance using the `yfinance` library for the period January 1, 2025 to March 20, 2026, yielding 444 observations. The five variables are:
- **Open**: opening price of the trading day (USD)
- **High**: highest price reached during the day (USD)
- **Low**: lowest price reached during the day (USD)
- **Close**: closing price — the target variable (USD)
- **Volume**: total trading volume in USD

Basic quality checks confirmed zero null values and no duplicate dates.

### Step 2 — Exploratory Data Analysis (`02_eda.ipynb`)
The EDA examined:
- Price and volume trends over the full period
- Return distribution (mean, standard deviation, skewness, kurtosis)
- Rolling volatility to identify high-turbulence regimes
- Augmented Dickey-Fuller (ADF) test for stationarity — daily returns are stationary; absolute prices are not
- Autocorrelation (ACF) and partial autocorrelation (PACF) to select meaningful lag lengths
- Monthly seasonality patterns

### Step 3 — Feature Engineering (`03_feature_engineering.ipynb`)
Twenty scale-invariant features were constructed:

| Category | Features |
|---|---|
| Return lags | `return_lag_1/2/3/5/7/14` — daily % return shifted by N days |
| Price/MA ratios | `Close_to_SMA7/21`, `Close_to_EMA12/26` — % deviation from MA |
| Momentum | `RSI_14`, `MACD_pct`, `MACD_signal_pct` |
| Volatility | `BB_width`, `BB_position`, `volatility_21d` |
| Volume | `Volume_change`, `Volume_ratio` |
| Calendar | `day_of_week`, `month` |

The target variable is the **next-day percentage return**:
```
Target_t = (Close_{t+1} - Close_t) / Close_t × 100
```

The last row was dropped after computing the target (no future close available). The final dataset contains 422 samples × 20 features.

### Step 4 — Modeling (`04_modeling.ipynb`)
The dataset was split **chronologically** (no shuffling) into:
- **Train**: 70% (~295 samples) — used to fit models
- **Validation**: 15% (~63 samples) — used for XGBoost early stopping
- **Test**: 15% (~63 samples) — used only for final evaluation

`StandardScaler` was fit exclusively on the training set and applied to validation and test to prevent data leakage. Ridge Regression was trained on scaled features; XGBoost on raw features (scaling is not required for tree models).

Predicted returns were converted back to prices for evaluation:
```
predicted_price_{t+1} = Close_t × (1 + predicted_return / 100)
```

### Step 5 — Evaluation (`05_evaluation.ipynb`)
Both models were evaluated on the test set using four regression metrics (MAE, RMSE, MAPE, R²) and a directional accuracy metric (does the model correctly predict whether price goes up or down the next day?).

---

## 7. Results

### Regression Metrics (test set)

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge Regression | 3.56 | 4.81 | 3.76 | 0.9339 |
| XGBoost | **3.24** | **4.34** | **3.40** | **0.9461** |

XGBoost outperforms Ridge on all four metrics. The mean absolute error of $3.24 represents approximately 3.4% of the typical price in the test period (~$95–145 range), which is competitive for a next-day crypto forecast using only technical features.

Both models explain more than 93% of the variance in the test set (R² > 0.93), indicating that the scale-invariant feature set generalizes well to the out-of-sample period.

### Directional Accuracy

| Model | Directional Accuracy |
|---|---|
| Ridge Regression | 46.9% |
| XGBoost | 48.4% |
| Random baseline | ~50% |

Neither model reliably predicts the direction of next-day price movement. This is expected and highlights an important distinction: predicting the **level** of price within ±$3.24 (R²=0.94) does not imply predicting the **direction** of a ±$2–5 daily move correctly. Direction prediction requires the model error to be smaller than the move itself — a harder problem. This result is consistent with the Efficient Market Hypothesis at the short horizon.

### Key Finding
The initial XGBoost implementation trained on absolute price features (raw lag prices, SMA, EMA) achieved R²=−2.33 on the test set despite strong validation performance — a symptom of distribution shift. Reengineering all features as returns and ratios completely resolved this failure, bringing XGBoost to R²=0.9461 and confirming that **feature scale-invariance is a prerequisite for generalizing tree models across crypto price regimes**.

---

## 8. References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785

2. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

3. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.

4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

5. Pandas Development Team. (2024). *pandas: Powerful Python data analysis toolkit* (v2.x). https://pandas.pydata.org

6. Aroussi, R. (2024). *yfinance: Download market data from Yahoo! Finance's API*. https://github.com/ranaroussi/yfinance

7. Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.

8. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

9. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
