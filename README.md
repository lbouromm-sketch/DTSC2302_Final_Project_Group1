# Forecasting Fed Rate Expectations with CPI and Jobs Data

**Group:** Joseph Nounagnon · Luxor Bourommavong · Ryan Port · Yovany Romero Gomez

## Research Question
Can machine learning models use monthly inflation (CPI) and jobs (Nonfarm Payrolls, NFP) data to forecast changes in the **2-year Treasury yield** better than a random walk and a classical time-series model? Does that forecasting value change across Fed policy regimes?

## Background
The Federal Reserve sets the federal funds rate to pursue maximum employment and stable prices. Predicting the Fed's decision directly (hike, hold, or cut) is impractical for three reasons:
- **Small sample:** there are only 8 meetings a year.
- **Mostly holds:** most decisions are "hold."
- **Already priced in:** fed funds futures reflect the outcome before most meetings.

Instead, we forecast the **2-year Treasury yield**. It moves daily and roughly tracks the average Fed rate that markets expect over the next two years.

**Hawkish vs. dovish.** These terms describe which way the Fed leans:
- **Hawkish:** focused on fighting inflation, so it favors **raising rates** or keeping them high. A hotter-than-expected CPI report usually makes markets expect a more hawkish Fed, and the 2-year yield **rises**.
- **Dovish:** focused on supporting jobs and growth, so it favors **cutting rates** or keeping them low. A weaker-than-expected jobs report usually makes markets expect a more dovish Fed, and the 2-year yield **falls**.

**Key terms:**
- **CPI (Consumer Price Index):** measures the monthly change in prices paid by consumers.
- **NFP (Nonfarm Payrolls):** the number of jobs added in the U.S. economy each month, excluding farm workers.

## Data
| Variable | FRED series | Frequency |
|---|---|---|
| 2-year Treasury yield (target) | `DGS2` | Daily → month-end |
| Headline CPI | `CPIAUCSL` (first release, via ALFRED) | Monthly |
| Core CPI | `CPILFESL` (first release, via ALFRED) | Monthly |
| Nonfarm Payrolls | `PAYEMS` (first release, via ALFRED) | Monthly |
| Initial jobless claims | `ICSA` | Weekly → monthly |
| Fed target rate (upper bound) | `DFEDTARU` | Daily → month-end |

- **Sample:** January 2000 – December 2025 (about 310 months).
- **First-release values:** we use the numbers as originally published, not today's revised data, so each model sees only what markets knew at the time.

## Method

### Target and features
- **Target:** change in the 2-year Treasury yield over month *t*, in basis points.
- **Forecast date:** the last trading day of month *t−1*.
- **Features, all known by the forecast date:**
  - Yield change in months *t−1* and *t−2*
  - Headline and core CPI, month over month (first release, lag 2)
  - NFP monthly change (first release, lag 2)
  - Initial jobless claims, 4-week average (lag 1)
  - Fed target rate level, plus a zero-rate indicator (lag 1)

### Preventing data leakage
- **Release timing:** CPI and NFP for a month are published in the *following* month. Data for month *t−1* is not available at the end of month *t−1*, so these features use lag 2.
- **First-release values:** we take the numbers as first published (ALFRED), never today's revised data.
- **Order is preserved:** no shuffling at any step. All tuning uses training data only.

### Train/test strategy
- **Training and validation:** January 2000 – December 2017. Hyperparameters are tuned with expanding-window `TimeSeriesSplit` (5 folds), so each validation fold comes after the data it was trained on.
- **Holdout test:** January 2018 – December 2025, untouched until final evaluation. It covers the zero-rate period, the 2022–23 hiking cycle, and the 2024–25 cuts.
- **Forecasting the holdout:** one step at a time, walk-forward. Each month, refit on all data up to the forecast date and forecast the next month.

### Baselines
- **Random walk:** forecasts zero change. This is the main baseline.
- **Historical mean:** forecasts the average monthly change in the training data.
- **ARIMA:** a classical time-series benchmark the ML models should beat.

### Machine learning models
1. **Random Forest regressor.** Averages many independent decision trees, each trained on a random sample of rows and features. It captures nonlinear effects and interactions (for example, CPI mattering more when the Fed is hiking) and resists overfitting on small samples. Tuned: number of trees, maximum depth, minimum samples per leaf.
2. **Gradient Boosting (XGBoost).** Builds trees one after another, each correcting the previous trees' errors. It is often the strongest model on tabular data, but more prone to overfitting, so it uses shallow trees, a low learning rate, and early stopping on the validation folds.

### Evaluation
- **RMSE and MAE** against each baseline.
- **Directional accuracy:** how often the model predicts the correct direction of the yield move, compared with 50%.
- **Diebold-Mariano test:** whether each model's gain over the random walk is statistically significant.
- **Regime breakdown (H1, H2):** metrics and feature importance (SHAP), split by period: zero-rate years vs. active hiking or cutting cycles, and pre-2021 vs. post-2021.

## Hypothesis
Averaged over the full holdout period, we expect CPI and jobs data to add little forecasting value over a random walk, because markets price new information quickly.

However, we expect the value to **depend on the policy regime**:
- **H1:** CPI and NFP improve forecasts (lower RMSE, higher directional accuracy) during active hiking or cutting cycles, but not during the near-zero-rate periods (2009–2015, 2020–2022).
- **H2:** CPI becomes the stronger predictor after 2021, when inflation became the Fed's main focus, while NFP matters more in earlier periods.

## Limitations
- **Noisy proxy:** the 2-year yield measures Fed expectations with noise. It also reflects term premium and flight-to-safety flows.
- **Little movement at zero rates:** during the near-zero-rate years, the yield had little room to move.
- **Small sample:** about 310 monthly observations is small for machine learning. We limit the number of features and constrain tree depth to reduce overfitting.
- **Regime dependence:** model performance may differ across rate regimes (Malliaris & Malliaris, 2009).

## Notebook Structure
1. Data collection (FRED / ALFRED)
2. Cleaning and aligning to monthly frequency
3. Feature engineering (lags, release-timing alignment)
4. Exploratory analysis
5. Baselines (random walk, historical mean, ARIMA)
6. ML models (Random Forest, XGBoost): tuning with `TimeSeriesSplit`, walk-forward forecasts
7. Evaluation, regime breakdown, and conclusions
