# StockSense: Weekly Retail Demand Forecasting

**Live dashboard:** https://stocksense-demand-forecasting-cnpffssfmyw2xjjugxuybh.streamlit.app/

A demand forecasting system for a retail chain (Walmart store sales data). It predicts next-week sales for each store and department, and it is built the way a real ML team would build it: time-based validation, a feature store, a served API, automated tests and CI.

## Result

On a 13-week holdout (37,364 store-department-week rows), the model cut forecast error by about **22%** compared with the best naive baseline.

| Model | WAPE | MAE |
|---|---|---|
| **Average of LightGBM + XGBoost** | **0.0751** | **1,209** |
| XGBoost | 0.0758 | n/a |
| LightGBM | 0.0765 | n/a |
| Naive: same as last week | 0.0962 | 1,549 |
| Naive: same week last year | 0.1068 | 1,720 |

WAPE is total absolute error divided by total actual sales (lower is better). The three models are close to each other, so on 13 weeks of data I do not claim that one beats the others. All three beat the baselines by 20 to 22%.

## Data

Kaggle "Walmart Recruiting - Store Sales Forecasting": 45 stores, 81 departments, 421,570 weekly rows from 2010-02-05 to 2012-10-26, joined with store type and size, temperature, fuel price, markdowns, CPI and unemployment.

## Approach

1. **Cleaning (decisions made from the data):**
   - 1,285 negative sales rows are returns, so they are clipped to 0.
   - MarkDown1 to 5 are missing before Nov 2011 because promotions data starts then, so missing means "no promotion" and is filled with 0.
   - CPI and Unemployment (585 missing each) are forward-filled within each store.
2. **Time-based split, no random split:** train on weeks up to 2012-07-27 (130 weeks) and validate on the last 13 weeks. A random split would let the model see the future.
3. **Baselines first:** "same as last week" and "same week last year". The model has to beat these.
4. **Features, all built from the past only:** lags (1, 2, 4, 52 weeks), rolling means (4 and 12 weeks, shifted by one week so the target week is excluded), calendar features, holiday flag, markdowns and economic indicators.
5. **Models:** LightGBM and XGBoost with fixed parameters (not tuned on the validation set), averaged into a final forecast.
6. **Feature store:** the engineered features are stored in a Hopsworks feature group (421,570 rows). The final models were trained from data read back from Hopsworks, and the results matched the local run within 0.0001 WAPE.
7. **Serving:** a FastAPI endpoint plus a Streamlit dashboard.

## What I tested

15 automated tests (`pytest`), all passing: health check, response fields, non-negative forecasts, consistent error values, repeatable answers, and 7 bad-input cases (date outside range, unknown store, unknown department, invalid date format, missing parameter, non-integer store, negative store). One test also scores 300 random rows through the API: WAPE 0.1052 for the model against 0.1260 for the naive baseline. That sample is small, so the headline numbers above come from the full holdout.

CI runs the tests on every push, then builds the Docker image and checks the `/health` endpoint.

## Where the model is weak

- **Store type B has the highest error** (WAPE 0.0879, against 0.0731 for type A and 0.0570 for type C).
- **Holiday weeks are harder:** WAPE 0.0854 against 0.0757 for non-holiday weeks (LightGBM). The holdout contains only one holiday week, so Thanksgiving and Christmas, the hardest weeks, were **not tested**.
- **Some departments fail.** Dept 54 has a very high error but only about $9.7k of sales. Dept 58 ($1.77M sales, WAPE 0.37) and Dept 36 (WAPE 0.42) carry real revenue and need their own model.
- **One 13-week window only.** I did not run walk-forward validation across several windows.
- **The API replays the holdout.** It returns the forecast and the actual sales for weeks in the holdout. It does not forecast unseen future weeks.
- Feature importance counts in the training logs measure how often a feature was split on, not its real effect, so I draw no conclusions from them.

## Problems I hit and fixed

| Issue | Fix |
|---|---|
| Hopsworks would not install on Python 3.14 | Created a Python 3.12 environment |
| Windows could not create the `/tmp` certificate folder | Created `C:\tmp` |
| Feature group creation failed: Delta library missing | Installed `hopsworks[python]` |
| Inserting data failed on Windows (HDFS connection dropped) | Ran the ingestion from Google Colab (Linux) |
| `.gitignore` was missing, so the virtual environment and 61 MB of data were committed and GitHub rejected the push | Recreated the Git history with a correct `.gitignore` and checked file sizes before pushing |

## Project structure

```
app/          FastAPI service and serving data
dashboard/    Streamlit dashboard
models/       Saved LightGBM and XGBoost models
notebooks/    Exploration, baseline, model training, comparison, Hopsworks ingestion
src/          Serving data builder
tests/        15 pytest tests
```

## Run it

```
python -m pip install -r requirements.txt
python -m pytest tests -v
uvicorn app.main:app --reload                       # API docs at /docs
python -m streamlit run dashboard/streamlit_app.py  # dashboard
```

Example request: `/forecast?store=1&dept=1&date=2012-08-03`

## What I would improve with more time

- Use walk-forward validation across several windows, and include the Thanksgiving and Christmas weeks.
- Train a separate model for the high-revenue departments that fail (36, 58).
- Add a multi-week forecast horizon and prediction intervals.
- Move retraining into a scheduled pipeline that reads from the feature store.
- Add monitoring for forecast drift.

## Tools

Python, Pandas, NumPy, LightGBM, XGBoost, scikit-learn, Hopsworks, FastAPI, Streamlit, pytest, Docker, GitHub Actions.
