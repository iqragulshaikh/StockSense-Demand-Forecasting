from pathlib import Path
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "store", "dept", "type", "size", "isholiday", "temperature", "fuel_price",
    "markdown1", "markdown2", "markdown3", "markdown4", "markdown5",
    "cpi", "unemployment", "week", "month",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_12",
]

lgbm = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_final.txt"))
xgbm = xgb.XGBRegressor()
xgbm.load_model(str(ROOT / "models" / "xgb_final.json"))
serving = pd.read_csv(ROOT / "app" / "serving_data.csv", parse_dates=["date"])

app = FastAPI(title="StockSense Forecast API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(serving)}


@app.get("/forecast")
def forecast(store: int, dept: int, date: str):
    try:
        d = pd.Timestamp(date)
    except Exception:
        raise HTTPException(status_code=422, detail="date must look like 2012-08-03")

    row = serving[(serving["store"] == store) & (serving["dept"] == dept) & (serving["date"] == d)]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for store={store}, dept={dept}, date={date}. "
                   f"Available dates: {serving['date'].min().date()} to {serving['date'].max().date()}",
        )

    X = row[FEATURES]
    pred = max(0.0, float((lgbm.predict(X)[0] + xgbm.predict(X)[0]) / 2))
    actual = float(row["weekly_sales"].iloc[0])
    return {
        "store": store,
        "dept": dept,
        "date": date,
        "forecast": round(pred, 2),
        "actual_sales": round(actual, 2),
        "abs_error": round(abs(pred - actual), 2),
    }