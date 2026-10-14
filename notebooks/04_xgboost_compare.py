import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])

FEATURES = [
    "Store", "Dept", "Type", "Size", "IsHoliday", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "week", "month",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_12",
]

CUTOFF = pd.Timestamp("2012-07-27")
train = df[df["Date"] <= CUTOFF]
valid = df[df["Date"] > CUTOFF].dropna(subset=["lag_1", "lag_52"]).copy()

def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()

y = valid["Weekly_Sales"]
results = {}

# ---- LightGBM (same fixed params as Step 3) ----
lgbm = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)
lgbm.fit(train[FEATURES], train["Weekly_Sales"])
valid["pred_lgbm"] = lgbm.predict(valid[FEATURES]).clip(min=0)
results["LightGBM"] = valid["pred_lgbm"]

# ---- XGBoost (fixed params, no tuning on validation) ----
xgbm = xgb.XGBRegressor(
    n_estimators=600, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)
xgbm.fit(train[FEATURES], train["Weekly_Sales"])
valid["pred_xgb"] = xgbm.predict(valid[FEATURES]).clip(min=0)
results["XGBoost"] = valid["pred_xgb"]

# ---- Simple average of both ----
valid["pred_avg"] = (valid["pred_lgbm"] + valid["pred_xgb"]) / 2
results["Average (LGBM+XGB)"] = valid["pred_avg"]

results["Naive last week"] = valid["lag_1"]
results["Naive last year"] = valid["lag_52"]

print("\n=== Model comparison (13-week holdout) ===")
rows = []
for name, pred in results.items():
    rows.append({
        "Model": name,
        "WAPE": round(wape(y, pred), 4),
        "MAE": round(np.abs(y - pred).mean()),
    })
table = pd.DataFrame(rows).sort_values("WAPE")
print(table.to_string(index=False))
table.to_csv("data/processed/model_comparison.csv", index=False)

# ---- Where does the best model fail? (uses LightGBM) ----
print("\n=== WAPE by store type ===")
for t, name in {0: "A", 1: "B", 2: "C"}.items():
    sub = valid[valid["Type"] == t]
    print(f"Type {name}: WAPE = {wape(sub['Weekly_Sales'], sub['pred_lgbm']):.4f} ({len(sub)} rows)")

print("\n=== 5 worst departments (min 200 rows, by WAPE) ===")
rows = []
for d, sub in valid.groupby("Dept"):
    if len(sub) >= 200:
        rows.append((d, wape(sub["Weekly_Sales"], sub["pred_lgbm"]), sub["Weekly_Sales"].sum()))
worst = pd.DataFrame(rows, columns=["Dept", "WAPE", "Total_Sales"]).sort_values("WAPE", ascending=False)
print(worst.head(5).to_string(index=False))