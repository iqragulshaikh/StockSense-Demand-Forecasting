import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib, os

df = pd.read_csv("data/processed/clean.csv", parse_dates=["Date"])
df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

key = ["Store", "Dept", "Date"]
sales = df[key + ["Weekly_Sales"]]

# ---------- Lag features (only from the past) ----------
def add_lag(frame, days, name):
    s = sales.copy()
    s["Date"] = s["Date"] + pd.Timedelta(days=days)
    s = s.rename(columns={"Weekly_Sales": name})
    return frame.merge(s, on=key, how="left")

for weeks in [1, 2, 4, 52]:
    df = add_lag(df, 7 * weeks, f"lag_{weeks}")

# ---------- Rolling means (shift(1) so current week is excluded) ----------
g = df.groupby(["Store", "Dept"])["Weekly_Sales"]
df["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
df["roll_mean_12"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())

# ---------- Calendar and encoding ----------
df["week"] = df["Date"].dt.isocalendar().week.astype(int)
df["month"] = df["Date"].dt.month
df["IsHoliday"] = df["IsHoliday"].astype(int)
df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

FEATURES = [
    "Store", "Dept", "Type", "Size", "IsHoliday", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "week", "month",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_12",
]

# ---------- Time-based split (same cutoff as the baseline) ----------
CUTOFF = pd.Timestamp("2012-07-27")
train = df[df["Date"] <= CUTOFF]
valid = df[df["Date"] > CUTOFF]

# Same rows as the baseline comparison (fair)
valid = valid.dropna(subset=["lag_1", "lag_52"]).copy()
print("Train rows:", len(train), "| Valid rows:", len(valid))

# ---------- Train (fixed parameters; no tuning on validation) ----------
model = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)
model.fit(train[FEATURES], train["Weekly_Sales"])

valid["pred"] = model.predict(valid[FEATURES]).clip(min=0)

def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()

y = valid["Weekly_Sales"]
print(f"LightGBM:        WAPE = {wape(y, valid['pred']):.4f} | MAE = {np.abs(y - valid['pred']).mean():,.0f}")
print(f"naive_last_week: WAPE = {wape(y, valid['lag_1']):.4f} | MAE = {np.abs(y - valid['lag_1']).mean():,.0f}")
print(f"naive_last_year: WAPE = {wape(y, valid['lag_52']):.4f} | MAE = {np.abs(y - valid['lag_52']).mean():,.0f}")

# ---------- Where does it fail? ----------
hol = valid[valid["IsHoliday"] == 1]
non = valid[valid["IsHoliday"] == 0]
print(f"Holiday weeks:     WAPE = {wape(hol['Weekly_Sales'], hol['pred']):.4f}")
print(f"Non-holiday weeks: WAPE = {wape(non['Weekly_Sales'], non['pred']):.4f}")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Top features:")
print(imp.head(8))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/lgbm_baseline.pkl")
df.to_csv("data/processed/features.csv", index=False)