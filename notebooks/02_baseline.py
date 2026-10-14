import os
import pandas as pd
import numpy as np

# ---------- Load & merge ----------
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
features = pd.read_csv("data/raw/features.csv", parse_dates=["Date"])
stores = pd.read_csv("data/raw/stores.csv")

df = train.merge(stores, on="Store").merge(
    features.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left"
)

# ---------- Cleaning (decisions documented in README) ----------
df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)          # returns -> 0
md_cols = [f"MarkDown{i}" for i in range(1, 6)]
df[md_cols] = df[md_cols].fillna(0)                            # no promo = 0
df = df.sort_values(["Store", "Date"])
df[["CPI", "Unemployment"]] = df.groupby("Store")[["CPI", "Unemployment"]].ffill()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/clean.csv", index=False)

# ---------- Time-based split (NO random split) ----------
CUTOFF = pd.Timestamp("2012-07-27")
history = df[df["Date"] <= CUTOFF]
valid = df[df["Date"] > CUTOFF].copy()
print("Train weeks:", history["Date"].nunique(), "| Valid weeks:", valid["Date"].nunique())

# ---------- Baselines (use only past weeks) ----------
key = ["Store", "Dept", "Date"]
sales = df[key + ["Weekly_Sales"]]

def shifted(days, name):
    s = sales.copy()
    s["Date"] = s["Date"] + pd.Timedelta(days=days)
    return s.rename(columns={"Weekly_Sales": name})

valid = valid.merge(shifted(7, "naive_last_week"), on=key, how="left")
valid = valid.merge(shifted(364, "naive_last_year"), on=key, how="left")

n_all = len(valid)
valid = valid.dropna(subset=["naive_last_week", "naive_last_year"])
print(f"Rows compared: {len(valid)} of {n_all}")

def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()

def mae(y, p):
    return np.abs(y - p).mean()

y = valid["Weekly_Sales"]
for name in ["naive_last_week", "naive_last_year"]:
    print(f"{name}: WAPE = {wape(y, valid[name]):.4f} | MAE = {mae(y, valid[name]):,.0f}")