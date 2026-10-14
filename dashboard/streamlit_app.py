import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.main import lgbm, xgbm, serving, FEATURES  # noqa: E402

st.set_page_config(page_title="StockSense", layout="wide")
st.title("StockSense: Weekly Demand Forecasts")
st.caption("Walmart store sales, 13-week holdout (Aug to Oct 2012). Model = average of LightGBM and XGBoost.")

c1, c2 = st.columns(2)
store = c1.selectbox("Store", sorted(serving["store"].unique()))
depts = sorted(serving.loc[serving["store"] == store, "dept"].unique())
dept = c2.selectbox("Department", depts)

sub = serving[(serving["store"] == store) & (serving["dept"] == dept)].sort_values("date").copy()
X = sub[FEATURES]
sub["forecast"] = ((lgbm.predict(X) + xgbm.predict(X)) / 2).clip(min=0)
sub["naive_last_week"] = sub["lag_1"]

def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()

y = sub["weekly_sales"]
m1, m2, m3 = st.columns(3)
m1.metric("Model WAPE", f"{wape(y, sub['forecast']):.3f}")
m2.metric("Naive (last week) WAPE", f"{wape(y, sub['naive_last_week']):.3f}")
m3.metric("Weeks shown", len(sub))

st.line_chart(
    sub.set_index("date")[["weekly_sales", "forecast", "naive_last_week"]]
    .rename(columns={"weekly_sales": "Actual"})
)

st.subheader("Overall results on the full holdout (37,364 rows)")
st.table(pd.DataFrame({
    "Model": ["Average (LightGBM + XGBoost)", "Naive last week"],
    "WAPE": [0.0751, 0.0962],
    "MAE": [1209, 1549],
}))
st.caption("Known limitation: the holdout has only one holiday week, so Thanksgiving and Christmas are not tested.")