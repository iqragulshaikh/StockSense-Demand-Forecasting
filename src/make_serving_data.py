import pandas as pd

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
df.columns = [c.lower() for c in df.columns]

CUTOFF = pd.Timestamp("2012-07-27")
serving = df[df["date"] > CUTOFF].dropna(subset=["lag_1", "lag_52"])
serving.to_csv("app/serving_data.csv", index=False)
print("Serving rows:", len(serving), "| Dates:", serving["date"].min(), "to", serving["date"].max())