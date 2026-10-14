import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv(Path(__file__).resolve().parents[1] / "data" / ".env", override=True)

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
# Hopsworks wants lowercase column names
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"date": "date"})

FEATURE_COLS = [
    "store", "dept", "date", "type", "size", "isholiday", "temperature",
    "fuel_price", "markdown1", "markdown2", "markdown3", "markdown4",
    "markdown5", "cpi", "unemployment", "week", "month",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_12",
    "weekly_sales",
]
df = df[FEATURE_COLS]
key = os.environ["HOPSWORKS_API_KEY"]
print("Key length:", len(key), "| has dot:", "." in key, "| has quote/space:", ('"' in key) or ("'" in key) or (" " in key))
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="stocksense_weekly_features",
    version=1,
    description="Weekly store-department features and sales for Walmart demand forecasting",
    primary_key=["store", "dept", "date"],
    event_time="date",
)
fg.insert(df)
print("Inserted rows:", len(df))

# Read back to confirm
fv = fs.get_or_create_feature_view(
    name="stocksense_view",
    version=1,
    query=fg.select_all(),
)
back = fg.read()
print("Read back:", back.shape)