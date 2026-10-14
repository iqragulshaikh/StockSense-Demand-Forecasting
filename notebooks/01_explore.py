import pandas as pd

train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
features = pd.read_csv("data/raw/features.csv", parse_dates=["Date"])
stores = pd.read_csv("data/raw/stores.csv")

print(train.shape, features.shape, stores.shape)
print(train.head())
print("Date range:", train["Date"].min(), "to", train["Date"].max())
print("Stores:", train["Store"].nunique(), "| Depts:", train["Dept"].nunique())
print("Negative sales rows:", (train["Weekly_Sales"] < 0).sum())
print("Missing values in features:")
print(features.isna().sum())

df = train.merge(stores, on="Store").merge(
    features.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left"
)
print(df.shape)