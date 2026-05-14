"""
Train RandomForestRegressor models for the Pakistani and International
vehicle markets, saving model + encoders pickles per market.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

from utils import clean_price, clean_mileage, extract_make_model

CSV_PATH = os.environ.get("CSV_PATH", os.path.join(os.path.dirname(__file__), "../../PakWheels Dataset.csv"))
CSV_PATH_INTL = os.environ.get("CSV_PATH_INTL", os.path.join(os.path.dirname(__file__), "../../International Dataset.csv"))

MODEL_PATH_PK = os.path.join(os.path.dirname(__file__), "model_pk.pkl")
ENCODERS_PATH_PK = os.path.join(os.path.dirname(__file__), "encoders_pk.pkl")
MODEL_PATH_INTL = os.path.join(os.path.dirname(__file__), "model_intl.pkl")
ENCODERS_PATH_INTL = os.path.join(os.path.dirname(__file__), "encoders_intl.pkl")

CATEGORICAL = ["make", "fuel_type", "transmission", "body_type", "province", "assembly"]
CURRENT_YEAR = 2026


def load_and_clean(csv_path: str, price_col: str = "Price", price_field: str = "price_pkr",
                   price_min: float = 50_000, price_max: float = 500_000_000) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Price
    df[price_field] = df[price_col].apply(clean_price)
    df = df.dropna(subset=[price_field])
    df[price_field] = df[price_field].astype(float)
    df = df[(df[price_field] >= price_min) & (df[price_field] <= price_max)]

    # Year
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1970) & (df["year"] <= CURRENT_YEAR + 1)]

    # Mileage
    df["mileage_km"] = df["Millage"].apply(clean_mileage).fillna(0)

    # Make / model from nam
    parsed = df["nam"].apply(lambda x: extract_make_model(str(x)))
    df["make"] = parsed.apply(lambda t: t[0])

    # Other features
    df["fuel_type"] = df["Fuel"].fillna("Unknown")
    df["transmission"] = df["Transmission"].fillna("Unknown")
    df["body_type"] = df["Body Type"].fillna("Unknown")
    df["province"] = df["Province"].fillna("Unknown")
    df["assembly"] = df["Assembly"].fillna("Unknown")

    # Feature engineering
    df["vehicle_age"] = CURRENT_YEAR - df["year"]

    # Engine capacity numeric (strip 'cc')
    df["engine_cc"] = (
        df["Engine Capacity"]
        .fillna("0")
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )

    return df


def _train_and_save(df: pd.DataFrame, target_col: str, currency_label: str,
                    model_path: str, encoders_path: str) -> None:
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c + "_enc" for c in CATEGORICAL] + ["year", "vehicle_age", "mileage_km", "engine_cc"]
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestRegressor (n_estimators=200)…")
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {currency_label} {rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

    joblib.dump(model, model_path)
    joblib.dump(encoders, encoders_path)
    print(f"Saved model    → {model_path}")
    print(f"Saved encoders → {encoders_path}")


def train_pakistan():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"[PK] Loading and cleaning data from {CSV_PATH}…")
    df = load_and_clean(CSV_PATH, price_col="Price", price_field="price_pkr",
                        price_min=50_000, price_max=500_000_000)
    print(f"[PK] Clean rows: {len(df)}")
    _train_and_save(df, target_col="price_pkr", currency_label="PKR",
                    model_path=MODEL_PATH_PK, encoders_path=ENCODERS_PATH_PK)


def train_international():
    if not os.path.exists(CSV_PATH_INTL):
        print(f"International CSV not found at {CSV_PATH_INTL}")
        return

    print(f"[INTL] Loading and cleaning data from {CSV_PATH_INTL}…")
    df = load_and_clean(CSV_PATH_INTL, price_col="Price", price_field="price_usd",
                        price_min=500, price_max=2_000_000)
    print(f"[INTL] Clean rows: {len(df)}")
    _train_and_save(df, target_col="price_usd", currency_label="USD",
                    model_path=MODEL_PATH_INTL, encoders_path=ENCODERS_PATH_INTL)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        train_pakistan()
    else:
        print(f"Skipping PK training — CSV not found at {CSV_PATH}")

    if os.path.exists(CSV_PATH_INTL):
        train_international()
    else:
        print(f"Skipping INTL training — CSV not found at {CSV_PATH_INTL}")
