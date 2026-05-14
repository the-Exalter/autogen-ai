"""
AutoGen Price Forecasting Pipeline
===================================
Trains depreciation curves and XGBoost time series models
on the vehicle-sales-data (car_prices.csv) dataset.

Steps:
1. Load and clean the dataset
2. Extract monthly time series per model
3. Fit log-linear depreciation curves per model
4. Train XGBoost time series model
5. Train Quantile Regression Forest for confidence intervals
6. Save all models and artifacts to autogen-ai/models/

Run: python3 train_forecasting.py --data /path/to/car_prices.csv

Output files in autogen-ai/models/forecasting/:
  - depreciation_curves.json     (fitted curves per model)
  - xgb_model.json               (XGBoost model)
  - qrf_model.pkl                 (Quantile RF model)
  - monthly_series.csv           (extracted time series)
  - model_stats.json             (evaluation metrics)
  - feature_names.json           (feature list for inference)
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────

TARGET_MODELS = [
    {"make": "Toyota",  "model": "Corolla"},
    {"make": "Honda",   "model": "Civic"},
    {"make": "Honda",   "model": "Accord"},
    {"make": "Toyota",  "model": "Camry"},
    {"make": "Ford",    "model": "F-150"},
    {"make": "Chevrolet", "model": "Silverado"},
    {"make": "Honda",   "model": "CR-V"},
    {"make": "Toyota",  "model": "RAV4"},
    {"make": "Nissan",  "model": "Altima"},
    {"make": "Ford",    "model": "Escape"},
]

OUTPUT_DIR = "autogen-ai/models/forecasting"
MIN_LISTINGS_PER_BUCKET = 5  # minimum listings to trust a price bucket
CURRENT_YEAR = 2026

# ── Step 1: Load and clean ───────────────────────────────────────────────────

def parse_saledate(date_str):
    """Parse saledate format: 'Tue Dec 16 2014 12:30:00 GMT-0800 (PST)'"""
    try:
        # Extract just the date part: "Tue Dec 16 2014"
        parts = str(date_str).strip().split()
        if len(parts) >= 4:
            date_part = f"{parts[1]} {parts[2]} {parts[3]}"
            return datetime.strptime(date_part, "%b %d %Y")
    except Exception:
        pass
    return None

def load_and_clean(csv_path):
    print(f"\n[1/5] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Raw rows: {len(df):,}")
    print(f"      Columns: {list(df.columns)}")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse sale date
    print("      Parsing sale dates...")
    df["sale_date"] = df["saledate"].apply(parse_saledate)
    df = df.dropna(subset=["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["year_month"] = df["sale_date"].dt.to_period("M")

    # Clean price
    df["price"] = pd.to_numeric(df["sellingprice"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 500]    # remove junk
    df = df[df["price"] < 200000] # remove obvious errors

    # Clean odometer
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    # Clean year
    df["vehicle_year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["vehicle_year"])
    df["vehicle_year"] = df["vehicle_year"].astype(int)
    df = df[df["vehicle_year"] >= 1990]
    df = df[df["vehicle_year"] <= CURRENT_YEAR]

    # Vehicle age at time of sale
    df["age_at_sale"] = df["sale_year"] - df["vehicle_year"]
    df = df[df["age_at_sale"] >= 0]
    df = df[df["age_at_sale"] <= 30]

    # Normalize make/model
    df["make"] = df["make"].str.strip().str.title()
    df["model"] = df["model"].str.strip().str.title()

    # Condition: keep numeric
    df["condition"] = pd.to_numeric(df["condition"], errors="coerce")

    print(f"      Clean rows: {len(df):,}")
    print(f"      Date range: {df['sale_date'].min().date()} to {df['sale_date'].max().date()}")

    return df

# ── Step 2: Extract monthly time series ─────────────────────────────────────

def extract_time_series(df):
    print("\n[2/5] Extracting monthly time series per model...")
    
    all_series = []
    
    for target in TARGET_MODELS:
        make = target["make"]
        model = target["model"]
        
        mask = (
            (df["make"].str.lower() == make.lower()) &
            (df["model"].str.lower() == model.lower())
        )
        sub = df[mask].copy()
        
        if len(sub) < 50:
            print(f"      {make} {model}: only {len(sub)} records — skipping")
            continue
        
        print(f"      {make} {model}: {len(sub):,} records")
        
        # Group by year_month — median price per month
        monthly = sub.groupby("year_month").agg(
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            listing_count=("price", "count"),
            median_age=("age_at_sale", "median"),
            median_odometer=("odometer", "median"),
        ).reset_index()
        
        # Only keep months with enough data
        monthly = monthly[monthly["listing_count"] >= MIN_LISTINGS_PER_BUCKET]
        monthly = monthly.sort_values("year_month")
        
        # Convert period to datetime for easier handling
        monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()
        monthly["month_index"] = range(len(monthly))
        monthly["make"] = make
        monthly["model"] = model
        
        # Rolling averages as features
        monthly["rolling_3m_price"] = monthly["median_price"].rolling(3, min_periods=1).mean()
        monthly["price_change_pct"] = monthly["median_price"].pct_change().fillna(0)
        
        all_series.append(monthly)
        print(f"        → {len(monthly)} monthly data points ({monthly['year_month'].min()} to {monthly['year_month'].max()})")
    
    if not all_series:
        raise ValueError("No models had sufficient data. Check your dataset.")
    
    combined = pd.concat(all_series, ignore_index=True)
    print(f"\n      Total monthly records: {len(combined):,}")
    return combined

# ── Step 3: Fit depreciation curves ─────────────────────────────────────────

def fit_depreciation_curves(df):
    """
    For each model, fit a log-linear depreciation curve:
    log(price) = intercept + slope * vehicle_age
    
    Using cohort data — what does a 2-year-old Corolla sell for
    vs a 5-year-old Corolla vs a 10-year-old Corolla.
    """
    print("\n[3/5] Fitting depreciation curves...")
    
    curves = {}
    
    for target in TARGET_MODELS:
        make = target["make"]
        model = target["model"]
        
        mask = (
            (df["make"].str.lower() == make.lower()) &
            (df["model"].str.lower() == model.lower()) &
            (df["age_at_sale"] >= 0) &
            (df["age_at_sale"] <= 20) &
            (df["price"] > 0)
        )
        sub = df[mask].copy()
        
        if len(sub) < 50:
            continue
        
        # Group by age — median price per age cohort
        by_age = sub.groupby("age_at_sale").agg(
            median_price=("price", "median"),
            count=("price", "count")
        ).reset_index()
        
        # Only use cohorts with enough data
        by_age = by_age[by_age["count"] >= MIN_LISTINGS_PER_BUCKET]
        
        if len(by_age) < 3:
            continue
        
        # Fit log-linear: log(price) ~ age
        X = by_age["age_at_sale"].values.reshape(-1, 1)
        y = np.log(by_age["median_price"].values)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # R-squared on log scale
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Annual depreciation rate
        # slope is change in log(price) per year
        # annual_depreciation = 1 - exp(slope)
        annual_depreciation_rate = 1 - np.exp(reg.coef_[0])
        
        # Price at age 0 (new car equivalent)
        new_price = np.exp(reg.intercept_)
        
        curves[f"{make}_{model}"] = {
            "make": make,
            "model": model,
            "intercept": float(reg.intercept_),
            "slope": float(reg.coef_[0]),
            "r2": float(r2),
            "annual_depreciation_rate": float(annual_depreciation_rate),
            "estimated_new_price_usd": float(new_price),
            "cohort_points": len(by_age),
            "total_records": len(sub),
        }
        
        print(f"      {make} {model}: "
              f"depreciation {annual_depreciation_rate:.1%}/yr, "
              f"R²={r2:.3f}, "
              f"est. new ${new_price:,.0f}")
    
    return curves

# ── Step 4: Train XGBoost ────────────────────────────────────────────────────

def train_xgboost(monthly_df, curves):
    """
    Train XGBoost to predict median monthly price.
    Features: age, mileage, month_index, rolling avg, 
              depreciation_slope, listing_count
    """
    print("\n[4/5] Training XGBoost time series model...")
    
    try:
        import xgboost as xgb
    except ImportError:
        print("      XGBoost not installed. Run: pip install xgboost")
        return None, None, []

    # Build feature matrix
    rows = []
    for _, row in monthly_df.iterrows():
        key = f"{row['make']}_{row['model']}"
        curve = curves.get(key, {})
        
        rows.append({
            "median_age": row.get("median_age", 5),
            "median_age_sq": row.get("median_age", 5) ** 2,
            "month_index": row["month_index"],
            "listing_count": row["listing_count"],
            "rolling_3m_price": row["rolling_3m_price"],
            "price_change_pct": np.clip(row["price_change_pct"], -0.5, 0.5),
            "depreciation_slope": curve.get("slope", -0.15),
            "median_odometer": row.get("median_odometer", 50000) if not pd.isna(row.get("median_odometer", np.nan)) else 50000,
            "target": row["median_price"],
        })
    
    feat_df = pd.DataFrame(rows)
    
    FEATURES = [
        "median_age", "median_age_sq", "month_index",
        "listing_count", "rolling_3m_price", "price_change_pct",
        "depreciation_slope", "median_odometer"
    ]
    
    X = feat_df[FEATURES]
    y = feat_df["target"]
    
    # Train/validation split — last 3 months of each model as holdout
    train_mask = feat_df.groupby(
        monthly_df["make"].astype(str) + "_" + monthly_df["model"].astype(str)
    ).cumcount(ascending=False) >= 3
    
    # Simple split: last 15% as validation
    split = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=20,
        verbosity=0,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred) * 100
    
    print(f"      XGBoost trained on {len(X_train)} samples")
    print(f"      Validation MAE: ${mae:,.0f}")
    print(f"      Validation MAPE: {mape:.1f}%")
    
    return model, {"mae": mae, "mape": mape}, FEATURES

# ── Step 5: Train Quantile RF ────────────────────────────────────────────────

def train_quantile_rf(monthly_df, curves, features):
    """
    Train Quantile Regression Forest for confidence intervals.
    Uses quantile-forest if available, falls back to sklearn RF.
    """
    print("\n[5/5] Training Quantile Regression Forest...")
    
    rows = []
    for _, row in monthly_df.iterrows():
        key = f"{row['make']}_{row['model']}"
        curve = curves.get(key, {})
        rows.append({
            "median_age": row.get("median_age", 5),
            "median_age_sq": row.get("median_age", 5) ** 2,
            "month_index": row["month_index"],
            "listing_count": row["listing_count"],
            "rolling_3m_price": row["rolling_3m_price"],
            "price_change_pct": np.clip(row["price_change_pct"], -0.5, 0.5),
            "depreciation_slope": curve.get("slope", -0.15),
            "median_odometer": row.get("median_odometer", 50000) if not pd.isna(row.get("median_odometer", np.nan)) else 50000,
            "target": row["median_price"],
        })
    
    feat_df = pd.DataFrame(rows)
    X = feat_df[features]
    y = feat_df["target"]
    
    split = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    try:
        from quantile_forest import RandomForestQuantileRegressor
        qrf = RandomForestQuantileRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        qrf.fit(X_train, y_train)
        qrf_type = "quantile_forest"
        print("      Using RandomForestQuantileRegressor")
    except ImportError:
        print("      quantile-forest not found, using standard RF")
        print("      Install with: pip install quantile-forest")
        qrf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        qrf.fit(X_train, y_train)
        qrf_type = "standard_rf"
    
    # Evaluate
    y_pred = qrf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred) * 100
    print(f"      QRF MAE: ${mae:,.0f}, MAPE: {mape:.1f}%")
    
    return qrf, qrf_type, {"mae": mae, "mape": mape}

# ── Forecast function (used by FastAPI) ─────────────────────────────────────

def project_price(make, model, current_age, horizon_months, 
                  curves, xgb_model, qrf_model, qrf_type, features,
                  current_price=None, odometer=None):
    """
    Project price at horizon_months into the future.
    Returns { median, lower, upper } in USD.
    """
    key = f"{make}_{model}"
    curve = curves.get(key)
    
    if not curve:
        return None
    
    future_age = current_age + (horizon_months / 12)
    
    # Depreciation curve projection
    log_price = curve["intercept"] + curve["slope"] * future_age
    curve_price = np.exp(log_price)
    
    if current_price:
        # Anchor to current market price rather than fitted new price
        current_log = curve["intercept"] + curve["slope"] * current_age
        current_fitted = np.exp(current_log)
        adjustment = current_price / current_fitted if current_fitted > 0 else 1.0
        curve_price *= adjustment
    
    # Build feature vector for XGBoost
    feat = {
        "median_age": future_age,
        "median_age_sq": future_age ** 2,
        "month_index": 100 + horizon_months,  # approximate future index
        "listing_count": 20,  # assumed typical
        "rolling_3m_price": current_price or curve_price,
        "price_change_pct": curve["slope"] / 12,  # monthly equivalent
        "depreciation_slope": curve["slope"],
        "median_odometer": (odometer or 50000) + (horizon_months * 1200),
    }
    
    feat_array = np.array([[feat[f] for f in features]])
    
    # XGBoost point estimate
    if xgb_model:
        try:
            xgb_price = float(xgb_model.predict(feat_array)[0])
        except Exception:
            xgb_price = curve_price
    else:
        xgb_price = curve_price
    
    # Blend curve and XGBoost (60/40 weight — curve anchors, XGB adjusts)
    blended = 0.6 * curve_price + 0.4 * xgb_price
    
    # Confidence intervals from QRF
    if qrf_model and qrf_type == "quantile_forest":
        try:
            preds = qrf_model.predict(feat_array, quantiles=[0.1, 0.5, 0.9])
            lower = float(preds[0][0])
            median = float(preds[0][1])
            upper = float(preds[0][2])
        except Exception:
            # Fallback: symmetric interval that widens with horizon
            spread = blended * (0.08 + horizon_months * 0.005)
            lower = blended - spread
            median = blended
            upper = blended + spread
    else:
        # Standard RF or no model: manual confidence interval
        # Interval widens with forecast horizon — this is statistically correct
        spread = blended * (0.08 + horizon_months * 0.005)
        lower = blended - spread
        median = blended
        upper = blended + spread
    
    return {
        "median_usd": round(median, 2),
        "lower_usd": round(max(lower, 100), 2),
        "upper_usd": round(upper, 2),
        "horizon_months": horizon_months,
        "annual_depreciation_rate": curve["annual_depreciation_rate"],
        "curve_r2": curve["r2"],
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoGen Forecasting Pipeline")
    parser.add_argument("--data", required=True, help="Path to car_prices.csv")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("AutoGen Price Forecasting Pipeline")
    print("=" * 60)
    
    # Step 1: Load
    df = load_and_clean(args.data)
    
    # Step 2: Time series
    monthly_df = extract_time_series(df)
    monthly_df.to_csv(f"{args.output}/monthly_series.csv", index=False)
    print(f"      Saved monthly_series.csv")
    
    # Step 3: Depreciation curves
    curves = fit_depreciation_curves(df)
    with open(f"{args.output}/depreciation_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"      Saved depreciation_curves.json ({len(curves)} models)")
    
    # Step 4: XGBoost
    xgb_model, xgb_stats, features = train_xgboost(monthly_df, curves)
    if xgb_model:
        xgb_model.save_model(f"{args.output}/xgb_model.json")
        with open(f"{args.output}/feature_names.json", "w") as f:
            json.dump(features, f)
        print(f"      Saved xgb_model.json")
    
    # Step 5: QRF
    qrf_model, qrf_type, qrf_stats = train_quantile_rf(
        monthly_df, curves, features
    )
    with open(f"{args.output}/qrf_model.pkl", "wb") as f:
        pickle.dump({"model": qrf_model, "type": qrf_type}, f)
    print(f"      Saved qrf_model.pkl")
    
    # Save stats
    stats = {
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": len(df),
        "models_trained": list(curves.keys()),
        "xgboost": xgb_stats,
        "qrf": qrf_stats,
        "feature_names": features,
    }
    with open(f"{args.output}/model_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Output directory: {args.output}/")
    print("\nModels trained:")
    for key, curve in curves.items():
        print(f"  {key}: {curve['annual_depreciation_rate']:.1%}/yr depreciation, R²={curve['r2']:.3f}")
    
    # Quick sanity check forecast
    print("\nSanity check — Toyota Corolla 5yr old, 3-month forecast:")
    result = project_price(
        "Toyota", "Corolla", 5, 3,
        curves, xgb_model, qrf_model, qrf_type, features,
        current_price=18000
    )
    if result:
        print(f"  Median: ${result['median_usd']:,.0f}")
        print(f"  Range:  ${result['lower_usd']:,.0f} — ${result['upper_usd']:,.0f}")

if __name__ == "__main__":
    main()
