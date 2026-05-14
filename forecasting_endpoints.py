"""
AutoGen Forecasting Endpoints
Add these to autogen-ai/main.py

These endpoints serve price forecasts using the trained models.
Models are loaded once on startup from autogen-ai/models/forecasting/
"""

# ── Add these imports to main.py ─────────────────────────────────────────────

import json
import pickle
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

# ── Load models on startup (add after existing startup code) ─────────────────

MODELS_DIR = Path(__file__).parent / "models" / "forecasting"

# Global model store
_forecast_models = {
    "curves": {},
    "xgb": None,
    "qrf": None,
    "qrf_type": "standard_rf",
    "features": [],
    "stats": {},
    "loaded": False,
}

def load_forecast_models():
    """Load all forecasting models from disk. Called on startup."""
    global _forecast_models
    
    try:
        # Depreciation curves
        curves_path = MODELS_DIR / "depreciation_curves.json"
        if curves_path.exists():
            with open(curves_path) as f:
                _forecast_models["curves"] = json.load(f)
            print(f"[Forecast] Loaded {len(_forecast_models['curves'])} depreciation curves")
        
        # Feature names
        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                _forecast_models["features"] = json.load(f)
        
        # XGBoost model
        xgb_path = MODELS_DIR / "xgb_model.json"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                _forecast_models["xgb"] = model
                print("[Forecast] XGBoost model loaded")
            except ImportError:
                print("[Forecast] XGBoost not installed — using curve-only forecasts")
        
        # QRF model
        qrf_path = MODELS_DIR / "qrf_model.pkl"
        if qrf_path.exists():
            with open(qrf_path, "rb") as f:
                qrf_data = pickle.load(f)
                _forecast_models["qrf"] = qrf_data.get("model")
                _forecast_models["qrf_type"] = qrf_data.get("type", "standard_rf")
            print(f"[Forecast] QRF model loaded (type: {_forecast_models['qrf_type']})")
        
        # Stats
        stats_path = MODELS_DIR / "model_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                _forecast_models["stats"] = json.load(f)
        
        _forecast_models["loaded"] = bool(_forecast_models["curves"])
        
        if not _forecast_models["loaded"]:
            print("[Forecast] No models found — run train_forecasting.py first")
    
    except Exception as e:
        print(f"[Forecast] Failed to load models: {e}")

# Call this in your FastAPI startup event:
# @app.on_event("startup")
# async def startup():
#     load_forecast_models()

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    make: str
    model: str
    vehicle_age: float           # current age in years
    current_price_usd: Optional[float] = None  # current asking/market price
    odometer_miles: Optional[float] = None     # current mileage in miles

class ForecastResponse(BaseModel):
    make: str
    model: str
    vehicle_age: float
    annual_depreciation_rate: float
    curve_r2: float
    forecasts: dict              # keyed by "3m", "6m", "12m"
    supported: bool
    message: str

class DepreciationResponse(BaseModel):
    make: str
    model: str
    annual_depreciation_rate: float
    estimated_new_price_usd: float
    curve_r2: float
    cohort_points: int

# ── Helper: project price ────────────────────────────────────────────────────

def _project(make, model, current_age, horizon_months, current_price=None, odometer=None):
    """Project price at a given horizon."""
    curves = _forecast_models["curves"]
    xgb_model = _forecast_models["xgb"]
    qrf_model = _forecast_models["qrf"]
    qrf_type = _forecast_models["qrf_type"]
    features = _forecast_models["features"]
    
    key = f"{make}_{model}"
    # Try case-insensitive match
    if key not in curves:
        for k in curves:
            if k.lower() == key.lower():
                key = k
                break
        else:
            return None
    
    curve = curves[key]
    future_age = current_age + (horizon_months / 12)
    
    # Curve projection
    log_price = curve["intercept"] + curve["slope"] * future_age
    curve_price = np.exp(log_price)
    
    if current_price:
        current_log = curve["intercept"] + curve["slope"] * current_age
        current_fitted = np.exp(current_log)
        if current_fitted > 0:
            curve_price *= (current_price / current_fitted)
    
    # XGBoost
    if xgb_model and features:
        odometer_val = (odometer or 50000) + (horizon_months * 1200)
        feat = {
            "median_age": future_age,
            "median_age_sq": future_age ** 2,
            "month_index": 100 + horizon_months,
            "listing_count": 20,
            "rolling_3m_price": current_price or curve_price,
            "price_change_pct": curve["slope"] / 12,
            "depreciation_slope": curve["slope"],
            "median_odometer": odometer_val,
        }
        feat_array = np.array([[feat.get(f, 0) for f in features]])
        try:
            xgb_price = float(xgb_model.predict(feat_array)[0])
        except Exception:
            xgb_price = curve_price
        blended = 0.6 * curve_price + 0.4 * xgb_price
    else:
        blended = curve_price
    
    # Confidence intervals
    if qrf_model and qrf_type == "quantile_forest" and features:
        try:
            feat_array = np.array([[{
                "median_age": future_age,
                "median_age_sq": future_age ** 2,
                "month_index": 100 + horizon_months,
                "listing_count": 20,
                "rolling_3m_price": current_price or curve_price,
                "price_change_pct": curve["slope"] / 12,
                "depreciation_slope": curve["slope"],
                "median_odometer": (odometer or 50000) + (horizon_months * 1200),
            }.get(f, 0) for f in features]])
            preds = qrf_model.predict(feat_array, quantiles=[0.1, 0.5, 0.9])
            lower = float(preds[0][0])
            median = float(preds[0][1])
            upper = float(preds[0][2])
        except Exception:
            spread = blended * (0.08 + horizon_months * 0.005)
            lower, median, upper = blended - spread, blended, blended + spread
    else:
        spread = blended * (0.08 + horizon_months * 0.005)
        lower, median, upper = blended - spread, blended, blended + spread
    
    change_pct = ((median - (current_price or blended)) / (current_price or blended)) * 100 if current_price else None
    
    return {
        "median_usd": round(median, 2),
        "lower_usd": round(max(lower, 100), 2),
        "upper_usd": round(upper, 2),
        "change_pct": round(change_pct, 1) if change_pct is not None else None,
        "annual_depreciation_rate": curve["annual_depreciation_rate"],
        "curve_r2": curve["r2"],
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────

# POST /forecast
# Returns 3-month, 6-month, and 12-month price forecasts
# with confidence intervals

"""
@app.post("/forecast", response_model=ForecastResponse)
async def forecast_price(req: ForecastRequest):
    if not _forecast_models["loaded"]:
        return ForecastResponse(
            make=req.make, model=req.model,
            vehicle_age=req.vehicle_age,
            annual_depreciation_rate=0,
            curve_r2=0,
            forecasts={},
            supported=False,
            message="Forecasting models not loaded. Run train_forecasting.py first."
        )
    
    # Check if this make/model is supported
    key_lower = f"{req.make}_{req.model}".lower()
    supported = any(k.lower() == key_lower for k in _forecast_models["curves"])
    
    if not supported:
        available = list(_forecast_models["curves"].keys())
        return ForecastResponse(
            make=req.make, model=req.model,
            vehicle_age=req.vehicle_age,
            annual_depreciation_rate=0,
            curve_r2=0,
            forecasts={},
            supported=False,
            message=f"Model not supported. Available: {available}"
        )
    
    # Get curve for metadata
    key = next(k for k in _forecast_models["curves"] if k.lower() == key_lower)
    curve = _forecast_models["curves"][key]
    
    # Generate forecasts for 3, 6, 12 month horizons
    forecasts = {}
    for horizon in [3, 6, 12]:
        result = _project(
            req.make, req.model,
            req.vehicle_age, horizon,
            req.current_price_usd,
            req.odometer_miles
        )
        if result:
            forecasts[f"{horizon}m"] = result
    
    return ForecastResponse(
        make=req.make,
        model=req.model,
        vehicle_age=req.vehicle_age,
        annual_depreciation_rate=curve["annual_depreciation_rate"],
        curve_r2=curve["r2"],
        forecasts=forecasts,
        supported=True,
        message="ok"
    )


# GET /forecast/depreciation?make=Toyota&model=Corolla
# Returns the fitted depreciation curve for a model

@app.get("/forecast/depreciation")
async def get_depreciation(make: str, model: str):
    if not _forecast_models["loaded"]:
        return {"error": "Models not loaded"}
    
    key_lower = f"{make}_{model}".lower()
    key = next((k for k in _forecast_models["curves"] 
                if k.lower() == key_lower), None)
    
    if not key:
        return {
            "supported": False,
            "available": list(_forecast_models["curves"].keys())
        }
    
    curve = _forecast_models["curves"][key]
    
    # Generate curve points for chart — age 0 to 15 years
    points = []
    for age in range(0, 16):
        log_price = curve["intercept"] + curve["slope"] * age
        price = np.exp(log_price)
        points.append({"age": age, "median_price_usd": round(price, 2)})
    
    return {
        "supported": True,
        "make": make,
        "model": model,
        "annual_depreciation_rate": curve["annual_depreciation_rate"],
        "estimated_new_price_usd": curve["estimated_new_price_usd"],
        "curve_r2": curve["r2"],
        "curve_points": points,
    }


# GET /forecast/supported
# Returns list of supported make/model combinations

@app.get("/forecast/supported")
async def get_supported_models():
    if not _forecast_models["loaded"]:
        return {"loaded": False, "models": []}
    
    models = []
    for key, curve in _forecast_models["curves"].items():
        models.append({
            "make": curve["make"],
            "model": curve["model"],
            "annual_depreciation_rate": curve["annual_depreciation_rate"],
            "r2": curve["r2"],
        })
    
    return {
        "loaded": True,
        "count": len(models),
        "models": models,
        "trained_at": _forecast_models["stats"].get("trained_at"),
    }


# GET /market/health  
# Returns overall market conditions based on recent price trends

@app.get("/market/health")
async def market_health():
    if not _forecast_models["loaded"]:
        return {"loaded": False}
    
    curves = _forecast_models["curves"]
    
    # Compute average annual depreciation across all models
    rates = [c["annual_depreciation_rate"] for c in curves.values()]
    avg_rate = np.mean(rates) if rates else 0
    
    # Classify market health
    if avg_rate < 0.10:
        health = "strong"
        message = "Vehicle values holding well — low depreciation market"
    elif avg_rate < 0.18:
        health = "neutral"
        message = "Average depreciation — balanced market conditions"
    else:
        health = "soft"
        message = "Higher than average depreciation — buyer's market"
    
    return {
        "health": health,
        "message": message,
        "average_annual_depreciation": round(avg_rate, 4),
        "models_analyzed": len(curves),
    }
"""
