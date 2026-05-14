import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

from ai_agent import generate_vehicle, detect_availability, stream_vehicle
from rag_context import get_rag_context

import json
import pickle
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

load_dotenv()

MODEL_PATH_PK = os.path.join(os.path.dirname(__file__), "model_pk.pkl")
ENCODERS_PATH_PK = os.path.join(os.path.dirname(__file__), "encoders_pk.pkl")
MODEL_PATH_INTL = os.path.join(os.path.dirname(__file__), "model_intl.pkl")
ENCODERS_PATH_INTL = os.path.join(os.path.dirname(__file__), "encoders_intl.pkl")

CATEGORICAL = ["make", "fuel_type", "transmission", "body_type", "province", "assembly"]
CURRENT_YEAR = 2026

ml_model_pk = None
encoders_pk = None
ml_model_intl = None
encoders_intl = None

MODELS_DIR = Path(__file__).parent / "models" / "forecasting"

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
        curves_path = MODELS_DIR / "depreciation_curves.json"
        if curves_path.exists():
            with open(curves_path) as f:
                _forecast_models["curves"] = json.load(f)
            print(f"[Forecast] Loaded {len(_forecast_models['curves'])} depreciation curves")

        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                _forecast_models["features"] = json.load(f)

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

        qrf_path = MODELS_DIR / "qrf_model.pkl"
        if qrf_path.exists():
            with open(qrf_path, "rb") as f:
                qrf_data = pickle.load(f)
                _forecast_models["qrf"] = qrf_data.get("model")
                _forecast_models["qrf_type"] = qrf_data.get("type", "standard_rf")
            print(f"[Forecast] QRF model loaded (type: {_forecast_models['qrf_type']})")

        stats_path = MODELS_DIR / "model_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                _forecast_models["stats"] = json.load(f)

        _forecast_models["loaded"] = bool(_forecast_models["curves"])

        if not _forecast_models["loaded"]:
            print("[Forecast] No models found — run train_forecasting.py first")

    except Exception as e:
        print(f"[Forecast] Failed to load models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model_pk, encoders_pk, ml_model_intl, encoders_intl
    if os.path.exists(MODEL_PATH_PK) and os.path.exists(ENCODERS_PATH_PK):
        ml_model_pk = joblib.load(MODEL_PATH_PK)
        encoders_pk = joblib.load(ENCODERS_PATH_PK)
        print("PK ML model loaded.")
    else:
        print("WARNING: model_pk.pkl not found. Run train.py first. PK price prediction will be unavailable.")

    if os.path.exists(MODEL_PATH_INTL) and os.path.exists(ENCODERS_PATH_INTL):
        ml_model_intl = joblib.load(MODEL_PATH_INTL)
        encoders_intl = joblib.load(ENCODERS_PATH_INTL)
        print("International ML model loaded.")
    else:
        print("WARNING: model_intl.pkl not found. International price prediction will be unavailable.")

    load_forecast_models()
    yield


app = FastAPI(title="AutoGen AI Microservice", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class GenerateVehicleRequest(BaseModel):
    make: str
    model: str
    year: int | None = None
    variant: str | None = None


class MaintenanceIntervals(BaseModel):
    oil_change_km: int
    timing_belt_km: int
    major_service_km: int


class VehicleKnowledgeCard(BaseModel):
    make: str
    model: str
    year: int | None = None
    variant: str | None = None
    body_type: str
    fuel_type: str
    engine_capacity: str
    transmission: str
    color: str | None = None
    assembly: str | None = None
    typical_price_pkr: int | None = None
    description: str
    features: list[str]
    known_issues: list[str]
    maintenance_intervals: MaintenanceIntervals
    parts_availability: str
    buying_checklist: list[str]
    market_position: str


class PredictPriceRequest(BaseModel):
    make: str
    model: str
    year: int
    mileage_km: float
    fuel_type: str
    transmission: str
    engine_capacity: str
    body_type: str
    province: str = "Punjab"
    assembly: str = "Local"
    market: str = "pakistan"


class AvailabilityRequest(BaseModel):
    make: str
    model: str
    year: int
    country: str
    city: str


class ForecastRequest(BaseModel):
    make: str
    model: str
    vehicle_age: float
    current_price_usd: Optional[float] = None
    odometer_miles: Optional[float] = None


class ForecastResponse(BaseModel):
    make: str
    model: str
    vehicle_age: float
    annual_depreciation_rate: float
    curve_r2: float
    forecasts: dict
    supported: bool
    message: str


class DepreciationResponse(BaseModel):
    make: str
    model: str
    annual_depreciation_rate: float
    estimated_new_price_usd: float
    curve_r2: float
    cohort_points: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded_pk": ml_model_pk is not None,
        "model_loaded_intl": ml_model_intl is not None,
    }


@app.post("/generate-vehicle", response_model=VehicleKnowledgeCard)
async def generate_vehicle_endpoint(req: GenerateVehicleRequest):
    try:
        rag_context = get_rag_context(req.make, req.year)
        data = generate_vehicle(
            make=req.make,
            model=req.model,
            year=req.year,
            variant=req.variant,
            rag_context=rag_context,
        )
        return data
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-vehicle-stream")
async def generate_vehicle_stream_endpoint(req: GenerateVehicleRequest):
    rag_context = get_rag_context(req.make, req.year)

    async def event_generator():
        try:
            async for event_type, data in stream_vehicle(
                make=req.make,
                model=req.model,
                year=req.year,
                variant=req.variant,
                rag_context=rag_context,
            ):
                yield {"event": event_type, "data": json.dumps(data)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


@app.post("/predict-price")
async def predict_price(req: PredictPriceRequest):
    if req.market == "international":
        model, encoders_used = ml_model_intl, encoders_intl
        price_field, currency = "predicted_price_usd", "USD"
    else:
        model, encoders_used = ml_model_pk, encoders_pk
        price_field, currency = "predicted_price_pkr", "PKR"

    if model is None or encoders_used is None:
        raise HTTPException(
            status_code=503,
            detail=f"ML model for market '{req.market}' not loaded. Run train.py first.",
        )

    # Encode categoricals — use 'Unknown' for unseen labels
    def safe_encode(encoder, value: str) -> int:
        classes = list(encoder.classes_)
        if value in classes:
            return encoder.transform([value])[0]
        return 0  # fallback to first class for unseen labels

    engine_cc_str = "".join(c for c in req.engine_capacity if c.isdigit() or c == ".")
    engine_cc = float(engine_cc_str) if engine_cc_str else 0.0

    vehicle_age = CURRENT_YEAR - req.year

    features_row = [
        safe_encode(encoders_used["make"], req.make),
        safe_encode(encoders_used["fuel_type"], req.fuel_type),
        safe_encode(encoders_used["transmission"], req.transmission),
        safe_encode(encoders_used["body_type"], req.body_type),
        safe_encode(encoders_used["province"], req.province),
        safe_encode(encoders_used["assembly"], req.assembly),
        req.year,
        vehicle_age,
        req.mileage_km,
        engine_cc,
    ]

    X = np.array([features_row])
    predicted = float(model.predict(X)[0])

    # Confidence range: ±15%
    return {
        price_field: round(predicted),
        "currency": currency,
        "market": req.market,
        "confidence_range": {
            "min": round(predicted * 0.85),
            "max": round(predicted * 1.15),
        },
    }


@app.post("/detect-availability")
async def detect_availability_endpoint(req: AvailabilityRequest):
    try:
        return detect_availability(
            make=req.make,
            model=req.model,
            year=req.year,
            country=req.country,
            city=req.city,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _project(make, model, current_age, horizon_months, current_price=None, odometer=None):
    """Project price at a given horizon."""
    curves = _forecast_models["curves"]
    xgb_model = _forecast_models["xgb"]
    qrf_model = _forecast_models["qrf"]
    qrf_type = _forecast_models["qrf_type"]
    features = _forecast_models["features"]

    key = f"{make}_{model}"
    if key not in curves:
        for k in curves:
            if k.lower() == key.lower():
                key = k
                break
        else:
            return None

    curve = curves[key]
    future_age = current_age + (horizon_months / 12)

    # Curve projection at future age
    log_price = curve["intercept"] + curve["slope"] * future_age
    curve_price = np.exp(log_price)

    # Anchor to current market price
    if current_price:
        current_log = curve["intercept"] + curve["slope"] * current_age
        current_fitted = np.exp(current_log)
        if current_fitted > 0:
            curve_price *= (current_price / current_fitted)

    # rolling_3m_price should reflect projected price at horizon
    # not current price — this is what was causing identical outputs
    projected_rolling = curve_price  # use the curve projection as rolling estimate

    if xgb_model and features:
        odometer_val = (odometer or 50000) + (horizon_months * 1200)
        feat = {
            "median_age": future_age,
            "median_age_sq": future_age ** 2,
            "month_index": 100 + horizon_months,
            "listing_count": 20,
            "rolling_3m_price": projected_rolling,  # fixed
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

    # QRF confidence intervals
    if qrf_model and qrf_type == "quantile_forest" and features:
        try:
            odometer_val = (odometer or 50000) + (horizon_months * 1200)
            feat_dict = {
                "median_age": future_age,
                "median_age_sq": future_age ** 2,
                "month_index": 100 + horizon_months,
                "listing_count": 20,
                "rolling_3m_price": projected_rolling,  # fixed
                "price_change_pct": curve["slope"] / 12,
                "depreciation_slope": curve["slope"],
                "median_odometer": odometer_val,
            }
            feat_array = np.array([[feat_dict.get(f, 0) for f in features]])
            preds = qrf_model.predict(feat_array, quantiles=[0.1, 0.5, 0.9])
            lower = float(preds[0][0])
            median_val = float(preds[0][1])
            upper = float(preds[0][2])

            # QRF may still be too flat — blend with curve-based intervals
            # and ensure intervals widen with horizon
            spread_pct = 0.08 + (horizon_months * 0.012)
            curve_lower = blended * (1 - spread_pct)
            curve_upper = blended * (1 + spread_pct * 0.6)

            # Take wider of QRF and curve intervals
            lower = min(lower, curve_lower)
            upper = max(upper, curve_upper)
            median_val = blended  # trust blended over QRF median

        except Exception:
            spread_pct = 0.08 + (horizon_months * 0.012)
            lower = blended * (1 - spread_pct)
            median_val = blended
            upper = blended * (1 + spread_pct * 0.6)
    else:
        spread_pct = 0.08 + (horizon_months * 0.012)
        lower = blended * (1 - spread_pct)
        median_val = blended
        upper = blended * (1 + spread_pct * 0.6)

    change_pct = (
        ((median_val - current_price) / current_price) * 100
        if current_price else None
    )

    return {
        "median_usd": round(median_val, 2),
        "lower_usd": round(max(lower, 100), 2),
        "upper_usd": round(upper, 2),
        "change_pct": round(change_pct, 1) if change_pct is not None else None,
        "annual_depreciation_rate": curve["annual_depreciation_rate"],
        "curve_r2": curve["r2"],
    }


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

    key = next(k for k in _forecast_models["curves"] if k.lower() == key_lower)
    curve = _forecast_models["curves"][key]

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


@app.get("/market/health")
async def market_health():
    if not _forecast_models["loaded"]:
        return {"loaded": False}

    curves = _forecast_models["curves"]

    rates = [c["annual_depreciation_rate"] for c in curves.values()]
    avg_rate = np.mean(rates) if rates else 0

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


