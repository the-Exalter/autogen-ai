import os
import json
import re
import asyncio
import joblib
import numpy as np
import httpx
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

from ai_agent import generate_vehicle, detect_availability, stream_vehicle
from rag_context import get_rag_context

import pickle
from pathlib import Path
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


async def firecrawl_search(query: str, limit: int = 3) -> list:
    """Search using Firecrawl API and return results."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return []

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                "https://api.firecrawl.dev/v1/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "limit": limit,
                    "scrapeOptions": {
                        "formats": ["markdown"]
                    }
                }
            )
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"[Firecrawl] Search error: {e}")
            return []


class ReadMoreRequest(BaseModel):
    make: str
    model: str
    year: Optional[int] = None
    variant: Optional[str] = None


@app.post("/read-more-stream")
async def read_more_stream(req: ReadMoreRequest):
    """
    Stream structured Read More content using SSE.
    Fetches from YouTube and Reddit via Firecrawl,
    then Claude structures it into sections.
    """
    vehicle_name = f"{req.make} {req.model}"
    if req.year:
        vehicle_name = f"{req.year} {vehicle_name}"

    async def generate():
        try:
            yield f"event: status\ndata: {json.dumps({'message': f'Researching {vehicle_name}...'})}\n\n"

            youtube_query = f"{req.make} {req.model} site:youtube.com"
            reddit_query = f"{req.make} {req.model} reddit stories history"
            general_query = f"{req.make} {req.model} history review"

            youtube_results, reddit_results, general_results = await asyncio.gather(
                firecrawl_search(youtube_query, 3),
                firecrawl_search(reddit_query, 3),
                firecrawl_search(general_query, 2),
                return_exceptions=True
            )

            if isinstance(youtube_results, Exception):
                youtube_results = []
            if isinstance(reddit_results, Exception):
                reddit_results = []
            if isinstance(general_results, Exception):
                general_results = []

            yield f"event: status\ndata: {json.dumps({'message': 'Synthesizing content...'})}\n\n"

            context_parts = []

            videos = []
            for r in youtube_results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    title = r.get("title", "")
                    description = r.get("description", "")
                    vid_match = re.search(r'watch\?v=([a-zA-Z0-9_-]{11})', url)
                    if vid_match:
                        videos.append({
                            "id": vid_match.group(1),
                            "title": title,
                            "description": description,
                            "url": url
                        })
                    markdown = r.get("markdown", "")
                    if markdown and len(markdown) > 200:
                        context_parts.append(
                            f"=== YouTube: {title} ===\n{markdown[:3000]}"
                        )

            reddit_items = []
            for r in reddit_results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    title = r.get("title", "")
                    description = r.get("description", "")
                    markdown = r.get("markdown", "")
                    reddit_items.append({
                        "title": title,
                        "url": url,
                        "description": description
                    })
                    if markdown and len(markdown) > 100:
                        context_parts.append(
                            f"=== Reddit: {title} ===\n{description}\n{markdown[:2000]}"
                        )
                    else:
                        context_parts.append(
                            f"=== Reddit: {title} ===\n{description}"
                        )

            for r in general_results:
                if isinstance(r, dict):
                    title = r.get("title", "")
                    markdown = r.get("markdown", "")
                    if markdown and len(markdown) > 200:
                        context_parts.append(
                            f"=== Web: {title} ===\n{markdown[:2000]}"
                        )

            if videos:
                yield f"event: videos\ndata: {json.dumps({'videos': videos})}\n\n"

            if not context_parts and not videos:
                yield f"event: error\ndata: {json.dumps({'message': 'No content found for this vehicle'})}\n\n"
                return

            combined_context = "\n\n".join(context_parts)

            read_more_prompt = f"""You are writing content for the Read More section of a vehicle knowledge platform.

Vehicle: {vehicle_name}

Here is research content gathered from YouTube videos, Reddit discussions, and web sources:

{combined_context}

Structure this into a JSON object with these exact sections.
Return ONLY raw JSON, no markdown fences:

{{
  "fast_facts": [
    {{"label": "string", "value": "string"}}
  ],
  "the_story": {{
    "title": "string",
    "paragraphs": ["string", "string"]
  }},
  "what_owners_say": [
    {{"quote": "string", "source": "string"}}
  ],
  "notable_facts": [
    "string"
  ]
}}

Rules:
- fast_facts: 3-5 key facts with short label and value (e.g. {{"label": "0-100 km/h", "value": "3.8 seconds"}})
- the_story: 2-3 short paragraphs about history and significance. NOT boring. Include real stories from the sources.
- what_owners_say: 2-4 real quotes or paraphrased stories from Reddit or YouTube transcripts. Include source name.
- notable_facts: 3-5 genuinely interesting facts. Not generic specs. Things like "Only 1,311 were ever made" or "Ferrari CEO Enzo Ferrari called it the best Ferrari ever built"
- Everything must be specific to this vehicle. No generic car advice.
- Keep paragraphs SHORT. 2-3 sentences max each.
- Return ONLY the JSON object."""

            ai_client = anthropic.AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )

            full_text = ""
            async with ai_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": read_more_prompt
                }]
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield f"event: delta\ndata: {json.dumps({'text': text})}\n\n"

            try:
                cleaned = full_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```[a-z]*\n?', '', cleaned)
                    cleaned = re.sub(r'\n?```$', '', cleaned)

                structured = json.loads(cleaned)
                yield f"event: done\ndata: {json.dumps({'content': structured, 'videos': videos, 'reddit': reddit_items})}\n\n"
            except json.JSONDecodeError as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Parse error: {str(e)}'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


class UsedMarketRequest(BaseModel):
    make: str
    model: str
    year: Optional[int] = None


@app.post("/find-used-listings")
async def find_used_listings(req: UsedMarketRequest):
    """
    Scrape Cars.com for real used listings of the requested vehicle.
    Returns up to 6 structured listings.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return {"found": False, "listings": [], "error": "Firecrawl not configured"}

    make_slug = req.make.lower().replace(" ", "-")
    model_slug = f"{make_slug}-{req.model.lower().replace(' ', '-')}"

    base_url = (
        f"https://www.cars.com/shopping/results/"
        f"?stock_type=used"
        f"&makes[]={make_slug}"
        f"&models[]={model_slug}"
        f"&maximum_distance=all"
        f"&sort=best_match_desc"
    )

    urls_to_try = [base_url]

    markdown = ""
    source_url = ""

    async with httpx.AsyncClient(timeout=30) as client:
        for url in urls_to_try:
            try:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"url": url, "formats": ["markdown"]}
                )
                data = resp.json()
                if data.get("success") and data.get("data", {}).get("markdown"):
                    markdown = data["data"]["markdown"]
                    source_url = url
                    break
            except Exception as e:
                print(f"[UsedMarket] Scrape error: {e}")
                continue

    if not markdown:
        return {"found": False, "listings": [], "source": "cars.com"}

    listings = []

    title_pattern = re.findall(
        r'##\s+\[([^\]]+)\]\((https://www\.cars\.com/vehicledetail/[^\)]+)\)',
        markdown
    )

    price_pattern = re.findall(r'\$(\d{1,3},\d{3})', markdown)

    mileage_pattern = re.findall(r'([\d,]+)\s+mi\.', markdown)

    dealer_pattern = re.findall(
        r'(?:Good Deal|Great Deal|Fair Deal|Certified Pre-Owned|High Demand)\s*\n+([A-Z][^\n]+(?:Toyota|Honda|Ford|Chevrolet|Nissan|Hyundai|Kia|BMW|Mercedes|Dodge|Ram|Jeep|Acura|Lexus|Mazda|Subaru|Volkswagen|Audi|Porsche|Motors|Auto|Cars|Dealer|Group|Center)[^\n]*)',
        markdown
    )

    image_pattern = re.findall(
        r'https://platform\.cstatic-images\.com/[^\)"\s]+\.jpg',
        markdown
    )

    location_pattern = re.findall(
        r'([A-Z][a-zA-Z\s]+,\s+[A-Z]{2})\s+\(',
        markdown
    )

    for i, (title, detail_url) in enumerate(title_pattern[:6]):
        listing = {
            "title": title.strip(),
            "url": detail_url.strip(),
            "price_usd": None,
            "mileage_mi": None,
            "dealer": None,
            "location": None,
            "image": None,
        }

        if i < len(price_pattern):
            try:
                listing["price_usd"] = int(price_pattern[i].replace(",", ""))
            except Exception:
                pass

        if i < len(mileage_pattern):
            try:
                listing["mileage_mi"] = int(mileage_pattern[i].replace(",", ""))
            except Exception:
                pass

        if i < len(dealer_pattern):
            listing["dealer"] = dealer_pattern[i].strip()

        if i < len(location_pattern):
            listing["location"] = location_pattern[i].strip()

        img_idx = i * 6
        if img_idx < len(image_pattern):
            listing["image"] = image_pattern[img_idx]

        listings.append(listing)

    total_match = re.search(r'([\d,]+\+?)\s+results', markdown)
    total_count = total_match.group(1) if total_match else f"{len(listings)}+"

    return {
        "found": len(listings) > 0,
        "listings": listings,
        "total_count": total_count,
        "source": "cars.com",
        "source_url": source_url,
        "make": req.make,
        "model": req.model,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


