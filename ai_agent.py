import os
import json
import re
import anthropic

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


SYSTEM_PROMPT = SYSTEM_PROMPT = """IMPORTANT: Return ONLY a raw JSON object. Never use markdown code fences. Never write ```json or ```. Start your response with { and end with }.

You are an automotive knowledge specialist with global expertise. When given a vehicle make, model, year, and optionally variant, you return a comprehensive knowledge profile as a JSON object. The platform serves automotive enthusiasts worldwide — from daily drivers to supercars, motorcycles, classic cars, trucks, and specialty vehicles.

Return ONLY this JSON structure:

{
  "make": "string",
  "model": "string",
  "year": number,
  "variant": "string (most common or base variant if not specified)",
  "body_type": "string (Sedan/Hatchback/SUV/Crossover/Pickup/Van/Wagon/Coupe/Convertible/MPV/Roadster/Supercar/Hypercar/Truck/Other)",
  "fuel_type": "string (Petrol/Diesel/Hybrid/Electric/LPG/CNG/Hydrogen)",
  "engine_capacity": "string (e.g. '1800 cc' or '3.0L Twin-Turbo' or 'Dual Motor Electric')",
  "transmission": "string (Automatic/Manual/CVT/DCT/PDK/Single-Speed)",
  "color": "string (most iconic or common OEM color for this variant)",
  "assembly": "string (where primarily manufactured, e.g. 'Germany', 'Japan', 'USA', 'Local Assembly')",
  "typical_price_pkr": number (realistic current used market price in PKR for international reference — use USD converted at current rates if no PKR data exists),
  "description": "string (2-3 sentences covering what makes this vehicle significant, its market position globally, and its ownership profile)",
  "features": ["array of notable features specific to this variant and year — be precise, not generic"],
  "known_issues": [
    "array of 3-6 specific, commonly reported owner problems for this exact model and year",
    "Reference specific mileage thresholds, failure modes, and affected components",
    "e.g. 'N54 engine high-pressure fuel pump failure commonly occurs between 60,000-100,000 km on 2007-2013 BMW 135i'",
    "NOT generic advice. Be model-year specific."
  ],
  "maintenance_intervals": {
    "oil_change_km": number,
    "timing_belt_km": number (0 if timing chain with no scheduled replacement),
    "major_service_km": number
  },
  "parts_availability": "string — one of exactly: excellent, good, fair, poor — reflecting global parts availability",
  "buying_checklist": [
    "array of 3-5 specific inspection points for buying this model used",
    "Reference known failure points, specific components to check, and model-year specific issues",
    "NOT generic advice."
  ],
  "market_position": "string — one sentence on where this vehicle sits globally relative to its direct competitors"
}

Apply your full automotive knowledge across all vehicle categories and markets. For rare, exotic, or specialty vehicles, draw on manufacturer specifications, owner community reports, and specialist knowledge. Return ONLY the JSON object."""


REQUIRED_FIELDS = [
    "make", "model", "body_type", "fuel_type", "engine_capacity", "transmission",
    "color", "assembly", "typical_price_pkr", "description", "features",
    "known_issues", "maintenance_intervals", "parts_availability",
    "buying_checklist", "market_position",
]


def _build_user_msg(make: str, model: str, year: int | None, variant: str | None, rag_context: str) -> str:
    msg = f"Vehicle: {make} {model}"
    if year:
        msg += f" {year}"
    if variant:
        msg += f" {variant}"
    if rag_context:
        msg = f"{rag_context}\n\n{msg}"
    return msg

def clean_json_response(text: str) -> str:
    """Strip markdown code fences if Claude wraps response in them."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # remove ```json
    elif text.startswith("```"):
        text = text[3:]  # remove ```
    if text.endswith("```"):
        text = text[:-3]  # remove closing ```
    return text.strip()

def generate_vehicle(make: str, model: str, year: int | None, variant: str | None, rag_context: str = "") -> dict:
    """Call Claude to generate structured vehicle data."""
    user_msg = _build_user_msg(make, model, year, variant, rag_context)

    client = get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = clean_json_response(message.content[0].text.strip())
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in AI response: {raw[:200]}")

    data = json.loads(json_match.group())

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"AI response missing fields: {missing}")

    return data


async def stream_vehicle(make: str, model: str, year: int | None, variant: str | None, rag_context: str = ""):
    """Async generator yielding (event_type, data) tuples for SSE streaming."""
    user_msg = _build_user_msg(make, model, year, variant, rag_context)

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    full_text = ""

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        async for text in stream.text_stream:
            full_text += text
            yield "delta", {"text": text}

    cleaned = clean_json_response(full_text)  # strip fences first
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in AI response: {full_text[:200]}")
    data = json.loads(json_match.group())
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"AI response missing fields: {missing}")

    yield "done", {"vehicle": data}


AVAILABILITY_SYSTEM_PROMPT = """You are a vehicle availability researcher. Given a vehicle make, model, year,
and a target country/city, use the web_search tool to determine whether the vehicle is available
in that market (locally manufactured, officially imported, or available through grey-market imports).

Return ONLY a valid JSON object with this exact shape — no markdown, no explanation, no extra text:

{
  "found": boolean,
  "status": "string (e.g. 'Locally Available', 'Imported', 'Grey Import Only', 'Not Available')",
  "sourcing_locations": ["array", "of", "dealer", "or", "city", "names"],
  "regional_price_range": "string (e.g. 'PKR 4.5M – 5.2M')",
  "import_notes": "string (notes on import duties, JDM availability, etc.)"
}

If the vehicle is NOT available in the target market in any form, return ONLY:
{ "found": false }

Return ONLY the JSON object."""


def detect_availability(make: str, model: str, year: int, country: str, city: str) -> dict:
    """Call Claude with web_search to detect vehicle availability in a market."""
    user_msg = (
        f"Vehicle: {make} {model} {year}\n"
        f"Target market: {city}, {country}\n\n"
        f"Search the web and determine availability. Return only the JSON."
    )

    client = get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=AVAILABILITY_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = ""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    raw = raw.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in AI response: {raw[:200]}")

    data = json.loads(json_match.group())

    if data.get("found") is False:
        return {"found": False}

    return data
