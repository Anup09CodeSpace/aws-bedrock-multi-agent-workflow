
import os
import json, logging
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY", "")
DEFAULT_RADIUS_M = int(os.environ.get("GEOAPIFY_RADIUS_M", "3000"))
DEFAULT_LIMIT = int(os.environ.get("GEOAPIFY_LIMIT", "7"))
DEFAULT_LANG = os.environ.get("GEOAPIFY_LANG", "en")

GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL = "https://api.geoapify.com/v2/places"
DETAILS_URL = "https://api.geoapify.com/v2/place-details"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------
# Helpers: HTTP + parsing
# ---------------------------
def http_get_json(url: str, timeout_s: int = 10) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "bedrock-action-lambda/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_bedrock_param(event: Dict[str, Any], name: str) -> Optional[Any]:
    """
    Bedrock action group event usually includes:
      event["parameters"] = [{"name":"city","type":"string","value":"Toronto"}, ...]
    We'll match case-insensitively.
    """
    params = event.get("parameters") or []
    for p in params:
        if isinstance(p, dict) and str(p.get("name", "")).lower() == name.lower():
            return p.get("value")
    return None


def normalize_str(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


# ---------------------------
# Geoapify: city -> lat/lon
# ---------------------------
def geocode_city(city: str) -> Tuple[float, float, Dict[str, Any]]:
    """
    Geoapify Geocoding API example usage appears in Geoapify samples:
      https://api.geoapify.com/v1/geocode/search?text=<city>&format=json&apiKey=<key>
    [1](https://github.com/geoapify/sample-query-places-python/blob/main/city_find.py)
    """
    q = urllib.parse.quote_plus(city)
    params = f"text={q}&format=json&lang={urllib.parse.quote_plus(DEFAULT_LANG)}&apiKey={urllib.parse.quote_plus(GEOAPIFY_API_KEY)}"
    url = f"{GEOCODE_URL}?{params}"

    data = http_get_json(url)
    results = data.get("results") or []
    if not results:
        raise ValueError(f"City not found: {city}")

    # Choose best match (first result)
    best = results[0]
    lat = float(best["lat"])
    lon = float(best["lon"])
    return lat, lon, best


# ---------------------------
# Geoapify: places search
# ---------------------------
TYPE_TO_CATEGORIES = {
    # Common "type" mappings (best effort)
    "restaurant": "catering.restaurant",
    "fine_dining": "catering.restaurant",
    "cafe": "catering.cafe",
    "coffee": "catering.cafe",
    "fast_food": "catering.fast_food",
    "pizza": "catering.fast_food,catering.restaurant",
    "bar": "catering.bar",
    "pub": "catering.pub",
    "bakery": "catering.bakery",
}


def build_places_url(lat: float, lon: float, radius_m: int, limit: int, categories: str) -> str:
    """
    Geoapify Places API supports:
      - endpoint /v2/places  [2](https://apidocs.geoapify.com/docs/places/)
      - location filter: filter=circle:<lon>,<lat>,<radius>
      - ranking bias: bias=proximity:<lon>,<lat>
    Example shown in Geoapify guidance [3](https://www.linkedin.com/pulse/how-fetch-points-interest-pois-near-location-geoapify-geoapify-glpkf)
    """
    query = {
        "categories": categories,
        "filter": f"circle:{lon},{lat},{radius_m}",
        "bias": f"proximity:{lon},{lat}",
        "limit": str(limit),
        "lang": DEFAULT_LANG,
        "apiKey": GEOAPIFY_API_KEY
    }
    return f"{PLACES_URL}?{urllib.parse.urlencode(query)}"


def build_details_url(place_id: str) -> str:
    """
    Geoapify Place Details API:
      /v2/place-details?id=<place_id>&features=details&apiKey=<key>
    [4](https://apidocs.geoapify.com/docs/place-details/)
    """
    query = {
        "id": place_id,
        "features": "details",
        "lang": DEFAULT_LANG,
        "apiKey": GEOAPIFY_API_KEY
    }
    return f"{DETAILS_URL}?{urllib.parse.urlencode(query)}"


def extract_cuisine(props: Dict[str, Any]) -> str:
    # Geoapify may provide cuisine inside a "catering" object for restaurants if available.
    catering = props.get("catering") or {}
    if isinstance(catering, dict) and catering.get("cuisine"):
        return str(catering.get("cuisine"))
    # Sometimes raw OSM tags may exist
    raw = ((props.get("datasource") or {}).get("raw") or {})
    if isinstance(raw, dict) and raw.get("cuisine"):
        return str(raw.get("cuisine"))
    return ""


def normalize_feature(feature: Dict[str, Any], details_feature: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = (feature.get("properties") or {})
    g = (feature.get("geometry") or {})
    coords = g.get("coordinates") or [None, None]  # [lon, lat]

    dp = (details_feature or {}).get("properties") or {}
    # Prefer details properties if present
    props = dp if dp else p

    return {
        "place_id": props.get("place_id") or p.get("place_id"),
        "name": props.get("name") or p.get("name"),
        "address": props.get("formatted") or p.get("formatted"),
        "city": props.get("city") or p.get("city"),
        "lat": coords[1],
        "lon": coords[0],
        "categories": props.get("categories") or p.get("categories"),
        "opening_hours": props.get("opening_hours") or p.get("opening_hours"),
        "cuisine": extract_cuisine(props) or extract_cuisine(p),
        # rating/budget may or may not exist depending on datasource/tags
        "rating": props.get("rating") or p.get("rating"),
        "price_level": props.get("price_level") or p.get("price_level"),
        "source": "geoapify"
    }


def apply_filters(
    items: List[Dict[str, Any]],
    cuisine: str = "",
    rating: Optional[int] = None,
    budget: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Best-effort filtering:
      - cuisine: filter when we can read it from tags/details
      - Rating/Budget: only filter if fields exist in returned payload
    Geoapify response commonly includes name/location/address/tags but not guaranteed rating/budget [3](https://www.linkedin.com/pulse/how-fetch-points-interest-pois-near-location-geoapify-geoapify-glpkf)
    """
    cuisine = cuisine.strip().lower()
    out = []

    for it in items:
        # cuisine filter (substring match, supports "italian;pizza" etc)
        if cuisine:
            c = normalize_str(it.get("cuisine")).lower()
            if cuisine not in c:
                continue

        # Rating filter (only if present)
        if rating is not None:
            r = safe_float(it.get("rating"))
            if r is not None and r < float(rating):
                continue

        # Budget filter: interpret as max price_level if price_level numeric, otherwise skip
        # NOTE: many POIs won't have price_level; we only filter when it's present.
        if budget is not None:
            pl = safe_float(it.get("price_level"))
            if pl is not None and pl > float(budget):
                continue

        out.append(it)

    return out


# ---------------------------
# Bedrock response wrapper
# ---------------------------
def bedrock_action_response(event, status_code: int, payload: dict):
    logger.info("Lambda Response: %s", json.dumps(payload))
    """
    Response envelope for Bedrock Agents Action Groups configured with Function details.
    Must echo event['function'] and event['actionGroup'].
    """
    return {
        "messageVersion": event.get("messageVersion", "1.0"),
        "response": {
            "actionGroup": event.get("actionGroup"),
            "function": event.get("function"),
            "httpStatusCode": status_code,
            "functionResponse": {
                "responseBody": {
                    "TEXT": {                        
                        "body": json.dumps(payload)                    
                    }
                }
            }
        }
    }


# ---------------------------
# Lambda handler
# ---------------------------
def lambda_handler(event, context):
    logger.info("BEDROCK EVENT: %s", json.dumps(event))
    if not GEOAPIFY_API_KEY:
        return bedrock_action_response(event, 500, {"error": "Missing env var GEOAPIFY_API_KEY"})

    # --- Read Bedrock parameters from screenshot contract ---
    city = normalize_str(get_bedrock_param(event, "city"))

    ANY_SENTINELS = {"any", "all", "*", "none", "n/a", "na", ""}
    # ...
    cuisine = normalize_str(get_bedrock_param(event, "cuisine"))
    # Normalize "any" → no filter
    if cuisine.strip().lower() in ANY_SENTINELS:
        cuisine = ""
        
    r_min = safe_int(get_bedrock_param(event, "rating"))
    r_type = normalize_str(get_bedrock_param(event, "type"))
    budget = safe_float(get_bedrock_param(event, "budget"))

    if not city:
        return bedrock_action_response(event, 400, {"error": "Parameter 'city' is required"})

    # Decide categories based on "type" param (best effort)
    categories = TYPE_TO_CATEGORIES.get(r_type.strip().lower(), "catering.restaurant")

    # Tuning knobs (can be env vars)
    radius_m = DEFAULT_RADIUS_M
    limit = DEFAULT_LIMIT

    try:
        # 1) Geocode city -> lat/lon [1](https://github.com/geoapify/sample-query-places-python/blob/main/city_find.py)
        lat, lon, geo = geocode_city(city)

        # 2) Search places via Geoapify Places API using circle filter + proximity bias [2](https://apidocs.geoapify.com/docs/places/)[3](https://www.linkedin.com/pulse/how-fetch-points-interest-pois-near-location-geoapify-geoapify-glpkf)
        places_url = build_places_url(lat, lon, radius_m, limit, categories)
        places = http_get_json(places_url)
        features = places.get("features") or []

        # If cuisine filter requested, enrich with Place Details for better coverage [4](https://apidocs.geoapify.com/docs/place-details/)
        want_details = bool(cuisine)  # you can extend this condition if desired
        details_cache = {}

        if want_details:
            for f in features[:limit]:
                pid = (f.get("properties") or {}).get("place_id")
                if not pid:
                    continue
                try:
                    det = http_get_json(build_details_url(pid))
                    det_features = det.get("features") or []
                    if det_features:
                        details_cache[pid] = det_features[0]
                    # small sleep to be gentle (optional)
                    time.sleep(0.02)
                except Exception:
                    # soft-fail details
                    pass

        # 3) Normalize
        items = []
        for f in features:
            pid = (f.get("properties") or {}).get("place_id")
            items.append(normalize_feature(f, details_cache.get(pid)))

        # 4) Apply filters
        filtered = apply_filters(items, cuisine=cuisine, rating=r_min, budget=budget)

        # 5) Return payload
        return bedrock_action_response(event, 200, {
            "query": {
                "city": city,
                "center": {"lat": lat, "lon": lon},
                "radius_m": radius_m,
                "categories": categories,
                "filters": {
                    "cuisine": cuisine or None,
                    "rating_min": r_min,
                    "budget_max": budget,
                    "type": r_type or None
                }
            },
            "count": len(filtered),
            "results": filtered[:limit]
        })

    except ValueError as ve:
        return bedrock_action_response(event, 404, {"error": str(ve), "city": city})
    except Exception as e:
        return bedrock_action_response(event, 500, {"error": "Unhandled error", "detail": str(e)})