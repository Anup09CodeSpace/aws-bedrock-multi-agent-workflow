import os
import json
import time
import logging
import traceback
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Geoapify endpoints
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL  = "https://api.geoapify.com/v2/places"
DETAILS_URL = "https://api.geoapify.com/v2/place-details"

# Config
GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY", "")
DEFAULT_RADIUS_M = int(os.environ.get("GEOAPIFY_RADIUS_M", "5000"))
DEFAULT_LIMIT    = int(os.environ.get("GEOAPIFY_LIMIT", "10"))
DEFAULT_LANG     = os.environ.get("GEOAPIFY_LANG", "en")

# You can override this if you confirm a different category in Geoapify categories list/playground.
HOTEL_CATEGORIES = os.environ.get("HOTEL_CATEGORIES", "accommodation.hotel")

# ---------- HTTP helper ----------
def http_get_json(url: str, timeout_s: int = 12) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "bedrock-hotel-collab/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

# ---------- Bedrock Function Details helpers ----------
def get_param(event: Dict[str, Any], name: str) -> Optional[Any]:
    for p in event.get("parameters") or []:
        if str(p.get("name", "")).lower() == name.lower():
            return p.get("value")
    return None

def bedrock_function_text_response(event: Dict[str, Any], status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT: For Bedrock Action Groups configured with Function details, use TEXT responseBody.
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

# ---------- Geoapify steps ----------
def geocode_city(city: str) -> Tuple[float, float, Dict[str, Any]]:
    """
    Geoapify Geocoding API pattern (text + format=json + apiKey).
    Example style used in Geoapify sample code. [1](https://apidocs.geoapify.com/docs)
    """
    q = urllib.parse.quote_plus(city)
    url = f"{GEOCODE_URL}?text={q}&format=json&lang={urllib.parse.quote_plus(DEFAULT_LANG)}&apiKey={urllib.parse.quote_plus(GEOAPIFY_API_KEY)}"
    data = http_get_json(url)
    results = data.get("results") or []
    if not results:
        raise ValueError(f"City not found: {city}")
    best = results[0]
    return float(best["lat"]), float(best["lon"]), best

def build_places_url(lat: float, lon: float, radius_m: int, limit: int, categories: str) -> str:
    """
    Geoapify Places API: GET /v2/places with categories + apiKey required. [2](https://apidocs.geoapify.com/docs/places/)[3](https://www.geoapify.com/places-api/)
    Example usage for circle filter & proximity bias is commonly shown in Geoapify guidance. 
    """
    params = {
        "categories": categories,
        "filter": f"circle:{lon},{lat},{radius_m}",
        "bias": f"proximity:{lon},{lat}",
        "limit": str(limit),
        "lang": DEFAULT_LANG,
        "apiKey": GEOAPIFY_API_KEY
    }
    return f"{PLACES_URL}?{urllib.parse.urlencode(params)}"

def build_details_url(place_id: str) -> str:
    """
    Geoapify Place Details API: GET /v2/place-details?id=...&features=details&apiKey=... [4](https://apidocs.geoapify.com/docs/place-details/)[5](https://www.geoapify.com/place-details-api/)
    """
    params = {
        "id": place_id,
        "features": "details",
        "lang": DEFAULT_LANG,
        "apiKey": GEOAPIFY_API_KEY
    }
    return f"{DETAILS_URL}?{urllib.parse.urlencode(params)}"

# ---------- Normalization & filters ----------
def safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except Exception:
        return None

def normalize_amenity_tokens(amenities: List[str]) -> List[str]:
    return [a.strip().lower() for a in (amenities or []) if str(a).strip()]

def extract_raw_tags(props: Dict[str, Any]) -> Dict[str, Any]:
    # Geoapify often includes datasource.raw containing OSM-like tags when available
    ds = props.get("datasource") or {}
    raw = ds.get("raw") or {}
    return raw if isinstance(raw, dict) else {}

def extract_star_rating_from_tags(raw: Dict[str, Any]) -> Optional[int]:
    # Best-effort: OSM commonly uses "stars" or "hotel:stars" for hotels (not guaranteed)
    for key in ("stars", "hotel:stars"):
        if key in raw:
            return safe_int(raw.get(key))
    return None

def extract_amenities_from_tags(raw: Dict[str, Any]) -> List[str]:
    """
    Best-effort mapping from tags to user-facing amenities.
    Only returns what we can confidently infer from tags.
    """
    out = set()

    # WiFi / Internet
    if raw.get("internet_access") in ("wlan", "yes", "wifi") or raw.get("wifi") in ("yes", "free"):
        out.add("Free WiFi")

    # Breakfast
    if raw.get("breakfast") in ("yes", "included") or raw.get("food") == "yes":
        out.add("Breakfast")

    # Parking
    if raw.get("parking") in ("yes", "private") or raw.get("parking:fee") in ("no", "yes"):
        out.add("Parking")

    # Pool / Gym (rare in OSM tags, but sometimes present)
    if raw.get("swimming_pool") in ("yes", "indoor", "outdoor"):
        out.add("Pool")
    if raw.get("gym") in ("yes",):
        out.add("Gym")

    return sorted(out)

def normalize_hotel(feature: Dict[str, Any], details_feature: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = (feature.get("properties") or {})
    geom = (feature.get("geometry") or {})
    coords = geom.get("coordinates") or [None, None]  # [lon, lat]

    dp = (details_feature or {}).get("properties") or {}
    props = dp if dp else p

    raw = extract_raw_tags(props) or extract_raw_tags(p)
    star_rating = extract_star_rating_from_tags(raw)
    amenity_list = extract_amenities_from_tags(raw)

    # NOTE: Price per night is generally not present in OSM-derived POI data.
    # We will not fabricate it; we return None.
    return {
        "hotel_name": props.get("name") or p.get("name"),
        "star_rating": star_rating,  # may be None
        "location": {
            "address": props.get("formatted") or p.get("formatted"),
            "city": props.get("city") or p.get("city"),
            "lat": coords[1],
            "lon": coords[0],
        },
        "price_per_night": None,  # not available from Geoapify/OSM in most cases
        "key_amenities": amenity_list,
        "short_description": None,  # avoid hallucination
        "place_id": props.get("place_id") or p.get("place_id"),
        "source": "geoapify"
    }

def passes_filters(
    hotel: Dict[str, Any],
    required_star: int,
    requested_amenities: List[str]
) -> bool:
    # Star rating filter: only enforce if we actually have a star rating in data
    data_star = hotel.get("star_rating")
    if data_star is not None and int(data_star) != int(required_star):
        return False

    # Amenities filter: only enforce requested amenities that we can match against extracted list
    if requested_amenities:
        have = set([a.lower() for a in (hotel.get("key_amenities") or [])])
        # If user asks for something we don't have evidence for, we don't reject automatically.
        # We only enforce matches for amenities we can clearly verify.
        verifiable = {"free wifi", "breakfast", "parking", "pool", "gym"}
        for req in requested_amenities:
            if req in verifiable and req not in have:
                return False

    return True

# ---------- Lambda handler ----------
def lambda_handler(event, context):
    # Always log at entry for debugging
    logger.info("EVENT: %s", json.dumps(event))

    try:
        if not GEOAPIFY_API_KEY:
            return bedrock_function_text_response(event, 500, {"error": "Missing env var GEOAPIFY_API_KEY"})

        fn = event.get("function") or ""
        # Adjust allowed function name(s) to match your action group
        allowed = {"hotel-ag", "list-hotels", "recommend-hotels", "get-hotel-recommendations"}
        if fn and fn not in allowed:
            return bedrock_function_text_response(event, 404, {"error": "Unknown function", "function": fn, "allowed": sorted(allowed)})

        # Required fields (your supervisor agent ensures these exist, but we validate anyway)
        city = (get_param(event, "city") or "").strip()
        guests = safe_int(get_param(event, "guest_number"))
        hotel_star_rating = safe_int(get_param(event, "hotel_star_rating"))

        # Optional fields (apply defaults as per your instruction)
        budget = (get_param(event, "budget") or "medium").strip().lower()
        amenities = get_param(event, "amenities") or ["Free WiFi"]

        # Normalize amenities input (could be list or string)
        if isinstance(amenities, str):
            amenities = [amenities]
        amenities_norm = normalize_amenity_tokens(amenities)

        if not city:
            return bedrock_function_text_response(event, 400, {"error": "Missing required parameter: city"})
        if guests is None:
            return bedrock_function_text_response(event, 400, {"error": "Missing required parameter: guests"})
        if hotel_star_rating not in (3, 4, 5):
            return bedrock_function_text_response(event, 400, {"error": "Missing/invalid required parameter: hotel_star_rating (must be 3,4,5)"})

        # 1) Geocode city -> lat/lon [1](https://apidocs.geoapify.com/docs)
        lat, lon, geo = geocode_city(city)

        # 2) Places search (hotels)
        # Geoapify Places API requires categories + apiKey and uses /v2/places endpoint. [2](https://apidocs.geoapify.com/docs/places/)[3](https://www.geoapify.com/places-api/)
        # Circle filter + proximity bias pattern is standard usage for nearby search. 
        places_url = build_places_url(lat, lon, DEFAULT_RADIUS_M, DEFAULT_LIMIT, HOTEL_CATEGORIES)
        places = http_get_json(places_url)
        features = places.get("features") or []

        # 3) Enrich with Place Details (optional but improves metadata availability)
        # Place Details API provides richer details using place_id. [4](https://apidocs.geoapify.com/docs/place-details/)[5](https://www.geoapify.com/place-details-api/)
        details_cache: Dict[str, Dict[str, Any]] = {}
        for f in features[:DEFAULT_LIMIT]:
            pid = (f.get("properties") or {}).get("place_id")
            if not pid:
                continue
            try:
                det = http_get_json(build_details_url(pid))
                det_features = det.get("features") or []
                if det_features:
                    details_cache[pid] = det_features[0]
                time.sleep(0.02)  # gentle pacing
            except Exception:
                # Soft-fail details
                pass

        # 4) Normalize + filter
        hotels: List[Dict[str, Any]] = []
        for f in features:
            pid = (f.get("properties") or {}).get("place_id")
            hotel_obj = normalize_hotel(f, details_cache.get(pid))
            if passes_filters(hotel_obj, required_star=hotel_star_rating, requested_amenities=amenities_norm):
                hotels.append(hotel_obj)

        # 5) “Budget” handling
        # Since per-night pricing is usually unavailable in POI data, we do not fabricate it.
        # Instead we return the budget preference in the query echo for the supervisor agent to present honestly.
        payload = {
            "query": {
                "city": city,
                "guests": guests,
                "hotel_star_rating": hotel_star_rating,
                "budget": budget,
                "amenities": amenities if amenities else ["Free WiFi"],
                "center": {"lat": lat, "lon": lon},
                "radius_m": DEFAULT_RADIUS_M,
                "categories": HOTEL_CATEGORIES
            },
            "count": len(hotels),
            "results": hotels[:DEFAULT_LIMIT]
        }
        logger.info("RESPONSE: %s", json.dumps(payload))
        return bedrock_function_text_response(event, 200, payload)

    except Exception as e:
        logger.error("UNHANDLED: %s", str(e))
        logger.error(traceback.format_exc())
        return bedrock_function_text_response(event, 500, {
            "error": "Unhandled exception in Lambda",
            "detail": str(e),
            "requestId": getattr(context, "aws_request_id", None)
        })