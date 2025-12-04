import time
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
from config import AVIATIONEDGE_API_KEY, AVIATIONEDGE_BASE_URL

# Set variables
DEFAULT_LIMIT = 100
TIMEOUT = 15
BACKOFF = [1, 2, 5]  # seconds

class AviationedgeError(Exception):
    pass

# Internal function to make requests to AviationEdge API
def _request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = urljoin(AVIATIONEDGE_BASE_URL + "/", endpoint.lstrip("/"))
    # AviationEdge expects the API key as 'key'
    q = {"key": AVIATIONEDGE_API_KEY, **params}

    for attempt, wait in enumerate([0] + BACKOFF, start=1):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=q, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == len([0] + BACKOFF):
                raise AviationedgeError(f"Network error: {e}") from e
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                # Some AviationEdge endpoints return a list directly
                data = resp.text
            return data
        elif resp.status_code in (429, 500, 502, 503, 504):
            if attempt == len([0] + BACKOFF):
                raise AviationedgeError(
                    f"Server/rate error {resp.status_code}: {resp.text[:200]}"
                )
            continue
        else:
            raise AviationedgeError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
    raise AviationedgeError("Exhausted retries")

def fetch_flight_history(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch historical flight schedule, status, and delay/cancellation data.
    """
    # Use the correct endpoint for historical data
    data = _request("flightsHistory", params)
    # The response is usually a list of flights
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "flights" in data:
        return data["flights"]
    else:
        return []

def fetch_all_delays(params: Dict[str, Any], max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch all historical delays.
    """
    return fetch_flight_history(params)

def fetch_future_schedules(iata_code: str, date: str, flight_type: str = "departure") -> List[Dict[str, Any]]:
    """
    Fetch future flight schedules for an airport on a specific date.
    
    Args:
        iata_code: Airport IATA code (e.g., "MAD")
        date: Date in YYYY-MM-DD format
        flight_type: "departure" or "arrival"
    
    Returns:
        List of scheduled flights
    """
    params = {
        "iataCode": iata_code,
        "type": flight_type,
        "date": date,
    }
    data = _request("flightsFuture", params)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "error" in data:
        return []
    return []