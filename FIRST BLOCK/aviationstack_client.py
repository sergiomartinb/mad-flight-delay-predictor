# aviationstack_client.py
import time
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
from config import AVIATIONSTACK_API_KEY, AVIATIONSTACK_BASE_URL

DEFAULT_LIMIT = 100  # per docs/connectors, max 100 per page
TIMEOUT = 15
BACKOFF = [1, 2, 5]  # seconds

class AviationstackError(Exception):
    pass

def _request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = urljoin(AVIATIONSTACK_BASE_URL + "/", endpoint.lstrip("/"))
    # Inject key
    q = {"access_key": AVIATIONSTACK_API_KEY, "limit": DEFAULT_LIMIT, **params}

    for attempt, wait in enumerate([0] + BACKOFF, start=1):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=q, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == len([0] + BACKOFF):
                raise AviationstackError(f"Network error: {e}") from e
            continue

        if resp.status_code == 200:
            data = resp.json()
            # Aviationstack returns a "pagination" and "data" block
            if isinstance(data, dict) and "data" in data:
                return data
            raise AviationstackError(f"Unexpected payload: {data}")
        elif resp.status_code in (429, 500, 502, 503, 504):
            # backoff & retry
            if attempt == len([0] + BACKOFF):
                raise AviationstackError(
                    f"Server/rate error {resp.status_code}: {resp.text[:200]}"
                )
            continue
        else:
            raise AviationstackError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
    raise AviationstackError("Exhausted retries")

def fetch_flights_page(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a single page from /flights with given params."""
    return _request("flights", params)

def fetch_all_flights(params: Dict[str, Any], max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Paginate /flights using 'limit' + 'offset'. 
    """
    results: List[Dict[str, Any]] = []
    offset = int(params.get("offset", 0))
    page_count = 0
    while True:
        page = fetch_flights_page({**params, "offset": offset, "limit": DEFAULT_LIMIT})
        data = page.get("data", [])
        results.extend(data)
        pagination = page.get("pagination", {})
        total = pagination.get("total", None)
        count = len(data)

        # Stop if no data or we’ve fetched all
        if count == 0:
            break
        offset += count
        page_count += 1
        if max_pages and page_count >= max_pages:
            break
        # If total known, stop when reached
        if total is not None and offset >= total:
            break
    return results
