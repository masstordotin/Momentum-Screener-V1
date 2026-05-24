"""Reusable NSE API client.

The NSE website expects callers to behave like a browser. This client keeps one
requests.Session open, warms it up so NSE sets cookies, and then reuses those
cookies for API calls.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry


class NSEClientError(RuntimeError):
    """Raised when NSE returns an invalid or unusable response."""


class NSEClient:
    """Small wrapper around official NSE endpoints."""

    BASE_URL = "https://www.nseindia.com"
    ARCHIVE_BASE_URL = "https://nsearchives.nseindia.com"
    LEGACY_ARCHIVE_BASE_URL = "https://archives.nseindia.com"
    NIFTY_500_STOCK_LIST_PATH = "/content/indices/ind_nifty500list.csv"
    SECURITY_BHAVCOPY_PATH_TEMPLATE = "/products/content/sec_bhavdata_full_{date}.csv"
    WARM_UP_PATHS = (
        "/get-quotes-equity-historical-data",
        "/get-quotes/equity?symbol=RELIANCE",
        "/market-data/live-market-indices",
        "/products-services/indices-nifty500-index",
        "/market-data/live-equity-market?symbol=NIFTY%20500",
        "/",
    )

    def __init__(
        self,
        timeout: tuple[int, int] = (5, 30),
        retries: int = 3,
        backoff_factor: float = 0.7,
    ) -> None:
        # A Session stores cookies and connection pools across requests.
        self.session = requests.Session()
        self.timeout = timeout

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Referer": f"{self.BASE_URL}/",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        retry_strategy = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff_factor,
            # Do not retry 503 here. NSE often uses it as an automated-request
            # block, so retrying only makes a blocked run painfully slow.
            status_forcelist=(429, 500, 502, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._cookies_ready = False

    def warm_up_cookies(self) -> None:
        """Visit NSE browser pages until the server sends usable cookies."""

        errors: list[str] = []

        for path in self.WARM_UP_PATHS:
            url = f"{self.BASE_URL}{path}"
            response = self._request(
                url,
                headers={
                    "Referer": f"{self.BASE_URL}/",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                },
            )

            if response.status_code < 400 and response.text:
                self._cookies_ready = True
                return

            errors.append(
                f"{path} -> HTTP {response.status_code}: {response.text[:120]}"
            )
            time.sleep(0.5)

        raise NSEClientError(
            "Could not warm up NSE cookies. NSE denied all browser warm-up "
            "pages. This can happen when NSE blocks the current IP/network or "
            "when it changes its edge security rules. Attempts: "
            + " | ".join(errors)
        )

    def get_index_constituents(self, index_name: str = "NIFTY 500") -> dict[str, Any]:
        """Fetch the current constituents and index snapshot for an NSE index."""

        encoded_index = quote(index_name)
        payload = self.get_json(f"/api/equity-stockIndices?index={encoded_index}")
        self._validate_data_list(payload, "index constituents")
        return payload

    def get_nifty500_stock_list_csv(self) -> str:
        """Download the official NSE Indices NIFTY 500 stock-list CSV."""

        if not self._cookies_ready:
            self.warm_up_cookies()

        response = self._request(
            f"{self.ARCHIVE_BASE_URL}{self.NIFTY_500_STOCK_LIST_PATH}",
            headers={
                "Accept": "text/csv,application/csv,text/plain,*/*",
                "Accept-Encoding": "gzip, deflate",
                "Referer": (
                    f"{self.BASE_URL}/products-services/indices-nifty500-index"
                ),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
            },
        )
        self._validate_http_response(response, expected_json=False)

        if "Symbol" not in response.text:
            raise NSEClientError("NIFTY 500 CSV did not contain a Symbol column.")

        return response.text

    def get_security_bhavcopy_csv(self, date_text: str) -> str:
        """Download NSE's official full security bhavcopy for DDMMYYYY."""

        if not self._cookies_ready:
            self.warm_up_cookies()

        path = self.SECURITY_BHAVCOPY_PATH_TEMPLATE.format(date=date_text)
        response = self._request_archive_csv(path)
        self._validate_http_response(response, expected_json=False)

        if "SYMBOL" not in response.text or "CLOSE_PRICE" not in response.text:
            raise NSEClientError(f"Bhavcopy for {date_text} did not look valid.")

        return response.text

    def get_equity_eod_history(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        series: str = "EQ",
    ) -> dict[str, Any]:
        """Fetch EOD historical cash-market data for one NSE equity symbol.

        Dates must be in NSE's expected dd-mm-yyyy format.
        """

        encoded_symbol = quote(symbol.upper().strip())
        endpoint = (
            "/api/historical/securityArchives"
            f"?from={from_date}&to={to_date}"
            f"&symbol={encoded_symbol}&dataType=priceVolumeDeliverable"
            f"&series={series}"
        )
        payload = self.get_json(
            endpoint,
            referer=f"{self.BASE_URL}/get-quotes/equity?symbol={encoded_symbol}",
        )
        self._validate_data_list(payload, f"EOD history for {symbol}")
        return payload

    def get_json(
        self,
        endpoint: str,
        referer: str | None = None,
    ) -> dict[str, Any]:
        """GET an NSE API endpoint and return validated JSON."""

        if not self._cookies_ready:
            self.warm_up_cookies()

        response = self._request_api(endpoint, referer=referer)

        # NSE sometimes invalidates cookies. Refresh once before failing.
        if response.status_code in (401, 403):
            self._cookies_ready = False
            time.sleep(1)
            self.warm_up_cookies()
            response = self._request_api(endpoint, referer=referer)

        self._validate_http_response(response, expected_json=True)

        try:
            payload = response.json()
        except ValueError as exc:
            raise NSEClientError("NSE returned a non-JSON response.") from exc

        if not isinstance(payload, dict):
            raise NSEClientError("NSE returned JSON, but it was not an object.")

        return payload

    def _request(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Run one HTTP GET and convert network errors into client errors."""

        try:
            return self.session.get(url, headers=headers, timeout=self.timeout)
        except RequestException as exc:
            raise NSEClientError(f"NSE request failed: {exc}") from exc

    def _request_api(
        self,
        endpoint: str,
        referer: str | None = None,
    ) -> requests.Response:
        """Run one API request using headers that match browser XHR calls."""

        return self._request(
            f"{self.BASE_URL}{endpoint}",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": referer
                or f"{self.BASE_URL}/market-data/live-market-indices",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "X-Requested-With": "XMLHttpRequest",
            },
        )

    def _request_archive_csv(self, path: str) -> requests.Response:
        """Fetch a CSV from NSE archives with archive-friendly headers."""

        headers = {
            "Accept": "text/csv,application/csv,text/plain,*/*",
            "Accept-Encoding": "gzip, deflate",
            "Referer": f"{self.BASE_URL}/all-reports",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }

        last_error: NSEClientError | None = None
        for base_url in (self.ARCHIVE_BASE_URL, self.LEGACY_ARCHIVE_BASE_URL):
            response = self._request(f"{base_url}{path}", headers=headers)
            if response.status_code < 400:
                return response

            last_error = NSEClientError(
                f"NSE archive request failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        if last_error is not None:
            raise last_error

        raise NSEClientError(f"Could not fetch NSE archive path: {path}")

    def _validate_http_response(
        self,
        response: requests.Response,
        expected_json: bool,
    ) -> None:
        """Check status code and basic content before trusting the response."""

        if response.status_code >= 400:
            if response.status_code == 403 and "Access Denied" in response.text:
                raise NSEClientError(
                    "NSE returned HTTP 403 Access Denied. The cookie warm-up "
                    "succeeded, but NSE blocked the API request from this "
                    "machine/network. Try again later, use a normal non-VPN "
                    "connection, or open nseindia.com once in a browser on the "
                    "same network before retrying."
                )

            if response.status_code == 503:
                raise NSEClientError(
                    "NSE returned HTTP 503 Service Unavailable for this "
                    "endpoint. This usually means NSE is temporarily "
                    "throttling/blocking automated historical-data requests "
                    "from the current machine/network. Retry later with a "
                    "slower pause, or try a different non-VPN network."
                )

            raise NSEClientError(
                f"NSE request failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        if not response.text:
            raise NSEClientError("NSE returned an empty response.")

        content_type = response.headers.get("Content-Type", "")
        if expected_json and "json" not in content_type.lower():
            raise NSEClientError(
                "NSE response was not JSON. "
                f"Content-Type was: {content_type or 'missing'}"
            )

    def _validate_data_list(self, payload: dict[str, Any], label: str) -> None:
        """Most NSE market-data responses expose rows under a data list."""

        rows = payload.get("data")
        if not isinstance(rows, list):
            raise NSEClientError(f"NSE {label} response did not contain a data list.")
