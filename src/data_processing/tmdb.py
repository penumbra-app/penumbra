"""Small TMDB client with bounded retries and reproducible disk snapshots.

Only explicit enrichment commands make requests. Credentials never appear in
cache files, error messages, or model inputs.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_URL = "https://api.themoviedb.org/3"
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
CREDENTIAL_NAMES = ("TMDB_API_READ_ACCESS_TOKEN", "TMDB_ACCESS_TOKEN", "TMDB_API_KEY")


class TMDBError(RuntimeError):
    """A safe error message that does not include request URLs or credentials."""


class TMDBAuthError(TMDBError):
    pass


class TMDBNotFound(TMDBError):
    pass


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TMDBError("Unexpected TMDB redirect; credentials were not forwarded")


def read_credentials(env_file: str | Path = ".env") -> dict[str, str]:
    """Read only TMDB secrets. Environment wins over a local dotenv file.

    Simple KEY=value / quoted value syntax is supported, without shell execution
    or variable interpolation. File contents are never included in errors.
    """
    configured = {name: os.environ[name].strip() for name in CREDENTIAL_NAMES
                  if os.environ.get(name, "").strip()}
    if configured:
        return configured
    path = Path(env_file)
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or name not in CREDENTIAL_NAMES:
            continue
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise TMDBAuthError("Invalid quoted TMDB credential in .env")
            value = value[1:end]
        else:
            value = value.split(" #", 1)[0].strip()
        if value:
            configured[name] = value
    return configured


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as destination:
            temporary = Path(destination.name)
            destination.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class MovieResponse:
    payload: dict
    fetched_at: str
    from_cache: bool = False


def _valid_payload(payload, tmdb_id):
    return (isinstance(payload, dict) and type(payload.get("id")) is int
            and payload["id"] == tmdb_id
            and isinstance(payload.get("credits"), dict)
            and isinstance(payload["credits"].get("cast"), list)
            and isinstance(payload["credits"].get("crew"), list)
            and isinstance(payload.get("keywords"), dict)
            and isinstance(payload["keywords"].get("keywords"), list))


class TMDBClient:
    def __init__(
        self, *, token: str | None = None, api_key: str | None = None,
        cache_dir: str | Path = "data/tmdb_cache", language: str = "en-US",
        timeout: float = 15, max_retries: int = 3, min_interval: float = .25,
        opener=None, sleep=time.sleep, clock=time.monotonic,
    ):
        if not re.fullmatch(r"[a-z]{2}(?:-[A-Z]{2})?", language):
            raise ValueError("language must have the form en or en-US")
        for name, value in (("timeout", timeout), ("min_interval", min_interval)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if timeout == 0:
            raise ValueError("timeout must be positive")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5")
        for value in (token, api_key):
            if value is not None and (not isinstance(value, str) or any(c.isspace() for c in value)):
                raise TMDBAuthError("TMDB credentials must not contain whitespace")
        self._token, self._api_key = token, api_key
        self.cache_dir, self.language = Path(cache_dir), language
        self.timeout, self.max_retries, self.min_interval = timeout, max_retries, min_interval
        self._open = opener or build_opener(_NoRedirects()).open
        self._sleep, self._clock, self._last_request = sleep, clock, None

    @classmethod
    def from_environment(cls, *, env_file=".env", **kwargs):
        secrets = read_credentials(env_file)
        return cls(token=secrets.get("TMDB_API_READ_ACCESS_TOKEN") or secrets.get("TMDB_ACCESS_TOKEN"),
                   api_key=secrets.get("TMDB_API_KEY"), **kwargs)

    def _cache_path(self, tmdb_id: int) -> Path:
        if type(tmdb_id) is not int or tmdb_id <= 0:
            raise ValueError("tmdb_id must be a positive integer")
        return self.cache_dir / f"{tmdb_id}-{self.language}.json"

    def cached_movie(self, tmdb_id: int) -> MovieResponse | None:
        path = self._cache_path(tmdb_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or data.get("schema_version") != 1
                    or data.get("language") != self.language
                    or not _valid_payload(data.get("payload"), tmdb_id)):
                return None
            fetched_at = data["fetched_at"]
            if not isinstance(fetched_at, str) or datetime.fromisoformat(fetched_at).utcoffset() is None:
                return None
        except (OSError, ValueError, TypeError, KeyError):
            return None
        return MovieResponse(data["payload"], fetched_at, True)

    def _retry_delay(self, retry_after, attempt):
        delay = min(2 ** attempt, 30)
        if retry_after:
            try:
                parsed = float(retry_after)
            except (TypeError, ValueError):
                try:
                    parsed = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    parsed = delay
            if math.isfinite(parsed):
                if parsed > 30:
                    raise TMDBError("TMDB requested a longer cooldown; retry this resumable command later")
                delay = max(delay, parsed)
        self._sleep(delay)

    def get_movie(self, tmdb_id: int, *, refresh: bool = False) -> MovieResponse:
        path = self._cache_path(tmdb_id)
        cached = None if refresh else self.cached_movie(tmdb_id)
        if cached is not None:
            return cached
        if not self._token and not self._api_key:
            raise TMDBAuthError("Set TMDB_API_READ_ACCESS_TOKEN (or TMDB_API_KEY) in your environment or .env")
        query = {"language": self.language, "append_to_response": "credits,keywords"}
        headers = {"Accept": "application/json", "User-Agent": "Flick/0.1 (movie metadata enrichment)"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        else:
            query["api_key"] = self._api_key
        request = Request(f"{API_URL}/movie/{tmdb_id}?{urlencode(query)}", headers=headers)
        for attempt in range(self.max_retries + 1):
            if self._last_request is not None:
                delay = self.min_interval - (self._clock() - self._last_request)
                if delay > 0:
                    self._sleep(delay)
            self._last_request = self._clock()
            try:
                with self._open(request, timeout=self.timeout) as response:
                    payload = json.load(response)
            except HTTPError as error:
                status, retry_after = error.code, error.headers.get("Retry-After") if error.headers else None
                error.close()
                if status in (401, 403):
                    raise TMDBAuthError("TMDB rejected the credential; check your API token/key") from None
                if status == 404:
                    raise TMDBNotFound(f"TMDB movie {tmdb_id} was not found") from None
                if (status == 429 or 500 <= status < 600) and attempt < self.max_retries:
                    self._retry_delay(retry_after, attempt)
                    continue
                raise TMDBError(f"TMDB request failed (HTTP {status}); cached progress is preserved") from None
            except (URLError, TimeoutError, OSError):
                if attempt < self.max_retries:
                    self._retry_delay(None, attempt)
                    continue
                raise TMDBError("Could not reach TMDB after retries; cached progress is preserved") from None
            except (ValueError, UnicodeError):
                raise TMDBError("TMDB returned invalid JSON") from None
            if not _valid_payload(payload, tmdb_id):
                raise TMDBError("TMDB returned incomplete or mismatched movie metadata")
            fetched_at = datetime.now(timezone.utc).isoformat()
            atomic_write(path, json.dumps({"schema_version": 1, "language": self.language,
                                          "fetched_at": fetched_at, "payload": payload}, ensure_ascii=False) + "\n")
            return MovieResponse(payload, fetched_at)
        raise AssertionError("Unreachable retry state")
