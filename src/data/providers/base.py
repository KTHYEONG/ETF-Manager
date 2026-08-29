"""Shared provider error type and fail-closed retrying HTTP GET."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from src.data.storage import JSONValue

logger = logging.getLogger(__name__)

MAX_ATTEMPTS: Final[int] = 5
DEFAULT_TIMEOUT_S: Final[float] = 30.0
QueryParam: TypeAlias = Mapping[str, str | int | float | bool | None]


class ProviderError(RuntimeError):
    """Vendor HTTP or payload contract failure; message must not include secrets."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Decoded JSON document plus the exact wire bytes for raw archiving."""

    status_code: int
    content: bytes
    body: JSONValue


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and _is_retryable_status(exc.response.status_code)


def _decode_json(response: httpx.Response) -> JSONValue:
    try:
        decoded: JSONValue = response.json()
    except ValueError as exc:
        raise ProviderError("provider response is not valid JSON") from exc
    if decoded is None or isinstance(decoded, dict | list):
        return decoded
    raise ProviderError("provider JSON root must be an object or array")


def get_json(
    client: httpx.Client,
    url: str,
    *,
    params: QueryParam | None = None,
    headers: Mapping[str, str] | None = None,
    retry_on_429: bool = True,
) -> ProviderResponse:
    """GET one JSON document with the vendor retry policy.

    Retries only HTTP 429/5xx responses up to MAX_ATTEMPTS; other statuses,
    transport failures, and undecodable bodies raise ProviderError carrying
    the status code or failure kind, never the raw body or the request URL.
    """

    def _is_retryable_for_policy(exc: BaseException) -> bool:
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        code = exc.response.status_code
        if code == 429:
            return retry_on_429
        return code >= 500

    def _attempt() -> ProviderResponse:
        response = client.get(url, params=params, headers=headers)
        if _is_retryable_status(response.status_code):
            if response.status_code == 429 and not retry_on_429:
                raise ProviderError(f"provider returned HTTP {response.status_code}")
            # Raised so the surrounding tenacity policy can retry it.
            response.raise_for_status()
        if response.status_code >= 400:
            raise ProviderError(f"provider returned HTTP {response.status_code}")
        result = ProviderResponse(
            status_code=response.status_code,
            content=response.content,
            body=_decode_json(response),
        )
        logger.info(
            "[DATA] event=provider_get host=%s status=%d bytes=%d",
            httpx.URL(url).host,
            response.status_code,
            len(response.content),
        )
        return result

    try:
        retrier = Retrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(_is_retryable_for_policy),
            reraise=True,
        )
        return retrier(_attempt)
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"provider returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"transport failure ({type(exc).__name__})") from exc
