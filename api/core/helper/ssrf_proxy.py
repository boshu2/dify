"""
Proxy requests to avoid SSRF
"""

import ipaddress
import logging
import socket
import time
from urllib.parse import urlparse

import httpx

from configs import dify_config
from core.helper.http_client_pooling import get_pooled_http_client

logger = logging.getLogger(__name__)


class SSRFProtectionError(ValueError):
    """Raised when a request is blocked due to SSRF protection."""

    pass


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, or otherwise reserved."""
    # Cloud metadata service IPs (AWS/GCP/Azure)
    cloud_metadata_ips = {"169.254.169.254", "169.254.169.253"}

    try:
        ip = ipaddress.ip_address(ip_str)
        # Check for private, loopback, link-local, reserved, and multicast addresses
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip_str in cloud_metadata_ips
            or ip_str.startswith("fd00:")  # IPv6 unique local addresses
        )
    except ValueError:
        return False


def _resolve_hostname(hostname: str) -> list[str]:
    """Resolve hostname to IP addresses."""
    try:
        # Get all IP addresses for the hostname
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return list({result[4][0] for result in results})
    except socket.gaierror:
        return []


def validate_url(url: str) -> None:
    """
    Validate URL to prevent SSRF attacks.

    Raises SSRFProtectionError if the URL points to a private/reserved IP address.
    """
    # Skip validation if SSRF protection is disabled via config
    if not dify_config.SSRF_PROTECTION_ENABLED:
        return

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFProtectionError(f"Invalid URL: {e}")

    # Check scheme
    if parsed.scheme not in ("http", "https"):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFProtectionError("Invalid URL: missing hostname")

    # Check if hostname is an IP address
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_private_ip(str(ip)):
            raise SSRFProtectionError(
                f"Access to private/reserved IP address {hostname} is blocked for security reasons."
            )
        return
    except ValueError:
        # Not an IP address, proceed to DNS resolution
        pass

    # Resolve hostname and check all resulting IPs
    resolved_ips = _resolve_hostname(hostname)
    if not resolved_ips:
        # Allow the request to proceed - the HTTP client will handle DNS resolution failure
        return

    for ip_str in resolved_ips:
        if _is_private_ip(ip_str):
            raise SSRFProtectionError(
                f"Hostname {hostname} resolves to private/reserved IP address {ip_str}. "
                "Access is blocked for security reasons."
            )


SSRF_DEFAULT_MAX_RETRIES = dify_config.SSRF_DEFAULT_MAX_RETRIES

BACKOFF_FACTOR = 0.5
STATUS_FORCELIST = [429, 500, 502, 503, 504]

_SSL_VERIFIED_POOL_KEY = "ssrf:verified"
_SSL_UNVERIFIED_POOL_KEY = "ssrf:unverified"
_SSRF_CLIENT_LIMITS = httpx.Limits(
    max_connections=dify_config.SSRF_POOL_MAX_CONNECTIONS,
    max_keepalive_connections=dify_config.SSRF_POOL_MAX_KEEPALIVE_CONNECTIONS,
    keepalive_expiry=dify_config.SSRF_POOL_KEEPALIVE_EXPIRY,
)


class MaxRetriesExceededError(ValueError):
    """Raised when the maximum number of retries is exceeded."""

    pass


def _create_proxy_mounts() -> dict[str, httpx.HTTPTransport]:
    return {
        "http://": httpx.HTTPTransport(
            proxy=dify_config.SSRF_PROXY_HTTP_URL,
        ),
        "https://": httpx.HTTPTransport(
            proxy=dify_config.SSRF_PROXY_HTTPS_URL,
        ),
    }


def _build_ssrf_client(verify: bool) -> httpx.Client:
    if dify_config.SSRF_PROXY_ALL_URL:
        return httpx.Client(
            proxy=dify_config.SSRF_PROXY_ALL_URL,
            verify=verify,
            limits=_SSRF_CLIENT_LIMITS,
        )

    if dify_config.SSRF_PROXY_HTTP_URL and dify_config.SSRF_PROXY_HTTPS_URL:
        return httpx.Client(
            mounts=_create_proxy_mounts(),
            verify=verify,
            limits=_SSRF_CLIENT_LIMITS,
        )

    return httpx.Client(verify=verify, limits=_SSRF_CLIENT_LIMITS)


def _get_ssrf_client(ssl_verify_enabled: bool) -> httpx.Client:
    if not isinstance(ssl_verify_enabled, bool):
        raise ValueError("SSRF client verify flag must be a boolean")

    return get_pooled_http_client(
        _SSL_VERIFIED_POOL_KEY if ssl_verify_enabled else _SSL_UNVERIFIED_POOL_KEY,
        lambda: _build_ssrf_client(verify=ssl_verify_enabled),
    )


def make_request(method, url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    # Validate URL to prevent SSRF attacks (unless explicitly disabled via ssrf_validate=False)
    if kwargs.pop("ssrf_validate", True):
        validate_url(url)

    if "allow_redirects" in kwargs:
        allow_redirects = kwargs.pop("allow_redirects")
        if "follow_redirects" not in kwargs:
            kwargs["follow_redirects"] = allow_redirects

    if "timeout" not in kwargs:
        kwargs["timeout"] = httpx.Timeout(
            timeout=dify_config.SSRF_DEFAULT_TIME_OUT,
            connect=dify_config.SSRF_DEFAULT_CONNECT_TIME_OUT,
            read=dify_config.SSRF_DEFAULT_READ_TIME_OUT,
            write=dify_config.SSRF_DEFAULT_WRITE_TIME_OUT,
        )

    # prioritize per-call option, which can be switched on and off inside the HTTP node on the web UI
    verify_option = kwargs.pop("ssl_verify", dify_config.HTTP_REQUEST_NODE_SSL_VERIFY)
    client = _get_ssrf_client(verify_option)

    retries = 0
    while retries <= max_retries:
        try:
            response = client.request(method=method, url=url, **kwargs)

            if response.status_code not in STATUS_FORCELIST:
                return response
            else:
                logger.warning(
                    "Received status code %s for URL %s which is in the force list",
                    response.status_code,
                    url,
                )

        except httpx.RequestError as e:
            logger.warning("Request to URL %s failed on attempt %s: %s", url, retries + 1, e)
            if max_retries == 0:
                raise

        retries += 1
        if retries <= max_retries:
            time.sleep(BACKOFF_FACTOR * (2 ** (retries - 1)))
    raise MaxRetriesExceededError(f"Reached maximum retries ({max_retries}) for URL {url}")


def get(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("GET", url, max_retries=max_retries, **kwargs)


def post(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("POST", url, max_retries=max_retries, **kwargs)


def put(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("PUT", url, max_retries=max_retries, **kwargs)


def patch(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("PATCH", url, max_retries=max_retries, **kwargs)


def delete(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("DELETE", url, max_retries=max_retries, **kwargs)


def head(url, max_retries=SSRF_DEFAULT_MAX_RETRIES, **kwargs):
    return make_request("HEAD", url, max_retries=max_retries, **kwargs)
