import logging
import os
import ssl
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib import parse
from urllib.error import HTTPError

# Create an SSL context with configurable certificate verification
# Set APOLLO_SSL_VERIFY=false to disable SSL verification (not recommended for production)
ssl_context = ssl.create_default_context()
ssl_context.set_ciphers("HIGH:!DH:!aNULL")

# Only disable SSL verification if explicitly configured (defaults to secure)
if os.environ.get("APOLLO_SSL_VERIFY", "true").lower() == "false":
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    logging.getLogger(__name__).warning(
        "SSL certificate verification is disabled for Apollo. This is insecure and should not be used in production."
    )

# Create an opener object and pass in a custom SSL context
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))

urllib.request.install_opener(opener)

logger = logging.getLogger(__name__)


def http_request(url: str, timeout: int | float, headers: Mapping[str, str] = {}) -> tuple[int, str | None]:
    try:
        request = urllib.request.Request(url, headers=dict(headers))
        res = urllib.request.urlopen(request, timeout=timeout)
        body = res.read().decode("utf-8")
        return res.code, body
    except HTTPError as e:
        if e.code == 304:
            logger.warning("http_request error,code is 304, maybe you should check secret")
            return 304, None
        logger.warning("http_request error,code is %d, msg is %s", e.code, e.msg)
        raise e


def url_encode(params: dict[str, Any]) -> str:
    return parse.urlencode(params)


def makedirs_wrapper(path: str) -> None:
    os.makedirs(path, exist_ok=True)
