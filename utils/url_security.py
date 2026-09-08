from urllib.parse import urlparse
import ipaddress
import socket


ALLOWED_DOMAINS = {
    "stackoverflow.com",
    "stackexchange.com",
    "api.stackexchange.com",
    "geocoding-api.open-meteo.com",
    "api.open-meteo.com",
}


def is_private_or_local(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        pass

    try:
        resolved = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(resolved)

        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except (socket.gaierror, ValueError):
        return False


def validate_external_url(url: str) -> dict:
    if not isinstance(url, str) or not url.strip():
        return {
            "allowed": False,
            "reason": "URL is empty or invalid.",
        }

    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return {
            "allowed": False,
            "reason": "URL could not be parsed.",
        }

    if parsed.scheme != "https":
        return {
            "allowed": False,
            "reason": "Only HTTPS URLs are allowed.",
        }

    hostname = parsed.hostname

    if not hostname:
        return {
            "allowed": False,
            "reason": "URL has no valid hostname.",
        }

    hostname = hostname.lower()

    if is_private_or_local(hostname):
        return {
            "allowed": False,
            "reason": "Private or local network addresses are not allowed.",
        }

    if not (
        hostname in ALLOWED_DOMAINS
        or any(hostname.endswith("." + domain) for domain in ALLOWED_DOMAINS)
    ):
        return {
            "allowed": False,
            "reason": "URL domain is not in the approved source allowlist.",
        }

    return {
        "allowed": True,
        "reason": "URL passed security validation.",
    }