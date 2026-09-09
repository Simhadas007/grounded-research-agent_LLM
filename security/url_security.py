"""
External URL security for the Grounded Research Agent.

Security goals
--------------
Protect outbound HTTP requests against:

- SSRF (Server-Side Request Forgery)
- localhost access
- loopback addresses
- private IP addresses
- link-local addresses
- multicast/reserved addresses
- non-HTTPS URLs
- credentials embedded in URLs
- unexpected ports
- malformed URLs

This module is a SECURITY BOUNDARY.

A caller should validate the URL BEFORE making an HTTP request.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_SCHEMES = {"https"}

# Your current application only needs normal HTTPS traffic.
# Port 443 is the safest default for the external APIs used by the project.
ALLOWED_PORTS = {443}


# ---------------------------------------------------------------------------
# Hostname validation
# ---------------------------------------------------------------------------

def _is_ip_address(hostname: str) -> bool:
    """
    Determine whether hostname is already an IPv4/IPv6 address.
    """

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _is_private_or_reserved_ip(hostname: str) -> bool:
    """
    Reject IP addresses that should never be contacted by the application.

    This includes:
    - private networks
    - loopback
    - link-local
    - multicast
    - unspecified
    - reserved
    - documentation/test ranges where applicable
    """

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return any(
        [
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_unspecified,
            address.is_reserved,
        ]
    )


def _resolve_hostname(hostname: str) -> list[str]:
    """
    Resolve a hostname and return all resolved IP addresses.

    Security reason:
    Checking only the hostname string is insufficient.

    Example attack:

        attacker.example
              ↓
        resolves to 127.0.0.1

    Therefore DNS-resolved addresses must also be validated.
    """

    try:
        results = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(
            f"Unable to resolve external hostname: {hostname}"
        ) from exc

    addresses: list[str] = []

    for result in results:
        sockaddr = result[4]

        if not sockaddr:
            continue

        ip = sockaddr[0]

        if ip not in addresses:
            addresses.append(ip)

    if not addresses:
        raise ValueError(
            f"Hostname did not resolve to an IP address: {hostname}"
        )

    return addresses


# ---------------------------------------------------------------------------
# Main URL validator
# ---------------------------------------------------------------------------

def validate_external_url(
    url: str,
    *,
    allowed_hosts: set[str] | None = None,
) -> dict:
    """
    Validate a URL before an outbound HTTP request.

    Parameters
    ----------
    url:
        URL that the application wants to contact.

    allowed_hosts:
        Optional explicit hostname allowlist.

        Example:
            {"api.open-meteo.com"}

        When supplied, the hostname MUST exactly match one of these hosts.

    Returns
    -------
    dict
        {
            "allowed": bool,
            "url": str,
            "reason": str
        }

    SECURITY POLICY
    ---------------
    This function fails closed.

    Anything suspicious or ambiguous is rejected.
    """

    if not isinstance(url, str):
        return {
            "allowed": False,
            "url": "",
            "reason": "URL must be a string.",
        }

    url = url.strip()

    if not url:
        return {
            "allowed": False,
            "url": "",
            "reason": "URL cannot be empty.",
        }

    # ---------------------------------------------------------------
    # Parse URL
    # ---------------------------------------------------------------

    try:
        parsed = urlparse(url)
    except Exception:
        return {
            "allowed": False,
            "url": url,
            "reason": "URL parsing failed.",
        }

    # ---------------------------------------------------------------
    # HTTPS only
    # ---------------------------------------------------------------

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return {
            "allowed": False,
            "url": url,
            "reason": "Only HTTPS URLs are allowed.",
        }

    # ---------------------------------------------------------------
    # Hostname must exist
    # ---------------------------------------------------------------

    hostname = parsed.hostname

    if not hostname:
        return {
            "allowed": False,
            "url": url,
            "reason": "URL does not contain a valid hostname.",
        }

    hostname = hostname.lower().rstrip(".")

    # ---------------------------------------------------------------
    # Reject embedded credentials
    # ---------------------------------------------------------------

    if parsed.username is not None or parsed.password is not None:
        return {
            "allowed": False,
            "url": url,
            "reason": (
                "URLs containing embedded usernames or passwords "
                "are not allowed."
            ),
        }

    # ---------------------------------------------------------------
    # Explicit host allowlist
    # ---------------------------------------------------------------

    if allowed_hosts is not None:

        normalized_allowed_hosts = {
            host.lower().rstrip(".")
            for host in allowed_hosts
        }

        if hostname not in normalized_allowed_hosts:
            return {
                "allowed": False,
                "url": url,
                "reason": (
                    f"Hostname '{hostname}' is not in the external "
                    "API allowlist."
                ),
            }

    # ---------------------------------------------------------------
    # Port validation
    # ---------------------------------------------------------------

    try:
        port = parsed.port
    except ValueError:
        return {
            "allowed": False,
            "url": url,
            "reason": "URL contains an invalid port.",
        }

    if port is not None and port not in ALLOWED_PORTS:
        return {
            "allowed": False,
            "url": url,
            "reason": (
                f"Port {port} is not allowed. "
                "Only HTTPS port 443 is permitted."
            ),
        }

    # ---------------------------------------------------------------
    # Direct IP validation
    # ---------------------------------------------------------------

    if _is_ip_address(hostname):

        if _is_private_or_reserved_ip(hostname):
            return {
                "allowed": False,
                "url": url,
                "reason": (
                    "Direct requests to private, loopback, link-local, "
                    "multicast, unspecified, or reserved IP addresses "
                    "are blocked."
                ),
            }

        return {
            "allowed": True,
            "url": url,
            "reason": "External HTTPS IP address passed security checks.",
        }

    # ---------------------------------------------------------------
    # Block obvious local hostnames
    # ---------------------------------------------------------------

    local_hostnames = {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }

    if hostname in local_hostnames:
        return {
            "allowed": False,
            "url": url,
            "reason": "Localhost destinations are not allowed.",
        }

    # Block common internal/local suffixes.
    if (
        hostname.endswith(".local")
        or hostname.endswith(".localhost")
        or hostname.endswith(".internal")
        or hostname.endswith(".home")
    ):
        return {
            "allowed": False,
            "url": url,
            "reason": "Internal/local hostnames are not allowed.",
        }

    # ---------------------------------------------------------------
    # DNS resolution security check
    # ---------------------------------------------------------------

    try:
        resolved_addresses = _resolve_hostname(hostname)
    except ValueError as exc:
        return {
            "allowed": False,
            "url": url,
            "reason": str(exc),
        }

    for address in resolved_addresses:

        if _is_private_or_reserved_ip(address):
            return {
                "allowed": False,
                "url": url,
                "reason": (
                    f"Hostname '{hostname}' resolves to a private or "
                    f"reserved address ({address}). Possible SSRF target."
                ),
            }

    # ---------------------------------------------------------------
    # Passed
    # ---------------------------------------------------------------

    return {
        "allowed": True,
        "url": url,
        "reason": "External HTTPS URL passed security checks.",
    }


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def require_safe_external_url(
    url: str,
    *,
    allowed_hosts: set[str] | None = None,
) -> str:
    """
    Validate a URL and raise an exception if it is unsafe.

    Useful immediately before an HTTP request:

        safe_url = require_safe_external_url(
            url,
            allowed_hosts={"api.open-meteo.com"},
        )

    Then perform the request using safe_url.
    """

    result = validate_external_url(
        url,
        allowed_hosts=allowed_hosts,
    )

    if not result["allowed"]:
        raise ValueError(
            f"Blocked unsafe external URL: {result['reason']}"
        )

    return result["url"]
