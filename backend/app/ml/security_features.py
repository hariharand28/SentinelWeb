"""Security feature extraction for URL-based phishing detection.

This module defines the SecurityFeatureExtractor class, which computes a
comprehensive set of security-relevant lexical and structural features
from a URL string for use in SentinelWeb's AI-based phishing detection
pipeline. No network or HTTP requests are made; all analysis is purely
static and derived from the URL string itself.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import ParseResult, parse_qsl, urlparse

from app.config import constants
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import FeatureExtractionError

logger = get_logger(__name__)

_SUSPICIOUS_PATH_KEYWORDS: dict[str, str] = {
    "path_contains_login": "login",
    "path_contains_verify": "verify",
    "path_contains_update": "update",
    "path_contains_account": "account",
    "path_contains_bank": "bank",
}

_ENCODED_CHAR_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
_MULTI_ENCODED_PATTERN = re.compile(r"%25[0-9A-Fa-f]{2}")


class SecurityFeatureExtractor:
    """Extracts security-relevant features from URLs for phishing detection.

    This extractor analyzes scheme, port, hostname, path, query, and
    encoding characteristics of a URL to derive signals commonly
    associated with phishing and malicious infrastructure. All analysis
    is static; no HTTP requests, DNS resolution, or certificate retrieval
    is performed.
    """

    def extract(self, url: str) -> dict[str, int | float | bool]:
        """Extract all security features from a given URL.

        Args:
            url: The URL string to analyze.

        Returns:
            A dictionary mapping feature names to their computed integer,
            float, or boolean values.

        Raises:
            FeatureExtractionError: If the URL is empty, malformed, or
                cannot be parsed into its constituent components.
        """
        self._validate_url(url)

        try:
            normalized_url = self._normalize_url(url)
            parsed = urlparse(normalized_url)
            self._validate_parsed_url(parsed)

            hostname = parsed.hostname or ""

            features: dict[str, int | float | bool] = {
                **self._extract_scheme_features(parsed),
                **self._extract_port_features(parsed),
                **self._extract_hostname_features(hostname),
                **self._extract_path_features(parsed.path),
                **self._extract_query_features(parsed.query),
                **self._extract_encoding_features(normalized_url),
                **self._extract_url_indicators(normalized_url, hostname),
                "ssl_feature_available": False,
            }

            return features

        except FeatureExtractionError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to extract security features for URL: %s", url
            )
            raise FeatureExtractionError(
                f"Failed to extract security features from URL: {url!r}"
            ) from exc

    def _validate_url(self, url: str) -> None:
        """Validate that the URL is a non-empty string.

        Args:
            url: The URL string to validate.

        Raises:
            FeatureExtractionError: If the URL is not a valid non-empty
                string.
        """
        if not isinstance(url, str) or not url.strip():
            logger.error("Invalid URL provided: %r", url)
            raise FeatureExtractionError("URL must be a non-empty string.")

    def _normalize_url(self, url: str) -> str:
        """Prepend a default scheme to the URL if one is missing.

        Args:
            url: The raw URL string.

        Returns:
            The URL string guaranteed to include a scheme.
        """
        stripped = url.strip()
        if "://" not in stripped:
            return f"http://{stripped}"
        return stripped

    def _validate_parsed_url(self, parsed: ParseResult) -> None:
        """Validate that a parsed URL contains a usable network location.

        Args:
            parsed: The parsed URL result.

        Raises:
            FeatureExtractionError: If the parsed URL lacks a network
                location, indicating malformed input.
        """
        if not parsed.netloc:
            logger.error("URL could not be parsed into valid components.")
            raise FeatureExtractionError("URL could not be parsed.")

    def _extract_scheme_features(self, parsed: ParseResult) -> dict[str, bool]:
        """Compute scheme-based features.

        Args:
            parsed: The parsed URL result.

        Returns:
            A dictionary of scheme-based feature values.
        """
        scheme = parsed.scheme.lower()
        return {
            "has_http": scheme == "http",
        }

    def _extract_port_features(self, parsed: ParseResult) -> dict[str, bool]:
        """Compute port-based features.

        Args:
            parsed: The parsed URL result.

        Returns:
            A dictionary of port-based feature values.
        """
        try:
            port = parsed.port
        except ValueError:
            port = None

        has_port = port is not None
        scheme = parsed.scheme.lower()
        default_port = constants.DEFAULT_SCHEME_PORTS.get(scheme)

        uses_default_port = has_port and port == default_port
        uses_non_standard_port = has_port and port != default_port

        return {
            "uses_default_port": uses_default_port,
            "uses_non_standard_port": uses_non_standard_port,
        }

    def _extract_hostname_features(self, hostname: str) -> dict[str, bool]:
        """Compute hostname-based features.

        Args:
            hostname: The hostname extracted from the parsed URL.

        Returns:
            A dictionary of hostname-based feature values.
        """
        return {
            "hostname_is_ip": self._is_ip_address(hostname),
            "hostname_contains_dash": "-" in hostname,
            "hostname_contains_digit": any(
                char.isdigit() for char in hostname
            ),
        }

    def _is_ip_address(self, hostname: str) -> bool:
        """Determine whether a hostname is a literal IP address.

        Args:
            hostname: The hostname extracted from the parsed URL.

        Returns:
            True if the hostname is a valid IPv4 or IPv6 address, False
            otherwise.
        """
        if not hostname:
            return False

        candidate = hostname.strip("[]")
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            return False

    def _extract_path_features(self, path: str) -> dict[str, int | bool]:
        """Compute path-based features.

        Args:
            path: The path component of the parsed URL.

        Returns:
            A dictionary of path-based feature values.
        """
        lowered_path = path.lower()
        segments = [segment for segment in path.split("/") if segment]

        features: dict[str, int | bool] = {"path_depth": len(segments)}

        for feature_name, keyword in _SUSPICIOUS_PATH_KEYWORDS.items():
            features[feature_name] = keyword in lowered_path

        return features

    def _extract_query_features(self, query: str) -> dict[str, int | bool]:
        """Compute query-based features.

        Args:
            query: The query component of the parsed URL.

        Returns:
            A dictionary of query-based feature values.
        """
        parameters = parse_qsl(query, keep_blank_values=True)
        parameter_count = len(parameters)

        suspicious_parameter_names = any(
            name.lower() in constants.SUSPICIOUS_QUERY_PARAM_NAMES
            for name, _ in parameters
        )

        return {
            "query_parameter_count": parameter_count,
            "long_query": len(query) > constants.LONG_QUERY_LENGTH_THRESHOLD,
            "suspicious_parameter_names": suspicious_parameter_names,
        }

    def _extract_encoding_features(self, url: str) -> dict[str, bool]:
        """Compute percent-encoding-based features.

        Args:
            url: The normalized URL string.

        Returns:
            A dictionary of encoding-based feature values.
        """
        has_percent_encoding = bool(_ENCODED_CHAR_PATTERN.search(url))
        multiple_encoding = bool(_MULTI_ENCODED_PATTERN.search(url))

        return {
            "has_percent_encoding": has_percent_encoding,
            "multiple_encoding": multiple_encoding,
        }

    def _extract_url_indicators(
        self, url: str, hostname: str
    ) -> dict[str, int | bool]:
        """Compute general structural URL risk indicators.

        Args:
            url: The normalized URL string.
            hostname: The hostname extracted from the parsed URL.

        Returns:
            A dictionary of URL indicator feature values.
        """
        dot_count = hostname.count(".")
        hyphen_count = hostname.count("-")
        url_length = len(url)

        return {
            "multiple_subdomains": dot_count
            > constants.MULTIPLE_SUBDOMAIN_DOT_THRESHOLD,
            "excessive_dots": dot_count > constants.EXCESSIVE_DOT_THRESHOLD,
            "excessive_hyphens": hyphen_count
            > constants.EXCESSIVE_HYPHEN_THRESHOLD,
            "long_url": url_length > constants.LONG_URL_LENGTH_THRESHOLD,
            "very_long_url": url_length
            > constants.VERY_LONG_URL_LENGTH_THRESHOLD,
        }