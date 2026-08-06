"""Domain feature extraction for URL-based phishing detection.

This module defines the DomainFeatureExtractor class, which computes a
comprehensive set of domain-level features from a URL string for use in
SentinelWeb's AI-based phishing detection pipeline.
"""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from urllib.parse import urlparse

import tldextract

from app.config import constants
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import FeatureExtractionError

logger = get_logger(__name__)


class DomainFeatureExtractor:
    """Extracts domain-level features from URLs for phishing detection.

    This extractor analyzes the registered domain, subdomains, and
    top-level domain (TLD) of a URL to derive signals commonly associated
    with phishing infrastructure, such as brand impersonation, IP-literal
    hosts, punycode encoding, and known URL shorteners.
    """

    def extract(self, url: str) -> dict[str, int | float | bool | str]:
        """Extract all domain-level features from a given URL.

        Args:
            url: The URL string to analyze.

        Returns:
            A dictionary mapping feature names to their computed integer,
            float, boolean, or string values.

        Raises:
            FeatureExtractionError: If the URL is empty, malformed, or
                cannot be parsed into its constituent domain components.
        """
        self._validate_url(url)

        try:
            if "://" not in url:
                url = f"http://{url}"

            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            self._validate_hostname(hostname)

            extracted = tldextract.extract(url)

            domain = extracted.domain or ""
            subdomain = extracted.subdomain or ""
            tld = extracted.suffix or ""
            registered_domain = ".".join(
                part for part in [extracted.domain, extracted.suffix] if part
            )
            is_ip = self._is_ip_address(hostname)
            subdomain_parts = self._get_subdomain_parts(subdomain)

            features: dict[str, int | float | bool | str] = {
                "domain_length": len(registered_domain),
                "hostname_label_count": len(hostname.split(".")),
                "subdomain_count": len(subdomain_parts),
                "tld_length": len(tld),
                # "is_com": int(tld.lower() == "com"),
                "is_ip": is_ip,
                "is_shortener": self._is_shortener(registered_domain),
                "known_safe_tld": self._is_safe_tld(tld),
                "known_risky_tld": self._is_risky_tld(tld),
                
                "punycode_domain": self._is_punycode(hostname),
                "repeated_subdomain": self._has_repeated_subdomain(
                    subdomain_parts
                ),
                "numeric_subdomain": self._has_numeric_subdomain(
                    subdomain_parts
                ),
                "domain_entropy": self._calculate_domain_entropy(
                    registered_domain
                ),
            }

            return features

        except FeatureExtractionError:
            raise
        except Exception as exc:
            logger.exception("Failed to extract domain features for URL: %s", url)
            raise FeatureExtractionError(
                f"Failed to extract domain features from URL: {url!r}"
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

    def _validate_hostname(self, hostname: str) -> None:
        """Validate that a hostname could be extracted from the URL.

        Args:
            hostname: The hostname extracted from the parsed URL.

        Raises:
            FeatureExtractionError: If no hostname could be determined.
        """
        if not hostname:
            logger.error("URL does not contain a valid hostname.")
            raise FeatureExtractionError(
                "URL could not be parsed into a valid hostname."
            )

    def _is_ip_address(self, hostname: str) -> bool:
        """Determine whether a hostname is a literal IP address.

        Args:
            hostname: The hostname extracted from the parsed URL.

        Returns:
            True if the hostname is a valid IPv4 or IPv6 address, False
            otherwise.
        """
        candidate = hostname.strip("[]")
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            return False

    def _get_subdomain_parts(self, subdomain: str) -> list[str]:
        """Split a subdomain string into its constituent labels.

        Args:
            subdomain: The subdomain portion of the extracted URL.

        Returns:
            A list of non-empty subdomain labels.
        """
        if not subdomain:
            return []
        return [part for part in subdomain.split(".") if part]

    def _is_shortener(self, registered_domain: str) -> bool:
        """Determine whether the registered domain is a known URL shortener.

        Args:
            registered_domain: The registered domain (domain + TLD).

        Returns:
            True if the domain matches a known URL shortener, False
            otherwise.
        """
        return registered_domain.lower() in constants.URL_SHORTENERS

    def _is_safe_tld(self, tld: str) -> bool:
        """Determine whether the TLD is considered generally safe.

        Args:
            tld: The top-level domain suffix.

        Returns:
            True if the TLD is in the configured safe TLD list, False
            otherwise.
        """
        return tld.lower() in constants.SAFE_TLDS

    def _is_risky_tld(self, tld: str) -> bool:
        """Determine whether the TLD is commonly associated with abuse.

        Args:
            tld: The top-level domain suffix.

        Returns:
            True if the TLD is in the configured risky TLD list, False
            otherwise.
        """
        return tld.lower() in constants.RISKY_TLDS

    def _is_simple_hostname(
        self,
        hostname: str,
        subdomain_parts: list[str],
    ) -> bool:
        """Detect a short, clean hostname shape without suspicious adornments."""
        hostname_lower = hostname.lower()
        if not hostname_lower or self._is_ip_address(hostname_lower):
            return False

        return (
            len(subdomain_parts) == 0
            and "-" not in hostname_lower
            and not any(char.isdigit() for char in hostname_lower)
        )

    def _is_punycode(self, hostname: str) -> bool:
        """Determine whether the hostname contains punycode-encoded labels.

        Args:
            hostname: The hostname extracted from the parsed URL.

        Returns:
            True if any label of the hostname starts with the punycode
            prefix, False otherwise.
        """
        labels = hostname.lower().split(".")
        return any(label.startswith("xn--") for label in labels)

    def _has_repeated_subdomain(self, subdomain_parts: list[str]) -> bool:
        """Determine whether any subdomain label is repeated.

        Args:
            subdomain_parts: The list of subdomain labels.

        Returns:
            True if at least one subdomain label occurs more than once,
            False otherwise.
        """
        if not subdomain_parts:
            return False
        label_counts = Counter(subdomain_parts)
        return any(count > 1 for count in label_counts.values())

    def _has_numeric_subdomain(self, subdomain_parts: list[str]) -> bool:
        """Determine whether any subdomain label is purely numeric.

        Args:
            subdomain_parts: The list of subdomain labels.

        Returns:
            True if at least one subdomain label consists entirely of
            digits, False otherwise.
        """
        return any(part.isdigit() for part in subdomain_parts)

    def _calculate_domain_entropy(self, registered_domain: str) -> float:
        """Calculate the Shannon entropy of the registered domain string.

        Args:
            registered_domain: The registered domain (domain + TLD).

        Returns:
            The Shannon entropy value in bits, or 0.0 for an empty string.
        """
        if not registered_domain:
            return 0.0

        length = len(registered_domain)
        frequency = Counter(registered_domain)

        entropy = -sum(
            (count / length) * math.log2(count / length)
            for count in frequency.values()
        )

        return round(entropy, 6)