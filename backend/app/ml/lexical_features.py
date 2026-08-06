"""Lexical feature extraction for URL-based phishing detection.

This module defines the LexicalFeatureExtractor class, which computes a
comprehensive set of lexical features from a URL string for use in
SentinelWeb's AI-based phishing detection pipeline.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import ParseResult, urlparse

from app.core.logger import get_logger
from app.exceptions.ml_exceptions import FeatureExtractionError

logger = get_logger(__name__)

_TOKEN_SPLIT_PATTERN = re.compile(r"[/\-_.?=&%#@:]+")
_ENCODED_CHAR_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
_PORT_PATTERN = re.compile(r":\d+$")




class LexicalFeatureExtractor:
    """Extracts lexical features from URLs for phishing detection.

    This extractor computes length-based, character-count-based, ratio-based,
    boolean, and statistical features purely from the lexical structure of a
    URL, without performing any network access or DNS resolution.
    """

    def extract(self, url: str) -> dict[str, int | float | bool]:
        """Extract all lexical features from a given URL.

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
            if "://" not in url:
                url = f"http://{url}"

            parsed = urlparse(url)
            self._validate_parsed_url(parsed)

            length_features = self._extract_length_features(url, parsed)
            char_counts = self._extract_character_counts(url)
            ratios = self._extract_ratios(url, char_counts)
            booleans = self._extract_boolean_features(url, parsed)
            statistical = self._extract_statistical_features(url)

            features: dict[str, int | float | bool] = {
                **length_features,
                **char_counts,
                **ratios,
                **booleans,
                **statistical,
            }

            return features

        except FeatureExtractionError:
            raise
        except Exception as exc:
            logger.exception("Failed to extract lexical features for URL: %s", url)
            raise FeatureExtractionError(
                f"Failed to extract lexical features from URL: {url!r}"
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

    def _validate_parsed_url(self, parsed: ParseResult) -> None:
        """Validate that a parsed URL contains a usable network location.

        Args:
            parsed: The parsed URL result.

        Raises:
            FeatureExtractionError: If the parsed URL lacks both a scheme
                and a network location, indicating malformed input.
        """
        if not parsed.netloc:
            logger.error("Invalid URL: Missing hostname.") 
            raise FeatureExtractionError("URL could not be parsed.")

    def _extract_length_features(
        self, url: str, parsed: ParseResult
    ) -> dict[str, int]:
        """Compute length-based features.

        Args:
            url: The original URL string.
            parsed: The parsed URL result.

        Returns:
            A dictionary of length-based feature values.
        """
        return {
            "url_length": len(url),
            "hostname_length": len(parsed.hostname) if parsed.hostname else 0,
            "path_length": len(parsed.path),
            "query_length": len(parsed.query),
            "fragment_length": len(parsed.fragment),
        }

    def _extract_character_counts(self, url: str) -> dict[str, int]:
        """Compute character-count-based features.

        Args:
            url: The original URL string.

        Returns:
            A dictionary of character-count feature values.
        """
        digit_count = sum(1 for char in url if char.isdigit())
        alphabet_count = sum(1 for char in url if char.isalpha())
        special_character_count = len(url) - digit_count - alphabet_count

        return {
            "dot_count": url.count("."),
            "slash_count": url.count("/"),
            "question_count": url.count("?"),
            "equal_count": url.count("="),
            "ampersand_count": url.count("&"),
            "hyphen_count": url.count("-"),
            "underscore_count": url.count("_"),
            "at_count": url.count("@"),
            "percent_count": url.count("%"),
            "colon_count": url.count(":"),
            "tilde_count": url.count("~"),
            "digit_count": digit_count,
            "alphabet_count": alphabet_count,
            "special_character_count": special_character_count,
        }

    def _extract_ratios(
        self, url: str, char_counts: dict[str, int]
    ) -> dict[str, float]:
        """Compute character-ratio-based features.

        Args:
            url: The original URL string.
            char_counts: Previously computed character-count features.

        Returns:
            A dictionary of ratio-based feature values.
        """
        total_length = len(url)
        if total_length == 0:
            return {
                "digit_ratio": 0.0,
                "alphabet_ratio": 0.0,
                "special_character_ratio": 0.0,
            }

        return {
            "digit_ratio": char_counts["digit_count"] / total_length,
            "alphabet_ratio": char_counts["alphabet_count"] / total_length,
            "special_character_ratio": (
                char_counts["special_character_count"] / total_length
            ),
        }

    def _extract_boolean_features(
        self, url: str, parsed: ParseResult
    ) -> dict[str, bool]:
        """Compute boolean lexical features.

        Args:
            url: The original URL string.
            parsed: The parsed URL result.

        Returns:
            A dictionary of boolean feature values.
        """
        url_lower = url.lower()
        return {
            "has_at_symbol": "@" in url,
            "has_encoded_characters": bool(_ENCODED_CHAR_PATTERN.search(url)),
            "has_port": self._has_port(parsed),

        }

    def _has_ip_address(self, hostname: str | None) -> bool:
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

    def _has_port(self, parsed: ParseResult) -> bool:
        """Determine whether the URL explicitly specifies a port.

        Args:
            parsed: The parsed URL result.

        Returns:
            True if a port is present in the network location, False
            otherwise.
        """
        try:
            return parsed.port is not None
        except ValueError:
            return bool(_PORT_PATTERN.search(parsed.netloc))

    def _extract_statistical_features(
        self, url: str
    ) -> dict[str, int | float]:
        """Compute statistical features derived from the URL string.

        Args:
            url: The original URL string.

        Returns:
            A dictionary of statistical feature values, including Shannon
            entropy and token-based length statistics.
        """
        tokens = [token for token in _TOKEN_SPLIT_PATTERN.split(url) if token]

        if tokens:
            token_lengths = [len(token) for token in tokens]
            longest_token_length = max(token_lengths)
            average_token_length = sum(token_lengths) / len(token_lengths)
        else:
            longest_token_length = 0
            average_token_length = 0.0
            
        digit_sequence_count = len(re.findall(r"\d+", url))

        return {
            "shannon_entropy": self._calculate_shannon_entropy(url),
            "longest_token_length": longest_token_length,
            "average_token_length": average_token_length,
            "token_count": len(tokens),
            "unique_character_count": len(set(url)),
            "digit_sequence_count": digit_sequence_count,
        }

    def _calculate_shannon_entropy(self, url: str) -> float:
        """Calculate the Shannon entropy of a URL string.

        Args:
            url: The original URL string.

        Returns:
            The Shannon entropy value in bits, or 0.0 for an empty string.
        """
        if not url:
            return 0.0

        length = len(url)
        frequency = Counter(url)

        entropy = -sum(
            (count / length) * math.log2(count / length)
            for count in frequency.values()
        )

        return round(entropy, 6)