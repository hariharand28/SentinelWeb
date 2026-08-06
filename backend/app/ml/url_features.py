"""URL structural feature extraction for phishing detection.

This module defines the URLFeatureExtractor class, which computes a
comprehensive set of general, structural, component-based, and
statistical features from a URL string for use in SentinelWeb's AI-based
phishing detection pipeline.
"""

from __future__ import annotations

import math
import re
from urllib.parse import ParseResult, urlparse

from app.core.logger import get_logger
from app.exceptions.ml_exceptions import FeatureExtractionError

logger = get_logger(__name__)

_FILE_EXTENSION_PATTERN = re.compile(r"\.([A-Za-z0-9]{1,10})$")



class URLFeatureExtractor:
    """Extracts general and structural features from URLs.

    This extractor analyzes the overall structure of a URL, including
    component lengths, delimiter counts, path composition, and character
    statistics, to derive signals useful for phishing detection.
    """

    def extract(self, url: str) -> dict[str, int | float | bool | str]:
        """Extract all URL structural features from a given URL.

        Args:
            url: The URL string to analyze.

        Returns:
            A dictionary mapping feature names to their computed integer,
            float, boolean, or string values.

        Raises:
            FeatureExtractionError: If the URL is empty, malformed, or
                cannot be parsed into its constituent components.
        """
        self._validate_url(url)

        try:
            normalized_url = self._normalize_url(url)
            parsed = urlparse(normalized_url)
            self._validate_parsed_url(parsed)

            features = {
                **self._extract_structure_features(normalized_url),
                **self._extract_component_features(parsed),
                **self._extract_statistical_features(normalized_url),
                **self._extract_root_features(parsed),
            }
            features.update(
                self._extract_ratio_features(normalized_url, features)
            )

            return features

        except FeatureExtractionError:
            raise
        except Exception  as exc:
            logger.exception(
                "Failed to extract URL features for URL: %s", url
            )
            raise FeatureExtractionError(
                f"Failed to extract URL features from URL: {url!r}"
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
            raise FeatureExtractionError("URL could not be parsed.")


    def _extract_structure_features(self, url: str) -> dict[str, int]:
        """Compute delimiter and structural character counts.

        Args:
            url: The normalized URL string.

        Returns:
            A dictionary of structural character-count feature values.
        """
        return {
            "slash_count": url.count("/"),
            "dot_count": url.count("."),
            "colon_count": url.count(":"),
            "semicolon_count": url.count(";"),
            "underscore_count": url.count("_"),
            "hyphen_count": url.count("-"),
            "at_count": url.count("@"),
            "ampersand_count": url.count("&"),
            "equal_count": url.count("="),
            "question_count": url.count("?"),
            "hash_count": url.count("#"),
            "percent_count": url.count("%"),
        }

    def _extract_component_features(
        self, parsed: ParseResult
    ) -> dict[str, int | bool | str]:
        """Compute path-component-based features.

        Args:
            parsed: The parsed URL result.

        Returns:
            A dictionary of component-based feature values, including
            directory count, file extension, and filename length.
        """
        path = parsed.path
        segments = [segment for segment in path.split("/") if segment]

        directory_count = 0
        filename = ""

        if segments:
            if path.endswith("/"):
                directory_count = len(segments)
            else:
                directory_count = len(segments) - 1
                filename = segments[-1]

        extension_match = _FILE_EXTENSION_PATTERN.search(filename)
        file_extension = extension_match.group(1).lower() if extension_match else ""

        return {
            "directory_count": max(directory_count, 0),
            "file_extension_length": len(file_extension),
            "has_file_extension": bool(file_extension),
            "filename_length": len(filename),
        }

    def _extract_statistical_features(self, url: str) -> dict[str, int]:
        """Compute character-level statistical features.

        Args:
            url: The normalized URL string.

        Returns:
            A dictionary of character statistic feature values.
        """
        uppercase_count = sum(1 for char in url if char.isupper())
        lowercase_count = sum(1 for char in url if char.islower())
        digit_count = sum(1 for char in url if char.isdigit())
        alphabet_count = sum(1 for char in url if char.isalpha())
        special_character_count = (
            len(url) - digit_count - alphabet_count
        )
        unique_character_count = len(set(url))

        return {
            "uppercase_count": uppercase_count,
            "lowercase_count": lowercase_count,
            "digit_count": digit_count,
            "alphabet_count": alphabet_count,
            "special_character_count": special_character_count,
            "unique_character_count": unique_character_count,
        }

    def _extract_ratio_features(
        self, url: str, features: dict[str, int | float | bool | str]
    ) -> dict[str, float]:
        """Compute character-ratio-based features.

        Args:
            url: The normalized URL string.
            features: Previously computed features containing character
                counts required for ratio calculations.

        Returns:
            A dictionary of ratio-based feature values.
        """
        total_length = len(url)
        if total_length == 0:
            return {
                "digit_ratio": 0.0,
                "special_character_ratio": 0.0,
            }

        digit_count = int(features["digit_count"])
        special_character_count = int(features["special_character_count"])

        return {
            "digit_ratio": digit_count / total_length,
            "special_character_ratio": special_character_count / total_length,
        }

    def _extract_root_features(
        self, parsed: ParseResult
    ) -> dict[str, bool]:
        """Compute generic features for clean homepage-style URLs."""
        is_root_path = parsed.path in {"", "/"}

        return {
            "is_root_path": is_root_path,
            "is_clean_root_url": (
                is_root_path
                and not parsed.query
                and not parsed.fragment
                and parsed.scheme.lower() == "https"
            ),
        }