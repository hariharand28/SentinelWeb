"""Unified feature extraction orchestration for phishing detection.

This module defines the FeatureExtractor class, which coordinates the
lexical, domain, security, and URL structural feature extractors to
produce a single, merged, and alphabetically sorted feature dictionary
for use by SentinelWeb's AI-based phishing detection pipeline.
"""

from __future__ import annotations

from app.core.logger import get_logger
from app.exceptions.ml_exceptions import FeatureExtractionError
from app.ml.domain_features import DomainFeatureExtractor
from app.ml.lexical_features import LexicalFeatureExtractor
from app.ml.security_features import SecurityFeatureExtractor
from app.ml.url_features import URLFeatureExtractor

logger = get_logger(__name__)


class FeatureExtractor:
    """Orchestrates all URL feature extractors into a unified feature set.

    This class composes the lexical, domain, security, and URL structural
    feature extractors, merges their outputs into a single dictionary,
    validates that no feature name collisions occur, and returns the
    combined feature set sorted alphabetically by key.
    """

    def __init__(self) -> None:
        """Initialize the FeatureExtractor with its component extractors."""
        self._lexical_extractor = LexicalFeatureExtractor()
        self._domain_extractor = DomainFeatureExtractor()
        self._security_extractor = SecurityFeatureExtractor()
        self._url_extractor = URLFeatureExtractor()

    def extract(self, url: str) -> dict[str, int | float | bool | str]:
        """Extract and merge all features for a given URL.

        Args:
            url: The URL string to analyze.

        Returns:
            An alphabetically sorted dictionary mapping feature names to
            their computed integer, float, boolean, or string values.

        Raises:
            FeatureExtractionError: If the URL is invalid, any underlying
                extractor fails, or duplicate feature names are detected
                across extractors.
        """
        self._validate_url(url)

        try:
            lexical_features = self._lexical_extractor.extract(url)
            domain_features = self._domain_extractor.extract(url)
            security_features = self._security_extractor.extract(url)
            url_features = self._url_extractor.extract(url)

            feature_sets = [
                lexical_features,
                domain_features,
                security_features,
                url_features,
            ]

            merged_features = self._merge_features(feature_sets)
            sorted_features = dict(sorted(merged_features.items()))

            logger.info(
                "Successfully extracted %d features for URL: %s",
                len(sorted_features),
                url,
            )

            return sorted_features

        except FeatureExtractionError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to extract combined features for URL: %s", url
            )
            raise FeatureExtractionError(
                f"Failed to extract features from URL: {url!r}"
            )

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

    def _merge_features(
        self,
        feature_sets: list[dict[str, int | float | bool | str]],
    ) -> dict[str, int | float | bool | str]:
        """Merge multiple feature dictionaries into a single dictionary.

        Args:
            feature_sets: A list of feature dictionaries produced by the
                individual extractors.

        Returns:
            A single dictionary containing all key-value pairs from every
            feature dictionary, with no data loss.
        """
        merged: dict[str, int | float | bool | str] = {}
        for feature_set in feature_sets:
            for key, value in feature_set.items():
                if key in merged:
                    logger.debug(
                        "Skipping duplicate feature '%s' from later extractor.",
                        key,
                    )
                    continue
                merged[key] = value
        return merged