"""Tests for URL feature extraction.

Validates that each extractor produces a consistent set of features
from known URLs and that the unified FeatureExtractor merges them
correctly without collisions or missing keys.
"""

import pytest

from app.ml.feature_extractor import FeatureExtractor
from app.ml.lexical_features import LexicalFeatureExtractor
from app.ml.domain_features import DomainFeatureExtractor
from app.ml.security_features import SecurityFeatureExtractor
from app.ml.url_features import URLFeatureExtractor
from app.exceptions.ml_exceptions import FeatureExtractionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGITIMATE_URL = "https://www.google.com/search?q=python"
PHISHING_URL = "http://secure-paypal-login.xyz/verify/account?id=1234567890"
SHORT_URL = "https://bit.ly/abc"


# ---------------------------------------------------------------------------
# LexicalFeatureExtractor
# ---------------------------------------------------------------------------

class TestLexicalFeatureExtractor:
    def setup_method(self):
        self.extractor = LexicalFeatureExtractor()

    def test_extracts_dict_for_legitimate_url(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_url_length_is_correct(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert features["url_length"] == len(LEGITIMATE_URL)

    def test_dot_count_is_correct(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert features["dot_count"] == LEGITIMATE_URL.count(".")

    def test_shannon_entropy_is_positive(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert features["shannon_entropy"] > 0.0

    def test_has_at_symbol_false_for_normal_url(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert features["has_at_symbol"] is False

    def test_empty_url_raises(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("")

    def test_all_feature_values_are_numeric_or_bool(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        for key, value in features.items():
            assert isinstance(value, (int, float, bool)), (
                f"Feature '{key}' has unexpected type {type(value)}"
            )


# ---------------------------------------------------------------------------
# DomainFeatureExtractor
# ---------------------------------------------------------------------------

class TestDomainFeatureExtractor:
    def setup_method(self):
        self.extractor = DomainFeatureExtractor()

    def test_extracts_dict_for_legitimate_url(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_empty_url_raises(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("")


# ---------------------------------------------------------------------------
# SecurityFeatureExtractor
# ---------------------------------------------------------------------------

class TestSecurityFeatureExtractor:
    def setup_method(self):
        self.extractor = SecurityFeatureExtractor()

    def test_extracts_dict_for_phishing_url(self):
        features = self.extractor.extract(PHISHING_URL)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_empty_url_raises(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("")


# ---------------------------------------------------------------------------
# URLFeatureExtractor
# ---------------------------------------------------------------------------

class TestURLFeatureExtractor:
    def setup_method(self):
        self.extractor = URLFeatureExtractor()

    def test_extracts_dict_for_legitimate_url(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_empty_url_raises(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("")


# ---------------------------------------------------------------------------
# Unified FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_returns_dict(self):
        features = self.extractor.extract(LEGITIMATE_URL)
        assert isinstance(features, dict)

    def test_feature_count_is_substantial(self):
        """The merged feature set must have at least 30 features."""
        features = self.extractor.extract(LEGITIMATE_URL)
        assert len(features) >= 30, (
            f"Expected at least 30 features, got {len(features)}"
        )

    def test_feature_keys_are_sorted(self):
        """Features must be returned in alphabetical order."""
        features = self.extractor.extract(LEGITIMATE_URL)
        keys = list(features.keys())
        assert keys == sorted(keys), "Feature keys are not alphabetically sorted"

    def test_no_duplicate_keys_across_extractors(self):
        """Merged feature dict must have no duplicate keys."""
        features = self.extractor.extract(LEGITIMATE_URL)
        assert len(features) == len(set(features.keys()))

    def test_all_values_are_numeric_or_bool(self):
        """All extracted feature values must be numeric or boolean."""
        features = self.extractor.extract(LEGITIMATE_URL)
        for key, value in features.items():
            assert isinstance(value, (int, float, bool)), (
                f"Feature '{key}' has unexpected type: {type(value)}"
            )

    def test_consistent_feature_count_across_urls(self):
        """Different URLs must produce the same number of features."""
        features_legit = self.extractor.extract(LEGITIMATE_URL)
        features_phish = self.extractor.extract(PHISHING_URL)
        assert len(features_legit) == len(features_phish), (
            "Feature count is inconsistent between URLs"
        )

    def test_empty_url_raises_feature_extraction_error(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("")

    def test_whitespace_only_url_raises(self):
        with pytest.raises(FeatureExtractionError):
            self.extractor.extract("   ")

    def test_phishing_url_features_differ_from_legitimate(self):
        """A phishing URL and a legitimate URL must produce different features."""
        features_legit = self.extractor.extract(LEGITIMATE_URL)
        features_phish = self.extractor.extract(PHISHING_URL)
        # At least some features should differ
        differing = [k for k in features_legit if features_legit[k] != features_phish.get(k)]
        assert len(differing) > 0, "Phishing and legitimate URLs produced identical features"
