"""Tests for the ML Predictor.

These tests exercise the full prediction pipeline:
feature extraction → preprocessing → Random Forest prediction.

NOTE: These tests require the trained model artifacts to be present in
backend/models/ (random_forest.pkl, scaler.pkl, label_encoder.pkl,
feature_columns.pkl). If they are missing, tests will be skipped.
"""

import pytest

from app.exceptions.ml_exceptions import ModelPersistenceError, PredictionError
from app.config import constants


# ---------------------------------------------------------------------------
# Skip entire module if model artifacts are missing
# ---------------------------------------------------------------------------

def _artifacts_present() -> bool:
    return (
        constants.MODEL_PATH.exists()
        and constants.SCALER_PATH.exists()
        and constants.LABEL_ENCODER_PATH.exists()
        and constants.FEATURE_COLUMNS_PATH.exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_present(),
    reason="Model artifacts not found in backend/models/. Run training first.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    from app.ml.predictor import Predictor
    return Predictor()


VALID_URLS = [
    "https://www.google.com",
    "https://www.github.com",
    "http://secure-paypal-login.xyz/verify/account?id=1234567890",
    "https://amazon.com/gp/product/B08N5WRWNW",
]


# ---------------------------------------------------------------------------
# Predictor initialization
# ---------------------------------------------------------------------------

class TestPredictorInit:
    def test_predictor_loads_without_error(self, predictor):
        assert predictor is not None

    def test_predictor_has_model(self, predictor):
        assert predictor.model is not None

    def test_predictor_has_feature_columns(self, predictor):
        assert len(predictor.feature_columns) > 0

    def test_feature_columns_are_strings(self, predictor):
        for col in predictor.feature_columns:
            assert isinstance(col, str)


# ---------------------------------------------------------------------------
# Prediction output structure
# ---------------------------------------------------------------------------

class TestPredictionOutput:
    @pytest.mark.parametrize("url", VALID_URLS)
    def test_predict_returns_dict(self, predictor, url):
        result = predictor.predict(url)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("url", VALID_URLS)
    def test_predict_has_required_keys(self, predictor, url):
        result = predictor.predict(url)
        assert "prediction" in result
        assert "confidence" in result
        assert "features" in result

    @pytest.mark.parametrize("url", VALID_URLS)
    def test_prediction_is_valid_label(self, predictor, url):
        result = predictor.predict(url)
        assert result["prediction"] in ("legitimate", "phishing"), (
            f"Unexpected prediction label: {result['prediction']!r}"
        )

    @pytest.mark.parametrize("url", VALID_URLS)
    def test_confidence_is_valid_probability(self, predictor, url):
        result = predictor.predict(url)
        confidence = result["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence out of range [0,1]: {confidence}"
        )

    @pytest.mark.parametrize("url", VALID_URLS)
    def test_features_is_dict_of_numerics(self, predictor, url):
        result = predictor.predict(url)
        features = result["features"]
        assert isinstance(features, dict)
        assert len(features) > 0
        for key, value in features.items():
            assert isinstance(value, (int, float, bool)), (
                f"Feature '{key}' has unexpected type: {type(value)}"
            )


# ---------------------------------------------------------------------------
# Invalid URL handling
# ---------------------------------------------------------------------------

class TestInvalidURLHandling:
    def test_empty_string_raises_prediction_error(self, predictor):
        with pytest.raises(PredictionError):
            predictor.predict("")

    def test_whitespace_string_raises_prediction_error(self, predictor):
        with pytest.raises(PredictionError):
            predictor.predict("   ")

    def test_none_raises_prediction_error(self, predictor):
        with pytest.raises(PredictionError):
            predictor.predict(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestPredictionDeterminism:
    def test_same_url_produces_same_prediction(self, predictor):
        """Random Forest prediction must be deterministic."""
        url = "https://www.google.com"
        result_1 = predictor.predict(url)
        result_2 = predictor.predict(url)
        assert result_1["prediction"] == result_2["prediction"]
        assert result_1["confidence"] == result_2["confidence"]
