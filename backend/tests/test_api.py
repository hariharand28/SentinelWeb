"""Tests for the FastAPI prediction API.

Tests the HTTP layer: /health endpoint, /predict with valid and invalid
input, response schema validation, and Gemini fallback behavior when
Gemini is unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.config import constants


# ---------------------------------------------------------------------------
# Skip predictor-dependent tests if model artifacts are missing
# ---------------------------------------------------------------------------

def _artifacts_present() -> bool:
    return (
        constants.MODEL_PATH.exists()
        and constants.SCALER_PATH.exists()
        and constants.LABEL_ENCODER_PATH.exists()
        and constants.FEATURE_COLUMNS_PATH.exists()
    )


# ---------------------------------------------------------------------------
# App client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app.main import app
    from app.ml.predictor import Predictor

    # Manually attach predictor to app state so TestClient works
    # without running the full lifespan.
    if _artifacts_present():
        app.state.predictor = Predictor()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /predict — valid URL
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _artifacts_present(),
    reason="Model artifacts not found in backend/models/. Run training first.",
)
class TestPredictValidURL:
    def test_predict_returns_200(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        assert response.status_code == 200

    def test_predict_response_has_prediction_field(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert "prediction" in data

    def test_predict_response_has_confidence_field(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert "confidence" in data

    def test_predict_response_has_explanation_field(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert "explanation" in data

    def test_predict_label_is_valid(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert data["prediction"] in ("legitimate", "phishing")

    def test_predict_confidence_in_range(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_explanation_is_non_empty_string(self, client):
        response = client.post(
            "/predict",
            json={"url": "https://www.google.com"},
        )
        data = response.json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"].strip()) > 0


# ---------------------------------------------------------------------------
# /predict — invalid input
# ---------------------------------------------------------------------------

class TestPredictInvalidInput:
    def test_missing_url_field_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_non_url_string_returns_422(self, client):
        response = client.post(
            "/predict",
            json={"url": "not-a-url-at-all"},
        )
        assert response.status_code == 422

    def test_empty_url_returns_422(self, client):
        response = client.post(
            "/predict",
            json={"url": ""},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Gemini fallback behavior
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _artifacts_present(),
    reason="Model artifacts not found in backend/models/. Run training first.",
)
class TestGeminiFallback:
    def test_prediction_succeeds_when_gemini_fails(self, client):
        """Even if Gemini throws an exception, /predict must return 200."""
        with patch(
            "app.api.routes.generate_explanation",
            side_effect=Exception("Gemini is unavailable"),
        ):
            response = client.post(
                "/predict",
                json={"url": "https://www.google.com"},
            )
        assert response.status_code == 200

    def test_fallback_explanation_is_returned_on_gemini_failure(self, client):
        """When Gemini fails, a non-empty fallback explanation must be returned."""
        with patch(
            "app.api.routes.generate_explanation",
            side_effect=Exception("Gemini is unavailable"),
        ):
            response = client.post(
                "/predict",
                json={"url": "https://www.google.com"},
            )
        data = response.json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"].strip()) > 0
