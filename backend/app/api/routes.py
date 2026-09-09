from fastapi import APIRouter, HTTPException, Request

from app.services.gemini_service import generate_explanation
from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import (
    ModelPersistenceError,
    PredictionError,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
    """Run Random Forest prediction only.

    Gemini is intentionally NOT called here so that the ML verdict
    is returned immediately.
    """
    predictor = http_request.app.state.predictor

    try:
        result = predictor.predict(str(request.url))

        logger.info(
            "Prediction completed for URL: %s",
            request.url,
        )

        return PredictionResponse(
            prediction=str(result["prediction"]),
            confidence=float(result["confidence"]),
            explanation="",
        )

    except PredictionError as exc:
        logger.error(
            "Prediction failed for URL: %s",
            request.url,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ModelPersistenceError as exc:
        logger.exception(
            "Model artifacts could not be loaded."
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error during prediction."
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc


@router.post("/explain")
def explain(request: PredictionRequest, http_request: Request) -> dict:
    """Generate the Gemini explanation separately from the ML prediction."""

    predictor = http_request.app.state.predictor

    try:
        # Re-run the lightweight prediction pipeline so Gemini receives
        # the same extracted features used by the Random Forest model.
        result = predictor.predict(str(request.url))

        explanation = generate_explanation(
            url=str(request.url),
            prediction=str(result["prediction"]),
            confidence=float(result["confidence"]),
            features=result.get("features", {}),
        )

        logger.info(
            "Explanation generated for URL: %s",
            request.url,
        )

        return {
            "explanation": explanation,
        }

    except Exception as exc:
        logger.exception(
            "Explanation generation failed for URL: %s",
            request.url,
        )

        return {
            "explanation": (
                "AI explanation is temporarily unavailable. "
                "The phishing prediction was generated successfully "
                "by the Random Forest model."
            )
        }