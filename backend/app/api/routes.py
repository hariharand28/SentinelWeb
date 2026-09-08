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
    """Predict whether the submitted URL is phishing or legitimate.

    The Predictor instance is loaded once at application startup and
    accessed here via ``http_request.app.state.predictor``, ensuring
    the heavy 200 MB model file is not reloaded per-request.

    Args:
        request: Incoming prediction request containing the URL.
        http_request: The raw FastAPI request used to access app state.

    Returns:
        PredictionResponse containing the predicted label, confidence,
        and a Gemini-generated explanation.

    Raises:
        HTTPException:
            400 - Invalid URL or prediction failure.
            500 - Model loading failure or unexpected server error.
    """
    predictor = http_request.app.state.predictor

    try:
        result = predictor.predict(str(request.url))

        logger.info(
            "Prediction completed for URL: %s",
            request.url,
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

    try:
        explanation = generate_explanation(
            url=str(request.url),
            prediction=result["prediction"],
            confidence=result["confidence"],
            features=result.get("features", {}),
        )
    except Exception:
        logger.exception(
            "Gemini explanation generation failed for URL: %s",
            request.url,
        )
        explanation = (
            "The website was successfully analyzed by the SentinelWeb "
            "Random Forest model. An AI-generated explanation is "
            "temporarily unavailable. Please try again later."
        )

    return PredictionResponse(
        prediction=str(result["prediction"]),
        confidence=float(result["confidence"]),
        explanation=explanation,
    )