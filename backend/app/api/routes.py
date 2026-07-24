from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import (
    ModelPersistenceError,
    PredictionError,
)
from app.ml.predictor import Predictor

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict whether the submitted URL is phishing or legitimate.

    Args:
        request: Incoming prediction request.

    Returns:
        PredictionResponse containing the predicted label and confidence.

    Raises:
        HTTPException:
            400 - Invalid URL or prediction failure.
            500 - Model loading failure or unexpected server error.
    """
    try:
        predictor = Predictor()
        result = predictor.predict(str(request.url))

        logger.info(
            "Prediction completed for URL: %s",
            request.url,
        )

        return PredictionResponse(
            prediction=str(result["prediction"]),
            confidence=float(result["confidence"]),
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