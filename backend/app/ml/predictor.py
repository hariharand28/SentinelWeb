import joblib
import pandas as pd

from app.config import constants
from app.core.logger import get_logger

from app.exceptions.ml_exceptions import (
    FeatureExtractionError,
    ModelPersistenceError,
    PredictionError,
)

from app.ml.feature_extractor import FeatureExtractor
from app.ml.preprocessor import Preprocessor

from sklearn.ensemble import RandomForestClassifier

logger = get_logger(__name__)


class Predictor:
    """Performs phishing detection inference on a single URL.

    Loads the trained RandomForest model along with the fitted scaler
    and label encoder, then exposes a prediction API that extracts
    features from a URL, preprocesses them, and returns the predicted
    label with its associated confidence score.

    Attributes:
        feature_extractor: Component responsible for extracting URL features.
        preprocessor: Component responsible for scaling features and
            decoding predicted labels.
        model: The trained RandomForestClassifier used for inference.
    """

    def __init__(self) -> None:
        """Initializes the Predictor and loads all required artifacts.

        Raises:
            ModelPersistenceError: If the trained model, scaler, or label
                encoder cannot be loaded.
        """
        self.feature_extractor: FeatureExtractor = FeatureExtractor()
        self.preprocessor: Preprocessor = Preprocessor()
        self.feature_columns: list[str] = []

        try:
            if not constants.MODEL_PATH.exists():
                logger.error(
                    "Trained model file does not exist: %s",
                    constants.MODEL_PATH,
                )
                raise ModelPersistenceError(
                    "Trained model file does not exist."
                )

            self.model = joblib.load(constants.MODEL_PATH)

            self.preprocessor.load()
            self.feature_columns = self.preprocessor.feature_columns
            logger.info("Predictor artifacts loaded successfully")
        except Exception as exc:
            logger.exception("Failed to load trained model artifacts")
            raise ModelPersistenceError(
                "Failed to load trained model artifacts"
            ) from exc

    def predict(self, url: str) -> dict[str, object]:
        """Predicts whether a given URL is phishing or legitimate.

        Args:
            url: The URL to classify.

        Returns:
            A dictionary containing the decoded prediction label and the
            model's confidence score for that prediction.

        Raises:
            PredictionError: If URL validation or prediction fails.
            FeatureExtractionError: If feature extraction fails.
        """
        self._validate_url(url)

        try:
            features = self.feature_extractor.extract(url)
        except Exception as exc:
            logger.exception("Failed to extract features for URL: %s", url)
            raise FeatureExtractionError(
                f"Failed to extract features for URL: {url}"
            ) from exc

        feature_dataframe = self._build_dataframe(features)

        try:
            scaled_features = self.preprocessor.transform(feature_dataframe)
            prediction = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)
            decoded_label = str(
                self.preprocessor.decode_labels(prediction)[0]
            )            
            confidence = round(
                float(probabilities[0].max()),
                4,
            )
        except Exception as exc:
            logger.exception("Failed to generate prediction for URL: %s", url)
            raise PredictionError(
                f"Failed to generate prediction for URL: {url}"
            ) from exc

        logger.info(
            "Prediction completed for URL: %s -> %s", url, decoded_label
        )

        return {
            "prediction": decoded_label,
            "confidence": confidence,
            "features": features,

        }

    def _validate_url(self, url: str) -> None:
        """Validates that the provided URL is a non-empty string.

        Args:
            url: The URL to validate.

        Raises:
            PredictionError: If the URL is not a non-empty string.
        """
        if not isinstance(url, str) or not url.strip():
            logger.error("Invalid URL provided for prediction: %r", url)
            raise PredictionError(f"Invalid URL provided for prediction: {url!r}")

    def _build_dataframe(self, features: dict[str, float]) -> pd.DataFrame:
        """Converts extracted URL features into a single-row DataFrame.

        Args:
            features: A dictionary of extracted feature values for a URL.

        Returns:
            A single-row DataFrame containing the extracted features.

        Raises:
            FeatureExtractionError: If the features cannot be converted
                into a DataFrame.
        """
        try:
            return pd.DataFrame([features])
        except Exception as exc:
            logger.exception("Failed to build feature DataFrame for prediction")
            raise FeatureExtractionError(
                "Failed to build feature DataFrame for prediction"
            ) from exc