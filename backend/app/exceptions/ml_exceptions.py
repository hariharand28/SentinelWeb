
from __future__ import annotations

from typing import Any


class SentinelWebError(Exception):
    """Base exception for the SentinelWeb project."""

    default_message = "An unexpected SentinelWeb error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.context = context or {}
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message


class ConfigurationError(SentinelWebError):
    default_message = "Invalid project configuration."


class APIError(SentinelWebError):
    default_message = "API request failed."


class ModelLoadError(SentinelWebError):
    default_message = "Unable to load the machine learning model."

class ModelPersistenceError(SentinelWebError):
    """Raised when model artifacts cannot be loaded or saved."""

    default_message = "Model persistence operation failed."
    
class DatasetLoaderError(SentinelWebError):
    default_message = "Dataset loading failed."


class FeatureExtractionError(SentinelWebError):
    default_message = "Feature extraction failed."

class PreprocessingError(SentinelWebError):
    """Raised when preprocessing fails."""

    default_message = "Feature preprocessing failed."
    

class TrainingError(SentinelWebError):
    default_message = "Model training failed."


class PredictionError(SentinelWebError):
    default_message = "Prediction failed."


class ThreatIntelligenceError(SentinelWebError):
    default_message = "Threat intelligence lookup failed."


class ValidationError(SentinelWebError):
    default_message = "Validation failed."


class SchemaValidationError(ValidationError):
    default_message = "Dataset schema is invalid."


class InvalidURLError(ValidationError):
    default_message = "Invalid URL detected."


class EmptyDatasetError(ValidationError):
    default_message = "Dataset is empty after validation."
