"""Feature preprocessing pipeline for phishing detection model training.

This module defines the Preprocessor class, which handles scaling of
numerical feature columns, encoding and decoding of classification
labels, and persistence of the fitted transformers used throughout
SentinelWeb's machine learning pipeline.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import constants
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import ModelPersistenceError, PreprocessingError

logger = get_logger(__name__)


class Preprocessor:
    """Preprocesses feature data for phishing detection model training.

    This class removes non-feature columns, scales numerical feature
    columns using a StandardScaler, and encodes or decodes classification
    labels using a LabelEncoder. It also supports persisting and loading
    the fitted transformers to and from disk.
    """

    def __init__(self) -> None:
        """Initialize the Preprocessor with fresh scaler and encoder."""
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the scaler on the given DataFrame and transform it.

        Args:
            df: The input DataFrame containing feature columns, and
                optionally non-feature columns to be excluded.

        Returns:
            A NumPy array of scaled numerical feature values.

        Raises:
            PreprocessingError: If the DataFrame is invalid, contains no
                usable feature columns, or fitting/transformation fails.
        """
        self._validate_dataframe(df)
        feature_df = self._remove_non_feature_columns(df)
        self._validate_feature_columns(feature_df)

        try:
            scaled_values = self._scaler.fit_transform(feature_df.to_numpy())
            self._is_fitted = True

            logger.info(
                "Fitted and transformed %d rows across %d feature columns.",
                feature_df.shape[0],
                feature_df.shape[1],
            )

            return scaled_values

        except Exception as exc:
            logger.exception("Failed to fit_transform DataFrame.")
            raise PreprocessingError(
                "Failed to fit and transform the provided DataFrame."
            ) from exc

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame using the previously fitted scaler.

        Args:
            df: The input DataFrame containing feature columns, and
                optionally non-feature columns to be excluded.

        Returns:
            A NumPy array of scaled numerical feature values.

        Raises:
            PreprocessingError: If the DataFrame is invalid, the scaler
                has not been fitted, or transformation fails.
        """
        self._validate_dataframe(df)

        if not self._is_fitted:
            logger.error("Transform called before scaler was fitted.")
            raise PreprocessingError(
                "Scaler must be fitted or loaded before calling transform."
            )

        feature_df = self._remove_non_feature_columns(df)
        self._validate_feature_columns(feature_df)

        try:
            scaled_values = self._scaler.transform(feature_df.to_numpy())

            logger.info(
                "Transformed %d rows across %d feature columns.",
                feature_df.shape[0],
                feature_df.shape[1],
            )

            return scaled_values

        except Exception as exc:
            logger.exception("Failed to transform DataFrame.")
            raise PreprocessingError(
                "Failed to transform the provided DataFrame."
            ) from exc

    def encode_labels(self, labels: Any) -> np.ndarray:
        """Encode categorical labels into numerical values.

        Args:
            labels: An iterable of categorical label values to encode.

        Returns:
            A NumPy array of encoded integer label values.

        Raises:
            PreprocessingError: If the labels are invalid or encoding
                fails.
        """
        self._validate_labels(labels)

        try:
            encoded = self._label_encoder.fit_transform(labels)

            logger.info("Encoded %d labels.", len(encoded))

            return encoded

        except Exception as exc:
            logger.exception("Failed to encode labels.")
            raise PreprocessingError(
                "Failed to encode the provided labels."
            ) from exc

    def decode_labels(self, labels: Any) -> np.ndarray:
        """Decode numerical labels back into their original categories.

        Args:
            labels: An iterable of encoded integer label values.

        Returns:
            A NumPy array of decoded original label values.

        Raises:
            PreprocessingError: If the labels are invalid or decoding
                fails.
        """
        self._validate_labels(labels)

        try:
            decoded = self._label_encoder.inverse_transform(labels)

            logger.info("Decoded %d labels.", len(decoded))

            return decoded

        except Exception as exc:
            logger.exception("Failed to decode labels.")
            raise PreprocessingError(
                "Failed to decode the provided labels."
            ) from exc

    def save(self) -> None:
        """Persist the fitted scaler and label encoder to disk.

        Raises:
            PreprocessingError: If the transformers have not been fitted.
            ModelPersistenceError: If persistence to disk fails.
        """
        if not self._is_fitted:
            logger.error("Save called before scaler was fitted.")
            raise PreprocessingError(
                "Scaler must be fitted before it can be saved."
            )

        try:
            constants.SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
            constants.LABEL_ENCODER_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )

            joblib.dump(self._scaler, constants.SCALER_PATH)
            joblib.dump(self._label_encoder, constants.LABEL_ENCODER_PATH)

            logger.info(
                "Persisted %s to %s and %s to %s.",
                constants.SCALER_NAME,
                constants.SCALER_PATH,
                constants.LABEL_ENCODER_NAME,
                constants.LABEL_ENCODER_PATH,
            )

        except Exception as exc:
            logger.exception("Failed to save scaler and encoder to disk.")
            raise ModelPersistenceError(
                "Failed to persist the scaler and label encoder."
            ) from exc

    def load(self) -> None:
        """Load a previously persisted scaler and label encoder from disk.

        Raises:
            ModelPersistenceError: If the persisted files do not exist or
                loading fails.
        """
        if (
            not constants.SCALER_PATH.exists()
            or not constants.LABEL_ENCODER_PATH.exists()
        ):
            logger.error(
                "Scaler or label encoder file does not exist on disk: "
                "%s, %s",
                constants.SCALER_PATH,
                constants.LABEL_ENCODER_PATH,
            )
            raise ModelPersistenceError(
                "Scaler or label encoder file does not exist on disk."
            )

        try:
            self._scaler = joblib.load(constants.SCALER_PATH)
            self._label_encoder = joblib.load(constants.LABEL_ENCODER_PATH)
            self._is_fitted = True

            logger.info(
                "Loaded %s from %s and %s from %s.",
                constants.SCALER_NAME,
                constants.SCALER_PATH,
                constants.LABEL_ENCODER_NAME,
                constants.LABEL_ENCODER_PATH,
            )

        except Exception as exc:
            logger.exception("Failed to load scaler and encoder from disk.")
            raise ModelPersistenceError(
                "Failed to load the scaler and label encoder."
            ) from exc

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the input is a non-empty pandas DataFrame.

        Args:
            df: The DataFrame to validate.

        Raises:
            PreprocessingError: If the input is not a valid, non-empty
                DataFrame.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Invalid or empty DataFrame provided.")
            raise PreprocessingError(
                "Input must be a non-empty pandas DataFrame."
            )

    def _validate_feature_columns(self, feature_df: pd.DataFrame) -> None:
        """Validate that feature columns remain after column removal.

        Args:
            feature_df: The DataFrame after non-feature columns have
                been removed.

        Raises:
            PreprocessingError: If no feature columns remain.
        """
        if feature_df.shape[1] == 0:
            logger.error("No feature columns remain after column removal.")
            raise PreprocessingError(
                "No feature columns remain after removing non-feature "
                "columns."
            )

    def _validate_labels(self, labels: Any) -> None:
        """Validate that the labels input is non-empty and iterable.

        Args:
            labels: The label values to validate.

        Raises:
            PreprocessingError: If the labels are None, not iterable, or
                empty.
        """
        if labels is None:
            logger.error("Labels must not be None.")
            raise PreprocessingError("Labels must not be None.")

        try:
            length = len(labels)
        except TypeError as exc:
            logger.error("Labels must be an iterable sequence.")
            raise PreprocessingError(
                "Labels must be an iterable sequence."
            ) from exc 

        if length == 0:
            logger.error("Labels must not be empty.")
            raise PreprocessingError("Labels must not be empty.")

    def _remove_non_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove configured non-feature columns from the DataFrame.

        Args:
            df: The input DataFrame potentially containing non-feature
                columns such as identifiers or raw URL strings.

        Returns:
            A DataFrame containing only numerical feature columns.
        """
        columns_to_drop = [
            column
            for column in constants.NON_FEATURE_COLUMNS
            if column in df.columns
        ]
        feature_df = df.drop(columns=columns_to_drop, errors="ignore")
        feature_df = feature_df.select_dtypes(include=[np.number])
        return feature_df