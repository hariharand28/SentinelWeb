"""Training pipeline for the phishing detection machine learning model.

This module orchestrates the end-to-end training workflow: loading the
dataset, extracting URL-based features, preprocessing (label encoding and
feature scaling), splitting the data, training a RandomForestClassifier,
evaluating performance, and persisting all trained artifacts to disk.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.config import constants
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import (
    FeatureExtractionError,
    ModelPersistenceError,
    TrainingError,
)
from app.ml.dataset_loader import DatasetLoader
from app.ml.feature_extractor import FeatureExtractor
from app.ml.preprocessor import Preprocessor

logger = get_logger(__name__)


class Trainer:
    """Trains, evaluates, and persists the phishing detection model.

    This class encapsulates the full training pipeline, including dataset
    loading, feature extraction, preprocessing, model training, evaluation,
    and artifact persistence.

    Attributes:
        dataset_loader (DatasetLoader): Loads the raw dataset.
        feature_extractor (FeatureExtractor): Extracts features from URLs.
        preprocessor (Preprocessor): Encodes labels and scales features.
        model (RandomForestClassifier): The classifier being trained.
        random_state (int): Random seed for reproducibility.
        test_size (float): Fraction of data reserved for testing.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        n_estimators: int = 200,
    ) -> None:
        """Initializes the Trainer with its dependent components.

        Args:
            random_state (int): Seed used for reproducibility across the
                train/test split and the classifier. Defaults to 42.
            test_size (float): Proportion of the dataset to include in the
                test split. Defaults to 0.2.
            n_estimators (int): Number of trees in the RandomForestClassifier.
                Defaults to 200.
        """
        self.dataset_loader = DatasetLoader()
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = Preprocessor()
        self.random_state = random_state
        self.test_size = test_size
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
    def _load_dataset(self) -> pd.DataFrame:
        """Load the local CSV dataset."""

        try:
            logger.info("Loading local dataset...")

            dataset = self.dataset_loader.load()

            dataset = self.dataset_loader.validate_dataframe(dataset)

            logger.info("Loaded %d records.", len(dataset))

            return dataset

        except Exception as exc:
            logger.exception("Failed to load dataset.")
            raise TrainingError(
                f"Dataset loading failed: {exc}"
            ) from exc
        

    def _extract_features(
        self, urls: List[str]
    ) -> pd.DataFrame:
        """Extracts features from a list of URLs using the FeatureExtractor.

        Args:
            urls (List[str]): List of URLs to extract features from.

        Returns:
            pd.DataFrame: A DataFrame of extracted features, one row per URL.

        Raises:
            FeatureExtractionError: If feature extraction fails for the
                dataset.
        """
        try:
            logger.info("Extracting features from %d URLs...", len(urls))
            feature_rows: List[Dict[str, Any]] = []
            for url in urls:
                features = self.feature_extractor.extract(url)
                feature_rows.append(features)
            features_df = pd.DataFrame(feature_rows)
            logger.info(
                "Feature extraction completed. Feature matrix shape: %s",
                features_df.shape,
            )
            return features_df
        except Exception as exc:
            logger.exception("Feature extraction failed.")
            raise FeatureExtractionError(f"Feature extraction failed: {exc}") from exc

    def _preprocess(
        self, features_df: pd.DataFrame, labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes labels and scales features using the Preprocessor.

        Args:
            features_df (pd.DataFrame): Extracted feature matrix.
            labels (pd.Series): Raw target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The scaled feature matrix and
                encoded labels.

        Raises:
            TrainingError: If preprocessing fails.
        """
        try:
            logger.info("Encoding labels and scaling features...")
            encoded_labels = self.preprocessor.encode_labels(labels)
            scaled_features = self.preprocessor.fit_transform(features_df)
            print("Training features:", features_df.columns.tolist())
            logger.info("Preprocessing completed successfully.")
            return scaled_features, encoded_labels
        except Exception as exc:
            logger.exception("Preprocessing failed.")
            raise TrainingError(f"Preprocessing failed: {exc}") from exc

    def _split_data(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into training and testing sets.

        Args:
            features (np.ndarray): Scaled feature matrix.
            labels (np.ndarray): Encoded labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_train, X_test, y_train, y_test.

        Raises:
            TrainingError: If splitting the dataset fails.
        """
        try:
            logger.info(
                "Splitting dataset into train/test sets with test_size=%s...",
                self.test_size,
            )
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=labels,
            )
            logger.info(
                "Split completed. Train size: %d, Test size: %d",
                len(x_train),
                len(x_test),
            )
            return x_train, x_test, y_train, y_test
        except Exception as exc:
            logger.exception("Dataset splitting failed.")
            raise TrainingError(f"Dataset splitting failed: {exc}") from exc

    def _fit_model(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the RandomForestClassifier on the training data.

        Args:
            x_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.

        Raises:
            TrainingError: If model fitting fails.
        """
        try:
            logger.info("Training RandomForestClassifier...")
            self.model.fit(x_train, y_train)
            logger.info("Model training completed successfully.")
        except Exception as exc:
            logger.exception("Model training failed.")
            raise TrainingError(f"Model training failed: {exc}") from exc

    def _evaluate_model(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates the trained model on the test set.

        Args:
            x_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): Test labels.

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision,
                recall, and f1-score.

        Raises:
            TrainingError: If evaluation fails.
        """
        try:
            logger.info("Evaluating model on test set...")
            predictions = self.model.predict(x_test)

            metrics: Dict[str, float] = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }

            logger.info(
                "Evaluation results - Accuracy: %.4f, Precision: %.4f, "
                "Recall: %.4f, F1-score: %.4f",
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            )
            return metrics
        except Exception as exc:
            logger.exception("Model evaluation failed.")
            raise TrainingError(f"Model evaluation failed: {exc}") from exc

    def _save_model(self) -> None:
        """Saves the trained model to disk using constants.MODEL_PATH.

        Creates parent directories automatically if they do not exist.

        Raises:
            ModelPersistenceError: If saving the model fails.
        """
        try:
            model_path = Path(constants.MODEL_PATH)
            model_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Saving trained model to %s...", model_path)
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully.")
        except Exception as exc:
            logger.exception("Failed to save model.")
            raise ModelPersistenceError(f"Failed to save model: {exc}") from exc

    def _save_preprocessor(self) -> None:
        """Saves the preprocessor's scaler and encoder to disk.

        Raises:
            ModelPersistenceError: If saving the preprocessor fails.
        """
        try:
            logger.info("Saving preprocessor artifacts (scaler and encoder)...")
            self.preprocessor.save()
            logger.info("Preprocessor artifacts saved successfully.")
        except Exception as exc:
            logger.exception("Failed to save preprocessor artifacts.")
            raise ModelPersistenceError(
                f"Failed to save preprocessor artifacts: {exc}"
            ) from exc

    def train(self) -> Dict[str, float]:
        """Runs the full training pipeline end-to-end.

        This method loads the dataset, extracts features, preprocesses
        the data, splits it into train/test sets, trains the model,
        evaluates it, and persists all resulting artifacts.

        Returns:
            Dict[str, float]: Evaluation metrics produced during training.

        Raises:
            TrainingError: If any step of the training pipeline fails.
            FeatureExtractionError: If feature extraction fails.
            ModelPersistenceError: If saving artifacts fails.
        """
        logger.info("Starting training pipeline...")

        dataset = self._load_dataset()

        if constants.URL_COLUMN not in dataset.columns:
            raise TrainingError(
                f"Missing expected URL column: {constants.URL_COLUMN}"
            )
        if constants.LABEL_COLUMN not in dataset.columns:
            raise TrainingError(
                f"Missing expected label column: {constants.LABEL_COLUMN}"
            )

        urls = dataset[constants.URL_COLUMN].tolist()
        labels = dataset[constants.LABEL_COLUMN]

        features_df = self._extract_features(urls)
        print(features_df.columns.tolist())
        print(features_df.shape)
        scaled_features, encoded_labels = self._preprocess(features_df, labels)
        x_train, x_test, y_train, y_test = self._split_data(
            scaled_features, encoded_labels
        )

        self._fit_model(x_train, y_train)
        metrics = self._evaluate_model(x_test, y_test)

        self._save_model()
        self._save_preprocessor()

        logger.info("Training pipeline completed successfully.")
        return metrics


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()