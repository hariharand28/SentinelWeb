
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from datasets import load_dataset

from app.config import constants
from app.config.constants import DEFAULT_RANDOM_STATE, REQUIRED_COLUMNS
from app.core.logger import get_logger
from app.exceptions.ml_exceptions import (
    DatasetLoaderError,
    EmptyDatasetError,
    SchemaValidationError,
)

logger = get_logger(__name__)


class DatasetLoader:
    """Load and validate phishing datasets."""

    @staticmethod
    def _validate_url(url: Any) -> bool:
        if pd.isna(url):
            return False

        value = str(url).strip()

        if not value:
            return False

        try:
            parsed = urlparse(
                value if "://" in value else f"http://{value}"
            )
            return bool(parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _normalize_label(value: Any) -> str | None:
        if pd.isna(value):
            return None

        normalized = str(value).strip().lower()
        if not normalized:
            return None

        if normalized in {"legitimate", "benign", "0"}:
            return "legitimate"
        if normalized in {"phishing", "1"}:
            return "phishing"

        mapped = constants.LABEL_MAPPING.get(normalized)
        if mapped == 0:
            return "legitimate"
        if mapped == 1:
            return "phishing"

        return None
    

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        try:
            logger.info("Loading CSV: %s", path)
            return pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as exc:
            raise DatasetLoaderError(str(exc)) from exc

    def load_parquet(self, path: str | Path) -> pd.DataFrame:
        try:
            logger.info("Loading Parquet: %s", path)
            return pd.read_parquet(path, dtype_backend="pyarrow")
        except Exception as exc:
            raise DatasetLoaderError(str(exc)) from exc

    def load_directory(self, directory: str | Path) -> pd.DataFrame:
        directory = Path(directory)
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise DatasetLoaderError("No parquet files found.")
        frames = []
        total = len(files)
        for index, file in enumerate(files, start=1):
            logger.info("Loading %d/%d : %s", index, total, file.name)
            frames.append(self.load_parquet(file))
        return pd.concat(frames, ignore_index=True)

    def load_huggingface(self, dataset_name: str, split: str = "train") -> pd.DataFrame:
        try:
            logger.info("Loading HuggingFace dataset %s", dataset_name)
            ds = load_dataset(dataset_name, split=split)
            return ds.to_pandas().convert_dtypes(dtype_backend="pyarrow")
        except Exception as exc:
            raise DatasetLoaderError(str(exc)) from exc

    def sample_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: int,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> pd.DataFrame:
        if len(df) <= sample_size:
            return df.copy()
        return df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)

    def load(self) -> pd.DataFrame:
        return self.load_csv(constants.DATASET_PATH)
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise EmptyDatasetError()

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SchemaValidationError(
                f"Missing required columns: {sorted(missing)}"
            )

        logger.info("Initial rows: %d", len(df))

        clean = df.copy()

        clean = clean.dropna(subset=["url", "type"])

        clean["url"] = clean["url"].astype(str).str.strip()

        clean["label"] = clean["type"].apply(self._normalize_label)

        clean = clean.dropna(subset=["label"])
        clean = clean[clean["url"] != ""]

        before = len(clean)

        conflicts = (
            clean.groupby("url")["label"]
            .nunique()
        )

        conflicting_urls = conflicts[conflicts > 1].index

        if len(conflicting_urls):
            logger.warning(
                "Removing %d URLs with conflicting labels.",
                len(conflicting_urls),
            )
            clean = clean[~clean["url"].isin(conflicting_urls)]

        clean = clean.drop_duplicates(subset=["url"])

        logger.info(
            "Duplicates removed: %d",
            before - len(clean),
        )

        invalid = ~clean["url"].apply(self._validate_url)

        if invalid.any():
            logger.warning(
                "Dropping %d invalid URLs.",
                int(invalid.sum()),
            )
            clean = clean[~invalid]

        if clean.empty:
            raise EmptyDatasetError()

        logger.info("Validation completed")
        logger.info("Rows: %d", len(clean))
        logger.info("Columns: %d", len(clean.columns))
        logger.info("Memory: %.2f MB",
                    clean.memory_usage(deep=True).sum() / 1024 / 1024)

        return clean.reset_index(drop=True)
