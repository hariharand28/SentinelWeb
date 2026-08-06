from __future__ import annotations

import re
from pathlib import Path
from typing import Final

# ==============================================================================
# Project Directories
# ==============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DATASETS_DIR: Final[Path] = PROJECT_ROOT / "datasets"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
CACHE_DIR: Final[Path] = PROJECT_ROOT / ".cache"

# ==============================================================================
# Dataset Configuration
# ==============================================================================

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({
    "url",
    "type",
})

OPTIONAL_COLUMNS: Final[frozenset[str]] = frozenset({
    "target",
    "sha256",
})

SUPPORTED_LABELS: Final[frozenset[str]] = frozenset({
    "legitimate",
    "phishing",
    "benign",
})

LABEL_MAPPING = {
    "legitimate": 0,
    "phishing": 1,

    "benign": 0,
    "0": 0,
    0: 0,

    "1": 1,
    1: 1,
}

# ==============================================================================
# Dataset Splits
# ==============================================================================
# Dataset
DATASET_PATH = DATASETS_DIR / "URL dataset.csv"
URL_COLUMN = "url"
LABEL_COLUMN = "label"

TRAIN_SPLIT: Final[str] = "train"
TEST_SPLIT: Final[str] = "test"
VALIDATION_SPLIT: Final[str] = "validation"

DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_SAMPLE_SIZE: Final[int] = 100000
TEST_SIZE: Final[float] = 0.20

# ==============================================================================
# URL Validation
# ==============================================================================

MIN_URL_LENGTH: Final[int] = 4
MAX_URL_LENGTH: Final[int] = 2048
DEFAULT_TIMEOUT: Final[int] = 10

# ==============================================================================
# Domain Features
# ==============================================================================

URL_SHORTENERS: Final[frozenset[str]] = frozenset({
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "is.gd",
    "ow.ly",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "adf.ly",
})

SAFE_TLDS: Final[frozenset[str]] = frozenset({
    "gov",
    "edu",
    "mil",
    "int",
})

RISKY_TLDS: Final[frozenset[str]] = frozenset({
    "xyz",
    "top",
    "pw",
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "club",
    "click",
})


# ==============================================================================
# Ports
# ==============================================================================

DEFAULT_SCHEME_PORTS: Final[dict[str, int]] = {
    "http": 80,
    "https": 443,
}

HTTP_PORTS: Final[frozenset[int]] = frozenset({
    80,
    8080,
})

HTTPS_PORTS: Final[frozenset[int]] = frozenset({
    443,
    8443,
})

# ==============================================================================
# Security Thresholds
# ==============================================================================

LONG_QUERY_LENGTH_THRESHOLD: Final[int] = 50
LONG_URL_LENGTH_THRESHOLD: Final[int] = 75
VERY_LONG_URL_LENGTH_THRESHOLD: Final[int] = 150

MULTIPLE_SUBDOMAIN_DOT_THRESHOLD: Final[int] = 2
EXCESSIVE_DOT_THRESHOLD: Final[int] = 3
EXCESSIVE_HYPHEN_THRESHOLD: Final[int] = 3

MAX_SUBDOMAIN_COUNT: Final[int] = 5
MAX_PATH_DEPTH: Final[int] = 10
MAX_QUERY_PARAMETERS: Final[int] = 20

HIGH_ENTROPY_THRESHOLD: Final[float] = 4.5

# ==============================================================================
# Query Parameters
# ==============================================================================

SUSPICIOUS_QUERY_PARAM_NAMES: Final[frozenset[str]] = frozenset({
    "redirect",
    "url",
    "next",
    "return",
    "continue",
    "dest",
    "destination",
    "target",
})

# ==============================================================================
# Model Files
# ==============================================================================

MODEL_NAME: Final[str] = "random_forest.pkl"
SCALER_NAME: Final[str] = "scaler.pkl"
LABEL_ENCODER_NAME: Final[str] = "label_encoder.pkl"
FEATURE_COLUMNS_NAME: Final[str] = "feature_columns.pkl"

# ==============================================================================
# Logging
# ==============================================================================

LOG_LEVEL: Final[str] = "INFO"

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ==============================================================================
# Regex
# ==============================================================================

URL_SCHEME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(https?|ftp)://",
    re.IGNORECASE,
)

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

IP_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
)

MULTIPLE_SLASH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"//.*/+"
)

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({
    "url",
    "type",
})

MODEL_PATH = MODELS_DIR / MODEL_NAME
SCALER_PATH = MODELS_DIR / SCALER_NAME
LABEL_ENCODER_PATH = MODELS_DIR / LABEL_ENCODER_NAME
FEATURE_COLUMNS_PATH = MODELS_DIR / FEATURE_COLUMNS_NAME

URL_COLUMN: Final[str] = "url"
LABEL_COLUMN: Final[str] = "label"