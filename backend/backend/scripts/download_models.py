from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import os
import shutil


MODEL_URL = os.environ.get(
    "MODEL_ARCHIVE_URL",
    "https://github.com/hariharand28/SentinelWeb/releases/latest/download/sentinelweb-models.zip",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
ARCHIVE_PATH = BASE_DIR / "sentinelweb-models.zip"

REQUIRED_FILES = {
    "feature_columns.pkl",
    "label_encoder.pkl",
    "random_forest.pkl",
    "scaler.pkl",
}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SentinelWeb ML model artifacts...")
    print(f"Source: {MODEL_URL}")

    with urlopen(MODEL_URL, timeout=300) as response:
        with ARCHIVE_PATH.open("wb") as output:
            shutil.copyfileobj(response, output)

    print("Model archive downloaded.")

    with ZipFile(ARCHIVE_PATH, "r") as archive:
        names = {
            Path(name).name
            for name in archive.namelist()
            if not name.endswith("/")
        }

        missing = REQUIRED_FILES - names

        if missing:
            raise RuntimeError(
                f"Model archive is missing required files: {sorted(missing)}"
            )

        archive.extractall(MODELS_DIR)

    ARCHIVE_PATH.unlink(missing_ok=True)

    missing_local = [
        name for name in REQUIRED_FILES
        if not (MODELS_DIR / name).exists()
    ]

    if missing_local:
        raise RuntimeError(
            f"Model installation failed. Missing: {missing_local}"
        )

    print("All SentinelWeb ML model artifacts installed successfully.")


if __name__ == "__main__":
    main()