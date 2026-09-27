from hashlib import sha256
from pathlib import Path
import json

from src.data.load_data import TRAIN_PATH, TEST_PATH, RUL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "reports" / "dataset_manifest.json"


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash for a file."""
    hash_sha256 = sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def build_file_metadata(file_path: Path) -> dict:
    """Create metadata for a dataset file."""
    return {
        "file_name": file_path.name,
        "relative_path": str(file_path.relative_to(PROJECT_ROOT)),
        "size_bytes": file_path.stat().st_size,
        "sha256": calculate_sha256(file_path),
    }


def create_manifest() -> None:
    """Create dataset manifest."""
    files = [
        TRAIN_PATH,
        TEST_PATH,
        RUL_PATH,
    ]

    manifest = {
        "dataset": "NASA C-MAPSS FD001",
        "source": "NASA C-MAPSS Jet Engine Simulated Data",
        "dataset_variant": "FD001",
        "files": [
            build_file_metadata(file_path)
            for file_path in files
        ],
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    print("Dataset manifest created successfully.")
    print(f"Location: {MANIFEST_PATH}")


if __name__ == "__main__":
    create_manifest()