from pathlib import Path

from utils import ensure_dir, resolve_project_root, setup_logger


REQUIRED_DIRS = [
    "data/raw",
    "data/yolo_dataset",
    "data/cropped_plates",
    "data/synthetic_snow_pairs",
    "data/processed",
    "data/test_samples",
    "models/yolo",
    "models/ocr",
    "configs",
    "src",
    "outputs/detections",
    "outputs/crops",
    "outputs/synthetic",
    "outputs/ocr_results",
    "outputs/evaluation",
]


def prepare_structure(project_root: Path) -> None:
    logger = setup_logger("prepare_folders")
    for rel in REQUIRED_DIRS:
        path = ensure_dir(project_root / rel)
        logger.info("Ensured: %s", path)


if __name__ == "__main__":
    root = resolve_project_root()
    prepare_structure(root)
