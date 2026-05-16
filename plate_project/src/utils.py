import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    # Use imdecode + fromfile for reliable Unicode path support on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size > 0 else None
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if ok:
        encoded.tofile(str(path))
        return
    ok = False
    if not ok:
        raise ValueError(f"Failed to write image: {path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    return x1, y1, x2, y2


def add_padding_to_box(x1: int, y1: int, x2: int, y2: int, pad_ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    return clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)


def normalize_plate_text(text: str) -> str:
    if text is None:
        return ""
    chars = [ch for ch in text.upper() if ch.isalnum()]
    return "".join(chars)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(a), len(b)])


def character_accuracy(pred: str, truth: str) -> float:
    pred_n = normalize_plate_text(pred)
    truth_n = normalize_plate_text(truth)
    if len(truth_n) == 0:
        return 1.0 if len(pred_n) == 0 else 0.0

    dist = levenshtein_distance(pred_n, truth_n)
    return max(0.0, 1.0 - (dist / len(truth_n)))


def to_absolute(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
