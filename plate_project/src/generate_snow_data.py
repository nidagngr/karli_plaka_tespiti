import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir, list_images, read_image, resolve_project_root, save_dataframe, setup_logger, to_absolute, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired synthetic degraded plates from clean crops.")
    parser.add_argument("--clean_dir", type=str, default="data/cropped_plates")
    parser.add_argument("--output_root", type=str, default="data/synthetic_snow_pairs")
    parser.add_argument("--labels_csv", type=str, default="", help="Optional CSV with crop_image/plate_text columns.")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--apply_perspective_correction", action="store_true")
    parser.add_argument("--apply_enhance", action="store_true")
    return parser.parse_args()


def correct_perspective_if_needed(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    w = int(rect[1][0])
    h = int(rect[1][1])
    if min(w, h) < 10:
        return img

    dst_w = max(w, h)
    dst_h = min(w, h)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    box = order_points_clockwise(box)
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(warped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def enhance_plate(img: np.ndarray) -> np.ndarray:
    up = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(up, d=5, sigmaColor=50, sigmaSpace=50)
    sharp = cv2.addWeighted(denoised, 1.3, cv2.GaussianBlur(denoised, (0, 0), 2.0), -0.3, 0)
    return cv2.resize(sharp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)


def add_snow_overlay(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    snow = np.zeros((h, w), dtype=np.float32)
    num_points = int((h * w) * random.uniform(0.008, 0.02))
    ys = np.random.randint(0, h, size=num_points)
    xs = np.random.randint(0, w, size=num_points)
    snow[ys, xs] = np.random.uniform(0.6, 1.0, size=num_points)
    snow = cv2.GaussianBlur(snow, (0, 0), sigmaX=random.uniform(0.8, 1.8))
    snow = np.clip(snow, 0.0, 1.0)
    snow_3 = np.repeat(snow[:, :, None], 3, axis=2)
    base = img.astype(np.float32) / 255.0
    mixed = np.clip(base * (1.0 - 0.35 * snow_3) + 0.85 * snow_3, 0.0, 1.0)
    return (mixed * 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray) -> np.ndarray:
    sigma = random.uniform(5.0, 20.0)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def add_jpeg_artifact(img: np.ndarray) -> np.ndarray:
    quality = random.randint(25, 65)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def add_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), random.uniform(0.8, 2.0))


def add_motion_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([5, 7, 9, 11])
    kernel = np.zeros((k, k), dtype=np.float32)
    angle = random.uniform(0, 180)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def degrade_plate(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
    out = img.copy()
    flags = {"snow": False, "blur": False, "motion_blur": False, "noise": False, "jpeg": False}

    if random.random() < 0.90:
        out = add_snow_overlay(out)
        flags["snow"] = True
    if random.random() < 0.70:
        out = add_blur(out)
        flags["blur"] = True
    if random.random() < 0.50:
        out = add_motion_blur(out)
        flags["motion_blur"] = True
    if random.random() < 0.60:
        out = add_gaussian_noise(out)
        flags["noise"] = True
    if random.random() < 0.60:
        out = add_jpeg_artifact(out)
        flags["jpeg"] = True
    return out, flags


def load_labels_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if "crop_image" in df.columns and "plate_text" in df.columns:
        return {Path(p).name: str(t) for p, t in zip(df["crop_image"], df["plate_text"])}
    if "image_path" in df.columns and "plate_text" in df.columns:
        return {Path(p).name: str(t) for p, t in zip(df["image_path"], df["plate_text"])}
    return {}


def main(args: argparse.Namespace) -> None:
    logger = setup_logger("generate_snow_data")
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = resolve_project_root()
    clean_dir = to_absolute(root, args.clean_dir)
    output_root = to_absolute(root, args.output_root)
    labels_csv = to_absolute(root, args.labels_csv) if args.labels_csv else None

    clean_train = ensure_dir(output_root / "train" / "clean")
    snow_train = ensure_dir(output_root / "train" / "snow")
    clean_val = ensure_dir(output_root / "val" / "clean")
    snow_val = ensure_dir(output_root / "val" / "snow")
    debug_dir = ensure_dir(root / "outputs" / "synthetic")

    images = list_images(clean_dir)
    if args.max_samples > 0:
        images = images[: args.max_samples]
    if not images:
        raise FileNotFoundError(f"No crop images found: {clean_dir}")

    labels_map = load_labels_map(labels_csv)
    rows: List[Dict] = []

    for i, image_path in enumerate(tqdm(images, desc="Generating paired synthetic data")):
        clean = read_image(image_path)
        if args.apply_perspective_correction:
            clean = correct_perspective_if_needed(clean)
        if args.apply_enhance:
            clean = enhance_plate(clean)

        snow, flags = degrade_plate(clean)
        split = "train" if random.random() < args.train_ratio else "val"
        sample_id = f"plate_{i:06d}.png"

        if split == "train":
            clean_out = clean_train / sample_id
            snow_out = snow_train / sample_id
        else:
            clean_out = clean_val / sample_id
            snow_out = snow_val / sample_id

        write_image(clean_out, clean)
        write_image(snow_out, snow)

        if i < 24:
            write_image(debug_dir / f"{Path(sample_id).stem}_clean.png", clean)
            write_image(debug_dir / f"{Path(sample_id).stem}_snow.png", snow)

        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "clean_path": str(clean_out),
                "snow_path": str(snow_out),
                "source_crop": str(image_path),
                "plate_text": labels_map.get(image_path.name, ""),
                "snow": flags["snow"],
                "blur": flags["blur"],
                "motion_blur": flags["motion_blur"],
                "noise": flags["noise"],
                "jpeg": flags["jpeg"],
            }
        )

    pairs_df = pd.DataFrame(rows)
    pairs_csv = output_root / "pairs.csv"
    save_dataframe(pairs_csv, pairs_df)
    logger.info("Saved pairs CSV: %s (%d samples)", pairs_csv, len(pairs_df))


if __name__ == "__main__":
    main(parse_args())

