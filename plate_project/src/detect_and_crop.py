import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from utils import (
    add_padding_to_box,
    ensure_dir,
    list_images,
    load_yaml,
    read_image,
    resolve_project_root,
    save_dataframe,
    setup_logger,
    to_absolute,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect plates with YOLO and crop plate regions.")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--source", type=str, default="data/raw", help="Image folder to run detection on.")
    parser.add_argument("--save_annotated", action="store_true", help="Save detection visuals.")
    return parser.parse_args()


def detect_and_crop(config_path: Path, source_dir: Path, save_annotated: bool) -> None:
    logger = setup_logger("detect_and_crop")
    project_root = resolve_project_root()
    cfg = load_yaml(config_path)

    yolo_cfg = cfg["yolo"]
    paths_cfg = cfg["paths"]

    weights = to_absolute(project_root, paths_cfg["yolo_weights"])
    crops_dir = to_absolute(project_root, paths_cfg["cropped_plates"])
    detections_dir = project_root / "outputs" / "detections"
    metadata_path = project_root / "outputs" / "crops" / "crop_metadata.csv"

    ensure_dir(crops_dir)
    ensure_dir(detections_dir)
    ensure_dir(metadata_path.parent)

    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    image_paths = list_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in source folder: {source_dir}")

    model = YOLO(str(weights))
    rows = []

    for image_path in image_paths:
        image = read_image(image_path)
        h, w = image.shape[:2]

        result = model.predict(
            source=image,
            conf=float(yolo_cfg["conf_threshold"]),
            iou=float(yolo_cfg["iou_threshold"]),
            max_det=int(yolo_cfg["max_det"]),
            verbose=False,
        )[0]

        annotated = image.copy()
        boxes = result.boxes
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = add_padding_to_box(
                int(x1), int(y1), int(x2), int(y2), float(yolo_cfg["crop_padding"]), w, h
            )

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            base_name = image_path.stem
            crop_name = f"{base_name}_plate_{i:02d}.png"
            crop_path = crops_dir / crop_name
            write_image(crop_path, crop)

            rows.append(
                {
                    "source_image": str(image_path),
                    "crop_image": str(crop_path),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "confidence": conf,
                }
            )

            if save_annotated:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"plate {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if save_annotated:
            write_image(detections_dir / f"{image_path.stem}_det.jpg", annotated)

    df = pd.DataFrame(rows)
    save_dataframe(metadata_path, df)
    logger.info("Saved %d crops. Metadata: %s", len(df), metadata_path)


if __name__ == "__main__":
    args = parse_args()
    root = resolve_project_root()
    cfg_path = to_absolute(root, args.config)
    source = to_absolute(root, args.source)
    detect_and_crop(cfg_path, source, args.save_annotated)
