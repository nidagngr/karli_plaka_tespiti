import argparse
from pathlib import Path

import cv2
import easyocr
from gan_restoration import build_restorer_from_config
from ocr_utils import choose_best_plate_candidate, format_plate_candidates, read_plate_candidates
from preprocess_utils import adjust_contrast, apply_clahe, clean_noise, ocr_preprocess, sharpen_image
from ultralytics import YOLO

from utils import (
    add_padding_to_box,
    ensure_dir,
    load_yaml,
    read_image,
    resolve_project_root,
    setup_logger,
    to_absolute,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: image -> YOLO -> crop -> OCR")
    parser.add_argument("--image", type=str, required=True, help="Input vehicle image path")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    return parser.parse_args()


def run_pipeline(image_path: Path, config_path: Path) -> None:
    logger = setup_logger("inference_pipeline")
    root = resolve_project_root()
    cfg = load_yaml(config_path)

    img = read_image(image_path)
    h, w = img.shape[:2]

    yolo_weights = to_absolute(root, cfg["paths"]["yolo_weights"])
    out_det = ensure_dir(root / "outputs" / "detections")
    out_crop = ensure_dir(root / "outputs" / "crops")
    out_ocr = ensure_dir(root / "outputs" / "ocr_results")
    out_restored = ensure_dir(out_ocr / "single_restored")
    out_steps = ensure_dir(out_ocr / "pipeline_steps")

    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weight missing: {yolo_weights}")

    yolo_cfg = cfg["yolo"]
    yolo_model = YOLO(str(yolo_weights))

    result = yolo_model.predict(
        source=img,
        conf=float(yolo_cfg["conf_threshold"]),
        iou=float(yolo_cfg["iou_threshold"]),
        max_det=int(yolo_cfg["max_det"]),
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        logger.warning("No plate detected.")
        return

    box = result.boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = add_padding_to_box(
        int(x1), int(y1), int(x2), int(y2), float(yolo_cfg["crop_padding"]), w, h
    )
    crop = img[y1:y2, x1:x2]
    crop_path = out_crop / f"{image_path.stem}_crop.png"
    write_image(crop_path, crop)

    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    det_path = out_det / f"{image_path.stem}_det.jpg"
    write_image(det_path, vis)

    ocr_cfg = cfg["ocr"]
    reader = easyocr.Reader(ocr_cfg["languages"], gpu=bool(ocr_cfg["gpu"]))
    restorer = build_restorer_from_config(root, cfg)

    clahe = apply_clahe(crop)
    contrast = adjust_contrast(clahe)
    sharpened = sharpen_image(contrast)
    cleaned = clean_noise(sharpened)
    ocr_ready = ocr_preprocess(cleaned)
    crop_for_ocr = restorer.restore(ocr_ready) if restorer is not None else ocr_ready
    restored_crop_path = out_restored / f"{image_path.stem}_restored.png"
    write_image(out_steps / f"{image_path.stem}_01_crop.png", crop)
    write_image(out_steps / f"{image_path.stem}_02_clahe.png", clahe)
    write_image(out_steps / f"{image_path.stem}_03_contrast.png", contrast)
    write_image(out_steps / f"{image_path.stem}_04_sharpening.png", sharpened)
    write_image(out_steps / f"{image_path.stem}_05_noise_cleaning.png", cleaned)
    write_image(out_steps / f"{image_path.stem}_06_ocr_preprocessing.png", ocr_ready)
    if restorer is not None and bool(cfg.get("gan", {}).get("save_restored", True)):
        write_image(restored_crop_path, crop_for_ocr)

    min_confidence = float(ocr_cfg.get("min_confidence", 0.0))
    candidates = read_plate_candidates(reader, crop_for_ocr, str(ocr_cfg["allowlist"]), min_confidence)
    best = choose_best_plate_candidate(candidates, beam_width=1)
    final_plate = best.normalized

    summary = out_ocr / f"{image_path.stem}_pipeline_result.txt"
    summary.write_text(
        "\n".join(
            [
                f"input_image: {image_path}",
                "pipeline: YOLO -> Crop -> CLAHE -> Contrast Adjustment -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR",
                f"crop_path: {crop_path}",
                f"restored_crop_path: {restored_crop_path if restorer is not None else ''}",
                f"final_plate: {final_plate}",
                f"ocr_confidence: {best.confidence:.4f}",
                f"ocr_source: {best.source}",
                f"ocr_raw_candidates: {' | '.join(candidate.raw_text for candidate in candidates)}",
                f"ocr_ranked_candidates: {format_plate_candidates(candidates, beam_width=1)}",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Pipeline complete. OCR: %s", final_plate)
    print(f"PLAKA: {final_plate}")


if __name__ == "__main__":
    args = parse_args()
    root = resolve_project_root()
    run_pipeline(
        to_absolute(root, args.image),
        to_absolute(root, args.config),
    )
