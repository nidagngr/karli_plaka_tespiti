import argparse
from pathlib import Path

import easyocr
import pandas as pd
from tqdm import tqdm

from gan_restoration import build_restorer_from_config
from ocr_utils import choose_best_plate_candidate, format_plate_candidates, read_plate_candidates
from utils import ensure_dir, load_yaml, read_image, resolve_project_root, save_dataframe, setup_logger, to_absolute, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR on degraded plate images.")
    parser.add_argument("--input_csv", type=str, default="data/synthetic_snow_pairs/pairs.csv")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    return parser.parse_args()


def run_ocr(input_csv: Path, config_path: Path) -> None:
    logger = setup_logger("run_ocr")
    cfg = load_yaml(config_path)
    root = resolve_project_root()

    output_dir = ensure_dir(root / "outputs" / "ocr_results")
    restored_dir = ensure_dir(output_dir / "restored")
    ocr_cfg = cfg["ocr"]
    reader = easyocr.Reader(ocr_cfg["languages"], gpu=bool(ocr_cfg["gpu"]))
    restorer = build_restorer_from_config(root, cfg)
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running OCR"):
        snow_path = Path(row["snow_path"])
        clean_path = Path(row.get("clean_path", "")) if "clean_path" in row else Path("")
        gt_text = str(row.get("plate_text", ""))
        sample_id = str(row.get("sample_id", snow_path.name))

        snow_img = read_image(snow_path)
        restored_img = restorer.restore(snow_img) if restorer is not None else snow_img.copy()

        min_confidence = float(ocr_cfg.get("min_confidence", 0.0))
        beam_width = int(ocr_cfg.get("beam_width", 8))
        before_candidates = read_plate_candidates(reader, snow_img, str(ocr_cfg["allowlist"]), min_confidence)
        after_candidates = read_plate_candidates(reader, restored_img, str(ocr_cfg["allowlist"]), min_confidence)

        best_before = choose_best_plate_candidate(before_candidates, beam_width=beam_width)
        best_after = choose_best_plate_candidate(after_candidates, beam_width=beam_width)
        pred_before = best_before.normalized
        pred_after = best_after.normalized
        restored_path = restored_dir / snow_path.name
        if restorer is not None and bool(cfg.get("gan", {}).get("save_restored", True)):
            write_image(restored_path, restored_img)

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(snow_path),
                "prediction_before": pred_before,
                "prediction_after": pred_after,
                "prediction_before_confidence": best_before.confidence,
                "prediction_after_confidence": best_after.confidence,
                "prediction_before_source": best_before.source,
                "prediction_after_source": best_after.source,
                "ground_truth": gt_text,
                "clean_path": str(clean_path),
                "restored_path": str(restored_path) if restorer is not None else "",
                "raw_before_candidates": " | ".join(candidate.raw_text for candidate in before_candidates),
                "raw_after_candidates": " | ".join(candidate.raw_text for candidate in after_candidates),
                "ranked_before_candidates": format_plate_candidates(before_candidates, beam_width=beam_width),
                "ranked_after_candidates": format_plate_candidates(after_candidates, beam_width=beam_width),
            }
        )

    if rows:
        save_dataframe(output_dir / "ocr_results.csv", pd.DataFrame(rows))
        logger.info("Saved OCR results.")


if __name__ == "__main__":
    args = parse_args()
    root = resolve_project_root()
    run_ocr(
        input_csv=to_absolute(root, args.input_csv),
        config_path=to_absolute(root, args.config),
    )
