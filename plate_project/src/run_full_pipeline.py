import argparse
import subprocess
import sys
from pathlib import Path

from utils import resolve_project_root, setup_logger, to_absolute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline in fixed order.")
    parser.add_argument("--pipeline_config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--source", type=str, default="data/raw")
    parser.add_argument("--labels_csv", type=str, default="")
    parser.add_argument("--skip_yolo_train", action="store_true")
    return parser.parse_args()


def run_cmd(cmd, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main(args: argparse.Namespace) -> None:
    logger = setup_logger("run_full_pipeline")
    root = resolve_project_root()
    py = sys.executable

    pipeline_cfg = to_absolute(root, args.pipeline_config)
    source = to_absolute(root, args.source)

    logger.info("1) YOLO training (optional)")
    if not args.skip_yolo_train:
        run_cmd([py, "src/train_yolo.py"], root)

    logger.info("2) YOLO detect + crop")
    run_cmd([py, "src/detect_and_crop.py", "--config", str(pipeline_cfg), "--source", str(source), "--save_annotated"], root)

    logger.info("3) Generate synthetic paired snow/degraded data")
    cmd = [py, "src/generate_snow_data.py", "--clean_dir", "data/cropped_plates", "--output_root", "data/synthetic_snow_pairs", "--apply_perspective_correction", "--apply_enhance"]
    if args.labels_csv:
        cmd.extend(["--labels_csv", args.labels_csv])
    run_cmd(cmd, root)

    logger.info("4) Run OCR")
    run_cmd([py, "src/run_ocr.py", "--input_csv", "data/synthetic_snow_pairs/pairs.csv", "--config", str(pipeline_cfg)], root)

    logger.info("5) Evaluate OCR")
    run_cmd([py, "src/evaluate_pipeline.py", "--results", "outputs/ocr_results/ocr_results.csv", "--pairs", "data/synthetic_snow_pairs/pairs.csv"], root)
    logger.info("Full pipeline completed.")


if __name__ == "__main__":
    main(parse_args())
