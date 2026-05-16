import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from utils import load_yaml, resolve_project_root, save_json, seed_everything, setup_logger, to_absolute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for license plate detection.")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml", help="Path to YOLO config.")
    return parser.parse_args()


def train_yolo(config_path: Path) -> None:
    logger = setup_logger("train_yolo")
    project_root = resolve_project_root()
    cfg = load_yaml(config_path)

    seed_everything(int(cfg["project"]["seed"]))

    dataset_yaml = to_absolute(project_root, cfg["paths"]["dataset_yaml"])
    output_root = to_absolute(project_root, cfg["paths"]["output_dir"])
    model_dir = to_absolute(project_root, cfg["paths"]["model_dir"])

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    train_cfg = cfg["train"]
    model = YOLO(train_cfg["pretrained_model"])

    logger.info("Starting YOLO training with dataset: %s", dataset_yaml)
    results = model.train(
        data=str(dataset_yaml),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        workers=int(train_cfg["workers"]),
        patience=int(train_cfg["patience"]),
        lr0=float(train_cfg["lr0"]),
        optimizer=train_cfg["optimizer"],
        cos_lr=bool(train_cfg["cos_lr"]),
        project=str(output_root),
        name=train_cfg["project_name"],
        device=train_cfg["device"],
    )

    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    if best_weight.exists():
        shutil.copy2(best_weight, model_dir / "best.pt")
        logger.info("Copied best model to: %s", model_dir / "best.pt")

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "save_dir": str(results.save_dir),
        "best_weight": str(model_dir / "best.pt"),
        "epochs": int(train_cfg["epochs"]),
        "imgsz": int(train_cfg["imgsz"]),
    }
    save_json(project_root / "outputs" / "evaluation" / "yolo_train_summary.json", summary)
    logger.info("YOLO training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg_path = to_absolute(resolve_project_root(), args.config)
    train_yolo(cfg_path)
