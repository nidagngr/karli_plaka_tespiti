import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import character_accuracy, normalize_plate_text, resolve_project_root, save_dataframe, save_json, setup_logger, to_absolute

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions against available ground truth.")
    parser.add_argument("--results", type=str, default="outputs/ocr_results/ocr_results.csv")
    parser.add_argument("--pairs", type=str, default="data/synthetic_snow_pairs/pairs.csv")
    return parser.parse_args()


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_plate_text(pred) == normalize_plate_text(gt))


def evaluate(results_csv: Path, pairs_csv: Path) -> None:
    logger = setup_logger("evaluate_pipeline")
    root = resolve_project_root()

    results_df = pd.read_csv(results_csv)
    pairs_df = pd.read_csv(pairs_csv)

    if "plate_text" in pairs_df.columns:
        gt_map = pairs_df[["sample_id", "plate_text"]].drop_duplicates()
        results_df = results_df.merge(gt_map, on="sample_id", how="left", suffixes=("", "_pairs"))

    results_df["gt"] = results_df.get("ground_truth", results_df.get("plate_text", "")).fillna("")
    results_df["gt"] = results_df["gt"].astype(str)
    eval_df = results_df[results_df["gt"].str.strip() != ""].copy()

    if len(eval_df) == 0:
        metrics = {
            "num_samples": int(len(results_df)),
            "num_samples_with_gt": 0,
            "message": "No ground-truth plate_text values found. Evaluation metrics were skipped.",
        }
        details_path = root / "outputs" / "evaluation" / "ocr_comparison_details.csv"
        metrics_path = root / "outputs" / "evaluation" / "ocr_comparison_metrics.json"
        save_dataframe(details_path, results_df)
        save_json(metrics_path, metrics)
        logger.warning(metrics["message"])
        return

    eval_df["exact_before"] = eval_df.apply(lambda x: exact_match(str(x.get("prediction_before", "")), str(x["gt"])), axis=1)
    eval_df["exact_after"] = eval_df.apply(lambda x: exact_match(str(x.get("prediction_after", "")), str(x["gt"])), axis=1)
    eval_df["char_acc_before"] = eval_df.apply(
        lambda x: character_accuracy(str(x.get("prediction_before", "")), str(x["gt"])), axis=1
    )
    eval_df["char_acc_after"] = eval_df.apply(
        lambda x: character_accuracy(str(x.get("prediction_after", "")), str(x["gt"])), axis=1
    )

    metrics = {
        "num_samples": int(len(results_df)),
        "num_samples_with_gt": int(len(eval_df)),
        "exact_match_before": float(np.mean(eval_df["exact_before"])),
        "exact_match_after": float(np.mean(eval_df["exact_after"])),
        "char_acc_before": float(np.mean(eval_df["char_acc_before"])),
        "char_acc_after": float(np.mean(eval_df["char_acc_after"])),
        "exact_match_gain": float(np.mean(eval_df["exact_after"]) - np.mean(eval_df["exact_before"])),
        "char_acc_gain": float(np.mean(eval_df["char_acc_after"]) - np.mean(eval_df["char_acc_before"])),
    }

    details_path = root / "outputs" / "evaluation" / "ocr_comparison_details.csv"
    metrics_path = root / "outputs" / "evaluation" / "ocr_comparison_metrics.json"
    plot_path = root / "outputs" / "evaluation" / "ocr_metrics_bar.png"

    save_dataframe(details_path, eval_df)
    save_json(metrics_path, metrics)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Exact Before", "Exact After", "Char Before", "Char After"]
    values = [
        metrics["exact_match_before"],
        metrics["exact_match_after"],
        metrics["char_acc_before"],
        metrics["char_acc_after"],
    ]
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("OCR Performance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    logger.info("Evaluation finished.")
    logger.info("Exact match before/after: %.4f / %.4f", metrics["exact_match_before"], metrics["exact_match_after"])
    logger.info("Char accuracy before/after: %.4f / %.4f", metrics["char_acc_before"], metrics["char_acc_after"])


if __name__ == "__main__":
    args = parse_args()
    root = resolve_project_root()
    evaluate(
        to_absolute(root, args.results),
        to_absolute(root, args.pairs),
    )
