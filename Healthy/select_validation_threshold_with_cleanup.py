import csv
from pathlib import Path

import nrrd
import numpy as np
import torch
from scipy.ndimage import label

from config import device, model
from dataset import TRAINING_MEAN, TRAINING_STD


VALID_SCAN_DIRECTORY = Path("data/scan_valid")
VALID_MASK_DIRECTORY = Path("data/segment_valid")

CHECKPOINT_PATH = Path(
    "checkpoints_fixed_training_normalization/model_50.pth"
)

CSV_PATH = Path(
    "validation_threshold_with_largest_component.csv"
)

MASK_SUFFIX = "_GT"

THRESHOLDS = [
    round(value, 2)
    for value in np.arange(0.30, 1.00, 0.01)
]

# Face-, edge-, and corner-touching voxels count as connected.
CONNECTIVITY_STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8,
)


def load_and_normalize_scan(scan_path: Path):
    scan_array, _ = nrrd.read(str(scan_path))

    scan_array = np.ascontiguousarray(
        scan_array,
        dtype=np.float32,
    )

    scan_tensor = torch.from_numpy(scan_array)

    scan_tensor = (
        scan_tensor - TRAINING_MEAN
    ) / TRAINING_STD

    # [D, H, W] -> [1, 1, D, H, W]
    return scan_tensor.unsqueeze(0).unsqueeze(0)


def keep_largest_component(binary_mask: np.ndarray):
    labeled_mask, number_of_components = label(
        binary_mask,
        structure=CONNECTIVITY_STRUCTURE,
    )

    if number_of_components == 0:
        return np.zeros_like(
            binary_mask,
            dtype=np.uint8,
        )

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    # Label 0 is background.
    component_sizes[0] = 0

    largest_component_label = int(
        np.argmax(component_sizes)
    )

    return (
        labeled_mask == largest_component_label
    ).astype(np.uint8)


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~target,
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        target,
    ).sum()

    epsilon = 1e-8

    dice = (
        2.0 * true_positive
        / (
            2.0 * true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
            + epsilon
        )
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
            + epsilon
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def average_metric(metric_rows, metric_name):
    return (
        sum(row[metric_name] for row in metric_rows)
        / len(metric_rows)
    )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    scan_paths = sorted(
        VALID_SCAN_DIRECTORY.glob("*.nrrd")
    )

    if len(scan_paths) == 0:
        raise RuntimeError(
            f"No validation scans found in "
            f"{VALID_SCAN_DIRECTORY}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    print(f"Using device: {device}")
    print(f"Found {len(scan_paths)} validation scans.")
    print()

    probabilities_by_scan = []

    with torch.no_grad():
        for scan_path in scan_paths:
            scan_id = scan_path.stem

            mask_path = (
                VALID_MASK_DIRECTORY
                / f"{scan_id}{MASK_SUFFIX}.nrrd"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Ground-truth mask not found: "
                    f"{mask_path}"
                )

            scan_tensor = load_and_normalize_scan(
                scan_path
            ).to(device)

            logits = model(scan_tensor)

            probabilities = (
                torch.sigmoid(logits)
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            ground_truth, _ = nrrd.read(
                str(mask_path)
            )

            ground_truth = np.ascontiguousarray(
                (ground_truth > 0).astype(np.uint8)
            )

            probabilities_by_scan.append(
                (
                    scan_id,
                    probabilities,
                    ground_truth,
                )
            )

    results = []

    best_raw_threshold = None
    best_raw_dice = -1.0

    best_cleaned_threshold = None
    best_cleaned_dice = -1.0

    print(
        "Threshold | Raw Dice | Clean Dice | "
        "Clean Precision | Clean Recall"
    )
    print("-" * 68)

    for threshold in THRESHOLDS:
        raw_metric_rows = []
        cleaned_metric_rows = []

        for (
            _,
            probabilities,
            ground_truth,
        ) in probabilities_by_scan:

            raw_prediction = (
                probabilities >= threshold
            ).astype(np.uint8)

            cleaned_prediction = keep_largest_component(
                raw_prediction
            )

            raw_metrics = calculate_metrics(
                raw_prediction,
                ground_truth,
            )

            cleaned_metrics = calculate_metrics(
                cleaned_prediction,
                ground_truth,
            )

            raw_metric_rows.append(raw_metrics)
            cleaned_metric_rows.append(cleaned_metrics)

        raw_average_dice = average_metric(
            raw_metric_rows,
            "dice",
        )

        raw_average_iou = average_metric(
            raw_metric_rows,
            "iou",
        )

        raw_average_precision = average_metric(
            raw_metric_rows,
            "precision",
        )

        raw_average_recall = average_metric(
            raw_metric_rows,
            "recall",
        )

        cleaned_average_dice = average_metric(
            cleaned_metric_rows,
            "dice",
        )

        cleaned_average_iou = average_metric(
            cleaned_metric_rows,
            "iou",
        )

        cleaned_average_precision = average_metric(
            cleaned_metric_rows,
            "precision",
        )

        cleaned_average_recall = average_metric(
            cleaned_metric_rows,
            "recall",
        )

        results.append(
            {
                "threshold": threshold,
                "raw_dice": raw_average_dice,
                "raw_iou": raw_average_iou,
                "raw_precision": raw_average_precision,
                "raw_recall": raw_average_recall,
                "cleaned_dice": cleaned_average_dice,
                "cleaned_iou": cleaned_average_iou,
                "cleaned_precision": cleaned_average_precision,
                "cleaned_recall": cleaned_average_recall,
            }
        )

        print(
            f"{threshold:9.2f} | "
            f"{raw_average_dice:8.4f} | "
            f"{cleaned_average_dice:10.4f} | "
            f"{cleaned_average_precision:15.4f} | "
            f"{cleaned_average_recall:12.4f}"
        )

        if raw_average_dice > best_raw_dice:
            best_raw_dice = raw_average_dice
            best_raw_threshold = threshold

        if cleaned_average_dice > best_cleaned_dice:
            best_cleaned_dice = cleaned_average_dice
            best_cleaned_threshold = threshold

    fieldnames = list(results[0].keys())

    with CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 58)
    print(
        f"Best raw threshold: "
        f"{best_raw_threshold:.2f}"
    )
    print(
        f"Best raw validation Dice: "
        f"{best_raw_dice:.4f}"
    )
    print()
    print(
        f"Best threshold with largest-component cleanup: "
        f"{best_cleaned_threshold:.2f}"
    )
    print(
        f"Best cleaned validation Dice: "
        f"{best_cleaned_dice:.4f}"
    )
    print("=" * 58)
    print(f"Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()

