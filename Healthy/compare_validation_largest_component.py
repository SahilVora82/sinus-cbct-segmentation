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
    "validation_largest_component_comparison.csv"
)

MASK_SUFFIX = "_GT"
THRESHOLD = 0.50

# 26-connectivity:
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
    scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)

    return scan_tensor


def keep_largest_component(
    binary_mask: np.ndarray,
):
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

    # Ignore label 0 because it is background.
    component_sizes[0] = 0

    largest_component_label = int(
        np.argmax(component_sizes)
    )

    cleaned_mask = (
        labeled_mask == largest_component_label
    ).astype(np.uint8)

    return cleaned_mask


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
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


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
    print(f"Threshold: {THRESHOLD}")
    print(f"Found {len(scan_paths)} validation scans.")
    print()

    results = []

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
            probabilities = torch.sigmoid(logits)

            raw_prediction = (
                probabilities >= THRESHOLD
            ).squeeze(0).squeeze(0).cpu().numpy()

            raw_prediction = np.ascontiguousarray(
                raw_prediction.astype(np.uint8)
            )

            cleaned_prediction = keep_largest_component(
                raw_prediction
            )

            ground_truth, _ = nrrd.read(
                str(mask_path)
            )

            ground_truth = np.ascontiguousarray(
                (ground_truth > 0).astype(np.uint8)
            )

            raw_metrics = calculate_metrics(
                raw_prediction,
                ground_truth,
            )

            cleaned_metrics = calculate_metrics(
                cleaned_prediction,
                ground_truth,
            )

            removed_voxels = int(
                raw_prediction.sum()
                - cleaned_prediction.sum()
            )

            result = {
                "scan": scan_id,
                "removed_voxels": removed_voxels,

                "raw_dice": raw_metrics["dice"],
                "cleaned_dice": cleaned_metrics["dice"],
                "dice_change": (
                    cleaned_metrics["dice"]
                    - raw_metrics["dice"]
                ),

                "raw_iou": raw_metrics["iou"],
                "cleaned_iou": cleaned_metrics["iou"],

                "raw_precision": raw_metrics["precision"],
                "cleaned_precision": cleaned_metrics["precision"],

                "raw_recall": raw_metrics["recall"],
                "cleaned_recall": cleaned_metrics["recall"],
            }

            results.append(result)

            print(scan_id)
            print(
                f"  Removed voxels: "
                f"{removed_voxels}"
            )
            print(
                f"  Dice:      "
                f"{raw_metrics['dice']:.4f} -> "
                f"{cleaned_metrics['dice']:.4f}"
            )
            print(
                f"  IoU:       "
                f"{raw_metrics['iou']:.4f} -> "
                f"{cleaned_metrics['iou']:.4f}"
            )
            print(
                f"  Precision: "
                f"{raw_metrics['precision']:.4f} -> "
                f"{cleaned_metrics['precision']:.4f}"
            )
            print(
                f"  Recall:    "
                f"{raw_metrics['recall']:.4f} -> "
                f"{cleaned_metrics['recall']:.4f}"
            )
            print()

    raw_average_dice = sum(
        row["raw_dice"] for row in results
    ) / len(results)

    cleaned_average_dice = sum(
        row["cleaned_dice"] for row in results
    ) / len(results)

    raw_average_iou = sum(
        row["raw_iou"] for row in results
    ) / len(results)

    cleaned_average_iou = sum(
        row["cleaned_iou"] for row in results
    ) / len(results)

    raw_average_precision = sum(
        row["raw_precision"] for row in results
    ) / len(results)

    cleaned_average_precision = sum(
        row["cleaned_precision"] for row in results
    ) / len(results)

    raw_average_recall = sum(
        row["raw_recall"] for row in results
    ) / len(results)

    cleaned_average_recall = sum(
        row["cleaned_recall"] for row in results
    ) / len(results)

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

    print("=" * 60)
    print("AVERAGE VALIDATION RESULTS")
    print()
    print(
        f"Dice:      "
        f"{raw_average_dice:.4f} -> "
        f"{cleaned_average_dice:.4f}"
    )
    print(
        f"IoU:       "
        f"{raw_average_iou:.4f} -> "
        f"{cleaned_average_iou:.4f}"
    )
    print(
        f"Precision: "
        f"{raw_average_precision:.4f} -> "
        f"{cleaned_average_precision:.4f}"
    )
    print(
        f"Recall:    "
        f"{raw_average_recall:.4f} -> "
        f"{cleaned_average_recall:.4f}"
    )
    print("=" * 60)
    print(f"Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()