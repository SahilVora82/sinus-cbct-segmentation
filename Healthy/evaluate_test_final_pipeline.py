import csv
from pathlib import Path

import nrrd
import numpy as np
import torch
from scipy.ndimage import label

from config import device, model
from dataset import TRAINING_MEAN, TRAINING_STD


TEST_SCAN_DIRECTORY = Path("data/scan_test")
TEST_MASK_DIRECTORY = Path("data/segment_test")

CHECKPOINT_PATH = Path(
    "checkpoints_fixed_training_normalization/model_50.pth"
)

OUTPUT_DIRECTORY = Path(
    "data/test_predictions_final_threshold_084_largest_component"
)

CSV_PATH = Path(
    "test_metrics_final_threshold_084_largest_component.csv"
)

MASK_SUFFIX = "_GT"
PREDICTION_SUFFIX = "_PRED"

# Selected using the validation set.
THRESHOLD = 0.84

# 26-connectivity:
# Voxels touching by a face, edge, or corner are connected.
CONNECTIVITY_STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8,
)


def load_and_normalize_scan(scan_path: Path):
    scan_array, scan_header = nrrd.read(str(scan_path))

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

    return scan_tensor, scan_header


def keep_largest_component(binary_mask: np.ndarray):
    labeled_mask, number_of_components = label(
        binary_mask,
        structure=CONNECTIVITY_STRUCTURE,
    )

    if number_of_components == 0:
        return (
            np.zeros_like(binary_mask, dtype=np.uint8),
            0,
            0,
        )

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    # Component 0 is background.
    component_sizes[0] = 0

    largest_component_label = int(
        np.argmax(component_sizes)
    )

    cleaned_mask = (
        labeled_mask == largest_component_label
    ).astype(np.uint8)

    removed_voxels = int(
        binary_mask.sum() - cleaned_mask.sum()
    )

    return (
        cleaned_mask,
        number_of_components,
        removed_voxels,
    )


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = int(
        np.logical_and(prediction, target).sum()
    )

    false_positive = int(
        np.logical_and(prediction, ~target).sum()
    )

    false_negative = int(
        np.logical_and(~prediction, target).sum()
    )

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
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def average(results, metric_name):
    return (
        sum(result[metric_name] for result in results)
        / len(results)
    )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    scan_paths = sorted(
        TEST_SCAN_DIRECTORY.glob("*.nrrd")
    )

    if len(scan_paths) == 0:
        raise RuntimeError(
            f"No test scans found in "
            f"{TEST_SCAN_DIRECTORY}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
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
    print(f"Saved epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Using device: {device}")
    print(f"Training mean: {TRAINING_MEAN}")
    print(f"Training standard deviation: {TRAINING_STD}")
    print(f"Threshold: {THRESHOLD}")
    print("Largest-component cleanup: Yes")
    print()
    print(f"Found {len(scan_paths)} test scans.")
    print()

    results = []

    with torch.no_grad():
        for scan_path in scan_paths:
            scan_id = scan_path.stem

            mask_path = (
                TEST_MASK_DIRECTORY
                / f"{scan_id}{MASK_SUFFIX}.nrrd"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Ground-truth mask not found: "
                    f"{mask_path}"
                )

            scan_tensor, _ = load_and_normalize_scan(
                scan_path
            )

            scan_tensor = scan_tensor.to(device)

            logits = model(scan_tensor)

            probabilities = torch.sigmoid(logits)

            raw_prediction = (
                probabilities >= THRESHOLD
            ).squeeze(0).squeeze(0).cpu().numpy()

            raw_prediction = np.ascontiguousarray(
                raw_prediction.astype(np.uint8)
            )

            (
                cleaned_prediction,
                number_of_components,
                removed_voxels,
            ) = keep_largest_component(
                raw_prediction
            )

            ground_truth, mask_header = nrrd.read(
                str(mask_path)
            )

            ground_truth = np.ascontiguousarray(
                (ground_truth > 0).astype(np.uint8)
            )

            metrics = calculate_metrics(
                cleaned_prediction,
                ground_truth,
            )

            prediction_path = (
                OUTPUT_DIRECTORY
                / f"{scan_id}{PREDICTION_SUFFIX}.nrrd"
            )

            nrrd.write(
                str(prediction_path),
                cleaned_prediction.astype(np.uint8),
                header=mask_header,
            )

            row = {
                "scan": scan_id,
                "threshold": THRESHOLD,
                "components_before_cleanup": (
                    number_of_components
                ),
                "removed_voxels": removed_voxels,
                **metrics,
            }

            results.append(row)

            print(scan_id)
            print(
                f"  Components before cleanup: "
                f"{number_of_components}"
            )
            print(
                f"  Removed voxels: "
                f"{removed_voxels}"
            )
            print(
                f"  Dice:      "
                f"{metrics['dice']:.4f}"
            )
            print(
                f"  IoU:       "
                f"{metrics['iou']:.4f}"
            )
            print(
                f"  Precision: "
                f"{metrics['precision']:.4f}"
            )
            print(
                f"  Recall:    "
                f"{metrics['recall']:.4f}"
            )
            print(
                f"  Saved:     "
                f"{prediction_path}"
            )
            print()

    average_dice = average(results, "dice")
    average_iou = average(results, "iou")
    average_precision = average(
        results,
        "precision",
    )
    average_recall = average(
        results,
        "recall",
    )

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
    print("FINAL TEST RESULTS")
    print()
    print(f"Dice:      {average_dice:.4f}")
    print(f"IoU:       {average_iou:.4f}")
    print(f"Precision: {average_precision:.4f}")
    print(f"Recall:    {average_recall:.4f}")
    print("=" * 60)
    print(f"Metrics saved to: {CSV_PATH}")
    print(
        f"Predictions saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()