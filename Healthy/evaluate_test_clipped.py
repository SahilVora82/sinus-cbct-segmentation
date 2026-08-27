import csv
from pathlib import Path

import nrrd
import numpy as np
import torch

from config import device, model


TEST_SCAN_DIRECTORY = Path("data/scan_test")
TEST_MASK_DIRECTORY = Path("data/segment_test")
OUTPUT_DIRECTORY = Path("data/test_predictions")

CHECKPOINT_PATH = Path("checkpoints/model_50.pth")
MASK_SUFFIX = "_GT"
PREDICTION_SUFFIX = "_PRED"
THRESHOLD = 0.5


def calculate_metrics(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.bool()
    target = target.bool()

    true_positive = torch.logical_and(prediction, target).sum().item()
    false_positive = torch.logical_and(prediction, ~target).sum().item()
    false_negative = torch.logical_and(~prediction, target).sum().item()
    true_negative = torch.logical_and(~prediction, ~target).sum().item()

    epsilon = 1e-8

    dice = (
        2.0 * true_positive
        / (2.0 * true_positive + false_positive + false_negative + epsilon)
    )

    iou = (
        true_positive
        / (true_positive + false_positive + false_negative + epsilon)
    )

    precision = (
        true_positive
        / (true_positive + false_positive + epsilon)
    )

    recall = (
        true_positive
        / (true_positive + false_negative + epsilon)
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def load_and_normalize_scan(scan_path: Path):
    scan_array, scan_header = nrrd.read(str(scan_path))

    scan_array = np.ascontiguousarray(scan_array, dtype=np.float32)
    scan_array = np.clip(scan_array, -1000, 3000)
    scan_tensor = torch.from_numpy(scan_array)

    mean = scan_tensor.mean()
    standard_deviation = scan_tensor.std().clamp_min(1e-8)

    scan_tensor = (scan_tensor - mean) / standard_deviation

    # [D, H, W] -> [1, 1, D, H, W]
    scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)

    return scan_tensor, scan_header


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {CHECKPOINT_PATH}"
        )

    if not TEST_SCAN_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Test scan directory was not found: {TEST_SCAN_DIRECTORY}"
        )

    if not TEST_MASK_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Test mask directory was not found: {TEST_MASK_DIRECTORY}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

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
    print()

    scan_paths = sorted(TEST_SCAN_DIRECTORY.glob("*.nrrd"))

    if len(scan_paths) == 0:
        raise RuntimeError(
            f"No NRRD test scans were found in {TEST_SCAN_DIRECTORY}"
        )

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
                    f"Matching test mask was not found for {scan_path.name}: "
                    f"{mask_path}"
                )

            scan_tensor, _ = load_and_normalize_scan(scan_path)
            mask_array, mask_header = nrrd.read(str(mask_path))

            mask_array = np.ascontiguousarray(mask_array)
            target = torch.from_numpy(mask_array > 0)

            scan_tensor = scan_tensor.to(device)

            logits = model(scan_tensor)
            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities >= THRESHOLD
            ).squeeze(0).squeeze(0).cpu()

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Shape mismatch for {scan_id}: "
                    f"prediction={tuple(prediction.shape)}, "
                    f"target={tuple(target.shape)}"
                )

            metrics = calculate_metrics(prediction, target)

            prediction_array = prediction.numpy().astype(np.uint8)

            output_path = (
                OUTPUT_DIRECTORY
                / f"{scan_id}{PREDICTION_SUFFIX}.nrrd"
            )

            prediction_header = dict(mask_header)

            # Let pynrrd determine these values from the prediction array.
            for key in [
                "type",
                "dimension",
                "sizes",
                "endian",
                "encoding",
            ]:
                prediction_header.pop(key, None)

            nrrd.write(
                str(output_path),
                prediction_array,
                header=prediction_header,
            )

            result = {
                "scan": scan_path.name,
                "prediction": output_path.name,
                **metrics,
            }

            results.append(result)

            print(scan_id)
            print(f"  Dice:      {metrics['dice']:.4f}")
            print(f"  IoU:       {metrics['iou']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  Saved:     {output_path}")
            print()

    average_dice = sum(row["dice"] for row in results) / len(results)
    average_iou = sum(row["iou"] for row in results) / len(results)
    average_precision = (
        sum(row["precision"] for row in results) / len(results)
    )
    average_recall = (
        sum(row["recall"] for row in results) / len(results)
    )

    csv_path = Path("test_metrics.csv")

    fieldnames = [
        "scan",
        "prediction",
        "dice",
        "iou",
        "precision",
        "recall",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 45)
    print("AVERAGE TEST RESULTS")
    print(f"Dice:      {average_dice:.4f}")
    print(f"IoU:       {average_iou:.4f}")
    print(f"Precision: {average_precision:.4f}")
    print(f"Recall:    {average_recall:.4f}")
    print("=" * 45)
    print(f"Metrics saved to: {csv_path}")
    print(f"Predictions saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
