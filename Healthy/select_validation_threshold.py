from pathlib import Path

import nrrd
import numpy as np
import torch

from config import device, model
from dataset import TRAINING_MEAN, TRAINING_STD


VALID_SCAN_DIRECTORY = Path("data/scan_valid")
VALID_MASK_DIRECTORY = Path("data/segment_valid")

CHECKPOINT_PATH = Path(
    "checkpoints_fixed_training_normalization/model_50.pth"
)

MASK_SUFFIX = "_GT"

THRESHOLDS = [
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]

def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    prediction = prediction.bool()
    target = target.bool()

    true_positive = torch.logical_and(
        prediction,
        target,
    ).sum().item()

    false_positive = torch.logical_and(
        prediction,
        ~target,
    ).sum().item()

    false_negative = torch.logical_and(
        ~prediction,
        target,
    ).sum().item()

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

    return dice, precision, recall


def load_scan(scan_path: Path):
    scan_array, _ = nrrd.read(str(scan_path))

    scan_array = np.ascontiguousarray(
        scan_array,
        dtype=np.float32,
    )

    scan_tensor = torch.from_numpy(scan_array)

    scan_tensor = (
        scan_tensor - TRAINING_MEAN
    ) / TRAINING_STD

    scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)

    return scan_tensor


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    scan_paths = sorted(
        VALID_SCAN_DIRECTORY.glob("*.nrrd")
    )

    print(f"Using checkpoint: {CHECKPOINT_PATH}")
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
                    f"Missing mask for {scan_id}: {mask_path}"
                )

            scan_tensor = load_scan(scan_path).to(device)

            mask_array, _ = nrrd.read(str(mask_path))

            target = torch.from_numpy(
                np.ascontiguousarray(mask_array > 0)
            )

            logits = model(scan_tensor)

            probabilities = (
                torch.sigmoid(logits)
                .squeeze(0)
                .squeeze(0)
                .cpu()
            )

            probabilities_by_scan.append(
                (
                    scan_id,
                    probabilities,
                    target,
                )
            )

    best_threshold = None
    best_average_dice = -1.0

    print(
        "Threshold | Avg Dice | Avg Precision | Avg Recall"
    )
    print("-" * 52)

    for threshold in THRESHOLDS:
        dice_scores = []
        precision_scores = []
        recall_scores = []

        for _, probabilities, target in probabilities_by_scan:
            prediction = probabilities >= threshold

            dice, precision, recall = calculate_metrics(
                prediction,
                target,
            )

            dice_scores.append(dice)
            precision_scores.append(precision)
            recall_scores.append(recall)

        average_dice = sum(dice_scores) / len(dice_scores)
        average_precision = (
            sum(precision_scores) / len(precision_scores)
        )
        average_recall = (
            sum(recall_scores) / len(recall_scores)
        )

        print(
            f"{threshold:9.2f} | "
            f"{average_dice:8.4f} | "
            f"{average_precision:13.4f} | "
            f"{average_recall:10.4f}"
        )

        if average_dice > best_average_dice:
            best_average_dice = average_dice
            best_threshold = threshold

    print()
    print("=" * 45)
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(f"Best average validation Dice: {best_average_dice:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()