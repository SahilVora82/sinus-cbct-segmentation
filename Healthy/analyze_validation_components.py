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
    "validation_connected_components_threshold_050.csv"
)

MASK_SUFFIX = "_GT"
THRESHOLD = 0.50

# 26-connectivity:
# Voxels touching by a face, edge, or corner count as connected.
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


def get_component_sizes(binary_mask: np.ndarray):
    labeled_mask, number_of_components = label(
        binary_mask,
        structure=CONNECTIVITY_STRUCTURE,
    )

    if number_of_components == 0:
        return []

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )[1:]

    component_sizes = sorted(
        component_sizes.tolist(),
        reverse=True,
    )

    return component_sizes


def print_component_report(
    scan_id: str,
    mask_type: str,
    component_sizes,
):
    total_positive_voxels = sum(component_sizes)

    print(f"  {mask_type}")
    print(
        f"    Number of components: "
        f"{len(component_sizes)}"
    )
    print(
        f"    Total positive voxels: "
        f"{total_positive_voxels}"
    )

    if len(component_sizes) == 0:
        print("    No positive components found.")
        return

    for rank, size in enumerate(
        component_sizes[:10],
        start=1,
    ):
        percentage = (
            100.0 * size / total_positive_voxels
        )

        print(
            f"    Component {rank}: "
            f"{size} voxels "
            f"({percentage:.2f}% of positive mask)"
        )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: "
            f"{CHECKPOINT_PATH}"
        )

    scan_paths = sorted(
        VALID_SCAN_DIRECTORY.glob("*.nrrd")
    )

    if len(scan_paths) == 0:
        raise RuntimeError(
            f"No validation scans were found in "
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
    print(
        f"Found {len(scan_paths)} validation scans."
    )
    print()

    csv_rows = []

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

            prediction = (
                probabilities >= THRESHOLD
            ).squeeze(0).squeeze(0).cpu().numpy()

            prediction = np.ascontiguousarray(
                prediction.astype(np.uint8)
            )

            ground_truth, _ = nrrd.read(
                str(mask_path)
            )

            ground_truth = np.ascontiguousarray(
                (ground_truth > 0).astype(np.uint8)
            )

            prediction_component_sizes = (
                get_component_sizes(prediction)
            )

            ground_truth_component_sizes = (
                get_component_sizes(ground_truth)
            )

            print("=" * 60)
            print(scan_id)

            print_component_report(
                scan_id,
                "PREDICTION",
                prediction_component_sizes,
            )

            print_component_report(
                scan_id,
                "GROUND TRUTH",
                ground_truth_component_sizes,
            )

            print()

            for mask_type, sizes in [
                (
                    "prediction",
                    prediction_component_sizes,
                ),
                (
                    "ground_truth",
                    ground_truth_component_sizes,
                ),
            ]:
                total_positive_voxels = sum(sizes)

                if len(sizes) == 0:
                    csv_rows.append(
                        {
                            "scan": scan_id,
                            "mask_type": mask_type,
                            "component_rank": 0,
                            "component_size": 0,
                            "percentage_of_positive_mask": 0.0,
                            "number_of_components": 0,
                            "total_positive_voxels": 0,
                        }
                    )

                for rank, size in enumerate(
                    sizes,
                    start=1,
                ):
                    percentage = (
                        100.0
                        * size
                        / total_positive_voxels
                    )

                    csv_rows.append(
                        {
                            "scan": scan_id,
                            "mask_type": mask_type,
                            "component_rank": rank,
                            "component_size": size,
                            "percentage_of_positive_mask": percentage,
                            "number_of_components": len(sizes),
                            "total_positive_voxels": total_positive_voxels,
                        }
                    )

    fieldnames = [
        "scan",
        "mask_type",
        "component_rank",
        "component_size",
        "percentage_of_positive_mask",
        "number_of_components",
        "total_positive_voxels",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 60)
    print(f"Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()

