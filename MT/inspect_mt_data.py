from pathlib import Path
from collections import defaultdict

import numpy as np
import nrrd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "MT_input_use_allSeg_padded_updated"

GROUND_TRUTH_SUFFIX = "_GT"


def patient_id_from_scan_id(scan_id: str) -> str:
    """
    Example:
        FileB1_MT_L -> FileB1
        FileB1_MT_R -> FileB1
    """
    parts = scan_id.split("_")

    if len(parts) >= 1:
        return parts[0]

    return scan_id


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find the MT source folder:\n{SOURCE_DIR}"
        )

    all_files = sorted(SOURCE_DIR.glob("*.nrrd"))

    scan_files = [
        file
        for file in all_files
        if not file.stem.endswith(GROUND_TRUTH_SUFFIX)
    ]

    mask_files = [
        file
        for file in all_files
        if file.stem.endswith(GROUND_TRUTH_SUFFIX)
    ]

    print("=" * 70)
    print("MT DATASET INSPECTION")
    print("=" * 70)
    print(f"Source folder: {SOURCE_DIR}")
    print(f"Total NRRD files: {len(all_files)}")
    print(f"Scan files: {len(scan_files)}")
    print(f"Ground-truth masks: {len(mask_files)}")
    print()

    patient_to_scans = defaultdict(list)
    missing_masks = []
    shape_mismatches = []
    all_mask_values = set()

    for scan_path in scan_files:
        scan_id = scan_path.stem
        mask_path = SOURCE_DIR / f"{scan_id}{GROUND_TRUTH_SUFFIX}.nrrd"

        patient_id = patient_id_from_scan_id(scan_id)
        patient_to_scans[patient_id].append(scan_id)

        if not mask_path.exists():
            missing_masks.append(scan_id)
            continue

        scan, _ = nrrd.read(str(scan_path))
        mask, _ = nrrd.read(str(mask_path))

        unique_values, counts = np.unique(
            mask,
            return_counts=True,
        )

        all_mask_values.update(unique_values.tolist())

        if scan.shape != mask.shape:
            shape_mismatches.append(
                (
                    scan_id,
                    scan.shape,
                    mask.shape,
                )
            )

        value_summary = ", ".join(
            f"{value}: {count:,} voxels"
            for value, count in zip(unique_values, counts)
        )

        print(f"Sample: {scan_id}")
        print(f"  Scan shape: {scan.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask values: {value_summary}")
        print(
            f"  Scan range: "
            f"{float(np.min(scan)):.2f} to "
            f"{float(np.max(scan)):.2f}"
        )
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Unique patients: {len(patient_to_scans)}")
    print(f"All mask values found: {sorted(all_mask_values)}")
    print()

    print("Samples grouped by patient:")
    for patient_id in sorted(patient_to_scans):
        samples = ", ".join(
            sorted(patient_to_scans[patient_id])
        )
        print(f"  {patient_id}: {samples}")

    print()

    if missing_masks:
        print("Scans missing a ground-truth mask:")
        for scan_id in missing_masks:
            print(f"  {scan_id}")
    else:
        print("Every scan has a matching ground-truth mask.")

    print()

    if shape_mismatches:
        print("Scan-mask shape mismatches:")
        for scan_id, scan_shape, mask_shape in shape_mismatches:
            print(
                f"  {scan_id}: "
                f"scan {scan_shape}, mask {mask_shape}"
            )
    else:
        print("All scan-mask pairs have matching shapes.")


if __name__ == "__main__":
    main()