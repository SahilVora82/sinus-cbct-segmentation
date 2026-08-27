import csv
import re
from pathlib import Path

import nrrd
import numpy as np


DATA_DIRECTORY = Path("data_source")
REPORT_FILE = Path("inspection_report.csv")


def get_patient_id(scan_id: str) -> str:
    """
    File4_L -> File4
    File4_R -> File4
    File12_L -> File12
    """
    match = re.match(r"^(File\d+)(?:_[LR])?$", scan_id)

    if match is None:
        return scan_id

    return match.group(1)


def safely_get_spacing(header):
    space_directions = header.get("space directions")

    if space_directions is None:
        return "Not found"

    try:
        spacing = []

        for direction in space_directions:
            direction = np.asarray(direction, dtype=float)
            spacing.append(float(np.linalg.norm(direction)))

        return spacing

    except Exception:
        return "Could not read"


def main():
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Could not find data folder: {DATA_DIRECTORY.resolve()}"
        )

    all_nrrd_files = sorted(DATA_DIRECTORY.glob("*.nrrd"))

    scan_files = [
        file_path
        for file_path in all_nrrd_files
        if not file_path.name.endswith("_GT.nrrd")
    ]

    mask_files = {
        file_path.name: file_path
        for file_path in all_nrrd_files
        if file_path.name.endswith("_GT.nrrd")
    }

    print(f"Total NRRD files: {len(all_nrrd_files)}")
    print(f"Scan files: {len(scan_files)}")
    print(f"Mask files: {len(mask_files)}")
    print()

    rows = []
    missing_masks = []
    invalid_cases = []
    patient_ids = set()

    for scan_path in scan_files:
        scan_id = scan_path.stem
        patient_id = get_patient_id(scan_id)
        patient_ids.add(patient_id)

        expected_mask_name = f"{scan_id}_GT.nrrd"
        mask_path = mask_files.get(expected_mask_name)

        if mask_path is None:
            print(f"MISSING MASK: {scan_path.name}")
            missing_masks.append(scan_path.name)
            continue

        try:
            scan, scan_header = nrrd.read(str(scan_path))
            mask, mask_header = nrrd.read(str(mask_path))

            unique_mask_values = np.unique(mask)
            positive_voxels = int(np.count_nonzero(mask > 0))

            shapes_match = scan.shape == mask.shape
            mask_empty = positive_voxels == 0
            divisible_by_8 = all(
                dimension % 8 == 0
                for dimension in scan.shape
            )

            scan_spacing = safely_get_spacing(scan_header)
            mask_spacing = safely_get_spacing(mask_header)

            spacing_matches = str(scan_spacing) == str(mask_spacing)

            problems = []

            if not shapes_match:
                problems.append("shape mismatch")

            if mask_empty:
                problems.append("empty mask")

            if not divisible_by_8:
                problems.append("shape not divisible by 8")

            if not spacing_matches:
                problems.append("spacing mismatch")

            if len(unique_mask_values) > 2:
                problems.append("more than two mask values")

            status = "OK" if not problems else "; ".join(problems)

            print(
                f"{scan_id}: "
                f"shape={scan.shape}, "
                f"mask_values={unique_mask_values.tolist()}, "
                f"positive_voxels={positive_voxels}, "
                f"status={status}"
            )

            if problems:
                invalid_cases.append((scan_id, status))

            rows.append({
                "patient_id": patient_id,
                "scan_id": scan_id,
                "scan_file": scan_path.name,
                "mask_file": mask_path.name,
                "scan_shape": str(scan.shape),
                "mask_shape": str(mask.shape),
                "shapes_match": shapes_match,
                "mask_values": str(unique_mask_values.tolist()),
                "positive_voxels": positive_voxels,
                "mask_empty": mask_empty,
                "divisible_by_8": divisible_by_8,
                "scan_spacing": str(scan_spacing),
                "mask_spacing": str(mask_spacing),
                "spacing_matches": spacing_matches,
                "status": status
            })

        except Exception as error:
            print(f"ERROR READING {scan_id}: {error}")
            invalid_cases.append((scan_id, str(error)))

    with open(REPORT_FILE, "w", newline="") as csv_file:
        fieldnames = [
            "patient_id",
            "scan_id",
            "scan_file",
            "mask_file",
            "scan_shape",
            "mask_shape",
            "shapes_match",
            "mask_values",
            "positive_voxels",
            "mask_empty",
            "divisible_by_8",
            "scan_spacing",
            "mask_spacing",
            "spacing_matches",
            "status"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    paired_mask_names = {
        f"{scan_path.stem}_GT.nrrd"
        for scan_path in scan_files
    }

    extra_masks = sorted(
        mask_name
        for mask_name in mask_files
        if mask_name not in paired_mask_names
    )

    print()
    print("--------------- SUMMARY ---------------")
    print(f"Unique patients: {len(patient_ids)}")
    print(f"Valid scan-mask pairs checked: {len(rows)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Extra masks without scans: {len(extra_masks)}")
    print(f"Cases with warnings/errors: {len(invalid_cases)}")
    print(f"Report saved to: {REPORT_FILE.resolve()}")

    if missing_masks:
        print("\nScans missing masks:")
        for file_name in missing_masks:
            print(f"  {file_name}")

    if extra_masks:
        print("\nMasks missing scans:")
        for file_name in extra_masks:
            print(f"  {file_name}")

    if invalid_cases:
        print("\nCases requiring attention:")
        for scan_id, problem in invalid_cases:
            print(f"  {scan_id}: {problem}")


if __name__ == "__main__":
    main()