import csv
import random
import re
import shutil
from pathlib import Path


SOURCE_DIRECTORY = Path("data_source")
OUTPUT_DIRECTORY = Path("data")
MANIFEST_FILE = Path("split_manifest.csv")

RANDOM_SEED = 42

TRAIN_PATIENTS = 21
VALID_PATIENTS = 5
TEST_PATIENTS = 4


def get_patient_id(scan_id: str) -> str:
    """
    File10_L -> File10
    File10_R -> File10
    """

    match = re.match(r"^(File\d+)(?:_[LR])?$", scan_id)

    if match is None:
        raise ValueError(
            f"Could not determine patient ID from filename: {scan_id}"
        )

    return match.group(1)


def patient_number(patient_id: str) -> int:
    match = re.search(r"\d+", patient_id)

    if match is None:
        return 0

    return int(match.group())


def create_output_directories():
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"\nThe output folder already exists:\n"
            f"{OUTPUT_DIRECTORY.resolve()}\n\n"
            f"Delete the 'data' folder before running this script again."
        )

    directories = [
        OUTPUT_DIRECTORY / "scan",
        OUTPUT_DIRECTORY / "segment",
        OUTPUT_DIRECTORY / "scan_valid",
        OUTPUT_DIRECTORY / "segment_valid",
        OUTPUT_DIRECTORY / "scan_test",
        OUTPUT_DIRECTORY / "segment_test",
        OUTPUT_DIRECTORY / "inference",
        OUTPUT_DIRECTORY / "inference_output",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)


def main():
    if not SOURCE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Could not find source folder:\n"
            f"{SOURCE_DIRECTORY.resolve()}"
        )

    scan_files = sorted(
        file_path
        for file_path in SOURCE_DIRECTORY.glob("*.nrrd")
        if not file_path.name.endswith("_GT.nrrd")
    )

    if not scan_files:
        raise ValueError("No scan files were found in data_source.")

    patient_cases = {}

    for scan_path in scan_files:
        scan_id = scan_path.stem
        patient_id = get_patient_id(scan_id)

        mask_path = SOURCE_DIRECTORY / f"{scan_id}_GT.nrrd"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for scan: {scan_path.name}"
            )

        patient_cases.setdefault(patient_id, []).append(
            {
                "scan_id": scan_id,
                "scan_path": scan_path,
                "mask_path": mask_path,
            }
        )

    patient_ids = sorted(
        patient_cases.keys(),
        key=patient_number
    )

    if len(patient_ids) != 30:
        raise ValueError(
            f"Expected 30 patients, but found {len(patient_ids)}."
        )

    if (
        TRAIN_PATIENTS
        + VALID_PATIENTS
        + TEST_PATIENTS
        != len(patient_ids)
    ):
        raise ValueError(
            "Train, validation, and test patient counts "
            "do not equal the total number of patients."
        )

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(patient_ids)

    train_ids = patient_ids[:TRAIN_PATIENTS]

    valid_start = TRAIN_PATIENTS
    valid_end = TRAIN_PATIENTS + VALID_PATIENTS
    valid_ids = patient_ids[valid_start:valid_end]

    test_ids = patient_ids[valid_end:]

    split_assignments = {}

    for patient_id in train_ids:
        split_assignments[patient_id] = "train"

    for patient_id in valid_ids:
        split_assignments[patient_id] = "validation"

    for patient_id in test_ids:
        split_assignments[patient_id] = "test"

    create_output_directories()

    output_locations = {
        "train": {
            "scan": OUTPUT_DIRECTORY / "scan",
            "segment": OUTPUT_DIRECTORY / "segment",
        },
        "validation": {
            "scan": OUTPUT_DIRECTORY / "scan_valid",
            "segment": OUTPUT_DIRECTORY / "segment_valid",
        },
        "test": {
            "scan": OUTPUT_DIRECTORY / "scan_test",
            "segment": OUTPUT_DIRECTORY / "segment_test",
        },
    }

    manifest_rows = []
    scan_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for patient_id in sorted(
        patient_cases.keys(),
        key=patient_number
    ):
        split_name = split_assignments[patient_id]
        destinations = output_locations[split_name]

        for case in patient_cases[patient_id]:
            scan_path = case["scan_path"]
            mask_path = case["mask_path"]

            scan_destination = destinations["scan"] / scan_path.name
            mask_destination = destinations["segment"] / mask_path.name

            shutil.copy2(scan_path, scan_destination)
            shutil.copy2(mask_path, mask_destination)

            scan_counts[split_name] += 1

            manifest_rows.append(
                {
                    "patient_id": patient_id,
                    "scan_id": case["scan_id"],
                    "split": split_name,
                    "scan_file": scan_path.name,
                    "mask_file": mask_path.name,
                }
            )

    with open(MANIFEST_FILE, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "patient_id",
                "scan_id",
                "split",
                "scan_file",
                "mask_file",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDataset split completed successfully.\n")

    print("TRAINING")
    print(f"  Patients: {len(train_ids)}")
    print(f"  Scans:    {scan_counts['train']}")
    print(f"  IDs:      {sorted(train_ids, key=patient_number)}")

    print("\nVALIDATION")
    print(f"  Patients: {len(valid_ids)}")
    print(f"  Scans:    {scan_counts['validation']}")
    print(f"  IDs:      {sorted(valid_ids, key=patient_number)}")

    print("\nTEST")
    print(f"  Patients: {len(test_ids)}")
    print(f"  Scans:    {scan_counts['test']}")
    print(f"  IDs:      {sorted(test_ids, key=patient_number)}")

    print(f"\nTotal patients: {len(patient_ids)}")
    print(f"Total scans:    {sum(scan_counts.values())}")
    print(f"Manifest:       {MANIFEST_FILE.resolve()}")
    print(f"Output folder:  {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()