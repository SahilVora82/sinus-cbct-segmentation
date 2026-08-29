from pathlib import Path
import json
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

TRAIN_SCAN = BASE / "data" / "scan"
TRAIN_MASK = BASE / "data" / "segment"

VAL_SCAN = BASE / "data" / "scan_valid"
VAL_MASK = BASE / "data" / "segment_valid"

WORK = ROOT / "nnUNet_work"

RAW = WORK / "raw"
PREPROCESSED = WORK / "preprocessed"
RESULTS = WORK / "results"

DATASET_NAME = "Dataset501_SinusMT"
DATASET_DIR = RAW / DATASET_NAME

IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

for folder in [
    RAW,
    PREPROCESSED,
    RESULTS,
    IMAGES_TR,
    LABELS_TR,
]:
    folder.mkdir(parents=True, exist_ok=True)


train_ids = sorted(
    p.stem
    for p in TRAIN_SCAN.glob("*.nrrd")
)

val_ids = sorted(
    p.stem
    for p in VAL_SCAN.glob("*.nrrd")
)

print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test included: NO")


def add_case(case_id, scan_dir, mask_dir):

    source_scan = (
        scan_dir
        / f"{case_id}.nrrd"
    )

    source_mask = (
        mask_dir
        / f"{case_id}_GT.nrrd"
    )

    destination_scan = (
        IMAGES_TR
        / f"{case_id}_0000.nrrd"
    )

    destination_mask = (
        LABELS_TR
        / f"{case_id}.nrrd"
    )

    if not source_scan.exists():
        raise FileNotFoundError(
            source_scan
        )

    if not source_mask.exists():
        raise FileNotFoundError(
            source_mask
        )

    shutil.copy2(
        source_scan,
        destination_scan,
    )

    shutil.copy2(
        source_mask,
        destination_mask,
    )


for case_id in train_ids:
    add_case(
        case_id,
        TRAIN_SCAN,
        TRAIN_MASK,
    )

for case_id in val_ids:
    add_case(
        case_id,
        VAL_SCAN,
        VAL_MASK,
    )


dataset_json = {

    # IMPORTANT:
    # We call this CBCT rather than CT.
    # nnU-Net will therefore use z-score
    # normalization instead of assuming
    # calibrated conventional CT HU values.
    "channel_names": {
        "0": "CBCT"
    },

    "labels": {
        "background": 0,
        "MT": 1,
        "air": 2,
    },

    "numTraining":
        len(train_ids)
        + len(val_ids),

    "file_ending":
        ".nrrd",

    "overwrite_image_reader_writer":
        "SimpleITKIO",
}


with open(
    DATASET_DIR / "dataset.json",
    "w",
) as f:

    json.dump(
        dataset_json,
        f,
        indent=2,
    )


# EXACT SAME SPLIT AS OUR MODEL.
splits = [
    {
        "train": train_ids,
        "val": val_ids,
    }
]


with open(
    WORK / "splits_final.json",
    "w",
) as f:

    json.dump(
        splits,
        f,
        indent=2,
    )


print()
print("NNUNET DATASET READY")
print("Dataset:", DATASET_DIR)
print("Training cases:", len(train_ids))
print("Validation cases:", len(val_ids))
print("Total:", len(train_ids) + len(val_ids))
print("Test cases used: 0")
print()
print("Split file:")
print(WORK / "splits_final.json")