from pathlib import Path
import shutil
import random
import re

import numpy as np
import nrrd


SEED = 42

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "MT_input_use_allSeg_padded_updated"

TRAIN_SCAN = BASE_DIR / "data" / "scan"
TRAIN_MASK = BASE_DIR / "data" / "segment"

VAL_SCAN = BASE_DIR / "data" / "scan_valid"
VAL_MASK = BASE_DIR / "data" / "segment_valid"

TEST_SCAN = BASE_DIR / "data" / "scan_test"
TEST_MASK = BASE_DIR / "data" / "segment_test"


def get_patient_id(scan_path: Path):
    stem = scan_path.stem

    # Remove left/right suffix so bilateral sinuses stay together.
    stem = re.sub(r"_(L|R)$", "", stem)

    return stem


def reset_dir(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Find scan/mask pairs
# ---------------------------------------------------------

mask_files = sorted(SOURCE_DIR.glob("*_GT.nrrd"))

pairs = []

for mask_path in mask_files:
    scan_name = mask_path.name.replace("_GT.nrrd", ".nrrd")
    scan_path = SOURCE_DIR / scan_name

    if not scan_path.exists():
        raise FileNotFoundError(f"Missing scan for {mask_path.name}")

    patient_id = get_patient_id(scan_path)

    pairs.append(
        {
            "patient": patient_id,
            "scan": scan_path,
            "mask": mask_path,
        }
    )


patients = sorted(set(p["patient"] for p in pairs))

print(f"\nTotal samples: {len(pairs)}")
print(f"Unique patients: {len(patients)}")


# ---------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------

random.seed(SEED)
random.shuffle(patients)

n_patients = len(patients)

n_train = round(n_patients * 0.70)
n_val = round(n_patients * 0.15)

train_patients = set(patients[:n_train])
val_patients = set(patients[n_train:n_train + n_val])
test_patients = set(patients[n_train + n_val:])


print("\nPATIENT SPLIT")
print("-" * 50)
print(f"Train patients: {len(train_patients)}")
print(f"Val patients:   {len(val_patients)}")
print(f"Test patients:  {len(test_patients)}")


# ---------------------------------------------------------
# Reset output folders
# ---------------------------------------------------------

for directory in [
    TRAIN_SCAN,
    TRAIN_MASK,
    VAL_SCAN,
    VAL_MASK,
    TEST_SCAN,
    TEST_MASK,
]:
    reset_dir(directory)


def copy_pair(pair, scan_dir, mask_dir):
    shutil.copy2(pair["scan"], scan_dir / pair["scan"].name)
    shutil.copy2(pair["mask"], mask_dir / pair["mask"].name)


train_pairs = []
val_pairs = []
test_pairs = []

for pair in pairs:

    patient = pair["patient"]

    if patient in train_patients:
        train_pairs.append(pair)
        copy_pair(pair, TRAIN_SCAN, TRAIN_MASK)

    elif patient in val_patients:
        val_pairs.append(pair)
        copy_pair(pair, VAL_SCAN, VAL_MASK)

    elif patient in test_patients:
        test_pairs.append(pair)
        copy_pair(pair, TEST_SCAN, TEST_MASK)

    else:
        raise RuntimeError("Patient not assigned.")


print("\nSAMPLE SPLIT")
print("-" * 50)
print(f"Train samples: {len(train_pairs)}")
print(f"Val samples:   {len(val_pairs)}")
print(f"Test samples:  {len(test_pairs)}")


# ---------------------------------------------------------
# Leakage check
# ---------------------------------------------------------

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print("\nPatient leakage check: PASS")


# ---------------------------------------------------------
# Compute TRAINING-ONLY normalization statistics
#
# Uses every voxel in every training scan.
# Same values must later be applied to train/val/test.
# ---------------------------------------------------------

print("\nCalculating training-set normalization...")

total_sum = 0.0
total_squared_sum = 0.0
total_voxels = 0

for i, pair in enumerate(train_pairs, start=1):

    volume, _ = nrrd.read(str(pair["scan"]))
    volume = volume.astype(np.float64)

    total_sum += volume.sum()
    total_squared_sum += np.square(volume).sum()
    total_voxels += volume.size

    print(
        f"\rProcessing training scan {i}/{len(train_pairs)}",
        end="",
        flush=True,
    )


mean = total_sum / total_voxels

variance = (
    total_squared_sum / total_voxels
    - mean ** 2
)

std = np.sqrt(variance)

print("\n")
print("=" * 60)
print("TRAINING NORMALIZATION")
print("=" * 60)
print(f"Mean: {mean}")
print(f"Std:  {std}")
print("=" * 60)


# ---------------------------------------------------------
# Verify labels
# ---------------------------------------------------------

labels_seen = set()

for pair in pairs:
    mask, _ = nrrd.read(str(pair["mask"]))
    labels_seen.update(np.unique(mask).tolist())

print(f"\nLabels found: {sorted(labels_seen)}")

assert labels_seen == {0, 1, 2}

print("Expected labels confirmed:")
print("  0 = background")
print("  1 = mucosal thickening")
print("  2 = sinus air")

print("\nDATA PREPARATION COMPLETE.")