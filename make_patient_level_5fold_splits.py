from pathlib import Path
import json
import random
import re
import shutil
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

SPLIT_PATH = (
    ROOT
    / "nnUNet_work"
    / "preprocessed"
    / "Dataset501_SinusMT"
    / "splits_final.json"
)

BACKUP_PATH = SPLIT_PATH.with_name("splits_final_single_split_backup.json")z

OLD_VALIDATION_DIR = (
    ROOT
    / "nnUNet_work"
    / "results"
    / "Dataset501_SinusMT"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)

SEED = 42
N_FOLDS = 5


def patient_id(case_name: str) -> str:
    """
    FileB13_MT_L -> FileB13
    FileB13_MT_R -> FileB13
    FileU36_MT_L -> FileU36

    This keeps left/right scans from the same patient in the same fold.
    """
    m = re.match(r"^(.*)_MT_[LR]$", case_name)
    if m:
        return m.group(1)

    # Fallback for any unexpected naming pattern.
    parts = case_name.split("_")
    if len(parts) >= 2 and parts[-1] in {"L", "R"}:
        return "_".join(parts[:-1])

    return case_name


if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find nnU-Net split file:\n{SPLIT_PATH}"
    )

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    old_splits = json.load(f)

if not isinstance(old_splits, list) or len(old_splits) < 1:
    raise RuntimeError("splits_final.json does not contain at least one split.")

# Preserve the current fold 0 EXACTLY.
fold0_train = list(old_splits[0]["train"])
fold0_val = list(old_splits[0]["val"])

all_cases = sorted(set(fold0_train) | set(fold0_val))

print("Current split:")
print(f"  Fold 0 train: {len(fold0_train)} scans")
print(f"  Fold 0 val:   {len(fold0_val)} scans")
print(f"  Total dev:    {len(all_cases)} scans")

# Confirm the existing trained 1000-epoch fold 0 corresponds to this validation set.
if OLD_VALIDATION_DIR.exists():
    existing_val_predictions = sorted(p.stem for p in OLD_VALIDATION_DIR.glob("*.nrrd"))
    expected_val = sorted(fold0_val)

    if existing_val_predictions != expected_val:
        print("\nExisting fold_0 validation predictions:")
        print(existing_val_predictions)
        print("\nCurrent splits_final.json fold_0 validation:")
        print(expected_val)
        raise RuntimeError(
            "\nSTOP: Existing 1000-epoch fold_0 validation cases do not match "
            "the current fold_0 split. Do not reuse that model until this is resolved."
        )
    else:
        print("  Existing 1000-epoch fold 0 matches the current validation split: PASS")
else:
    print("  WARNING: Existing fold_0 validation directory not found.")

# Check current fold 0 for patient leakage.
train_patients = {patient_id(c) for c in fold0_train}
val_patients = {patient_id(c) for c in fold0_val}
leak = train_patients & val_patients

if leak:
    raise RuntimeError(
        f"STOP: patient leakage already exists in fold 0: {sorted(leak)}"
    )

print(
    f"  Fold 0 patients: {len(train_patients)} train / "
    f"{len(val_patients)} val"
)
print("  Fold 0 patient leakage: PASS")

# Group the ORIGINAL fold-0 training cases by patient.
# Those 48 scans must become validation exactly once across folds 1-4.
groups = defaultdict(list)
for case in fold0_train:
    groups[patient_id(case)].append(case)

group_items = [(pid, sorted(cases)) for pid, cases in groups.items()]

# Shuffle first so equal-size patient groups do not always fall into the same pattern,
# then sort largest patient groups first for balanced fold sizes.
rng = random.Random(SEED)
rng.shuffle(group_items)
group_items.sort(key=lambda x: len(x[1]), reverse=True)

# Four bins because fold 0 is already fixed.
bins = [
    {"patients": [], "cases": []}
    for _ in range(N_FOLDS - 1)
]

for pid, cases in group_items:
    # Put the next patient into the bin with the fewest scans.
    # Tie-break on number of patients, then bin index.
    target = min(
        range(len(bins)),
        key=lambda i: (
            len(bins[i]["cases"]),
            len(bins[i]["patients"]),
            i,
        ),
    )
    bins[target]["patients"].append(pid)
    bins[target]["cases"].extend(cases)

new_splits = [
    {
        "train": sorted(fold0_train),
        "val": sorted(fold0_val),
    }
]

for b in bins:
    val_cases = sorted(b["cases"])
    val_set = set(val_cases)
    train_cases = sorted(c for c in all_cases if c not in val_set)

    new_splits.append(
        {
            "train": train_cases,
            "val": val_cases,
        }
    )

# ---- Validation checks ----

# 1. Every fold must have zero patient leakage.
for i, split in enumerate(new_splits):
    tr_p = {patient_id(c) for c in split["train"]}
    va_p = {patient_id(c) for c in split["val"]}
    overlap = tr_p & va_p

    if overlap:
        raise RuntimeError(
            f"Patient leakage in fold {i}: {sorted(overlap)}"
        )

# 2. Every development scan must be validation exactly once across the five folds.
val_occurrences = defaultdict(int)
for split in new_splits:
    for c in split["val"]:
        val_occurrences[c] += 1

missing = [c for c in all_cases if val_occurrences[c] == 0]
duplicated = [c for c in all_cases if val_occurrences[c] != 1]

if missing or duplicated:
    raise RuntimeError(
        f"Invalid 5-fold partition.\nMissing: {missing}\n"
        f"Not exactly once: {duplicated}"
    )

# 3. Each fold's train + val must equal all 57 dev cases.
for i, split in enumerate(new_splits):
    combined = set(split["train"]) | set(split["val"])
    if combined != set(all_cases):
        raise RuntimeError(f"Fold {i} does not contain all development cases.")

# Backup the original single split before overwriting.
if not BACKUP_PATH.exists():
    shutil.copy2(SPLIT_PATH, BACKUP_PATH)
    print(f"\nBacked up original split to:\n{BACKUP_PATH}")
else:
    print(f"\nBackup already exists:\n{BACKUP_PATH}")

with open(SPLIT_PATH, "w", encoding="utf-8") as f:
    json.dump(new_splits, f, indent=2)

print("\n========== NEW PATIENT-LEVEL 5-FOLD SPLIT ==========")

for i, split in enumerate(new_splits):
    tr_p = {patient_id(c) for c in split["train"]}
    va_p = {patient_id(c) for c in split["val"]}

    print(
        f"Fold {i}: "
        f"train={len(split['train']):2d} scans / {len(tr_p):2d} patients | "
        f"val={len(split['val']):2d} scans / {len(va_p):2d} patients | "
        f"leakage=PASS"
    )

print("\nValidation coverage:")
print(f"  {len(all_cases)} / {len(all_cases)} development scans appear in validation exactly once.")
print("  The 16 untouched test scans were not used here.")
print("\nIMPORTANT:")
print("  Fold 0 was preserved exactly, so your existing 1000-epoch fold_0 model can be reused.")
print("  You only need to train folds 1, 2, 3, and 4.")
print(f"\nUpdated split file:\n{SPLIT_PATH}")
