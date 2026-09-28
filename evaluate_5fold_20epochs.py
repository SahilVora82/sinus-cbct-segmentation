from pathlib import Path
import json
import csv
import numpy as np
import nrrd

ROOT = Path(__file__).resolve().parent

SPLIT_PATH = (
    ROOT / "nnUNet_work" / "preprocessed" / "Dataset501_SinusMT" / "splits_final.json"
)

RESULT_ROOT = (
    ROOT / "nnUNet_work" / "results" / "Dataset501_SinusMT"
    / "nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres"
)

GT_DIR = ROOT / "MT" / "data" / "segment_valid"
TRAIN_GT_DIR = ROOT / "MT" / "data" / "segment"
OUT_CSV = ROOT / "mt_5fold_20epoch_oof_metrics.csv"

MT_LABEL = 1


def find_gt(case):
    candidates = [
        GT_DIR / f"{case}_GT.nrrd",
        TRAIN_GT_DIR / f"{case}_GT.nrrd",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find GT for {case}")


def metrics(gt_mt, pred_mt):
    tp = np.logical_and(gt_mt, pred_mt).sum()
    fp = np.logical_and(~gt_mt, pred_mt).sum()
    fn = np.logical_and(gt_mt, ~pred_mt).sum()

    denom = gt_mt.sum() + pred_mt.sum()
    dice = 2 * tp / denom if denom else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(dice), float(precision), float(recall)


with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

if len(splits) != 5:
    raise RuntimeError(f"Expected 5 folds, found {len(splits)}")

rows = []

for fold in range(5):
    pred_dir = RESULT_ROOT / f"fold_{fold}" / "validation"

    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Missing fold {fold} validation folder:\n{pred_dir}"
        )

    expected_cases = splits[fold]["val"]

    print(f"\n===== FOLD {fold} =====")
    fold_dices = []

    for case in expected_cases:
        pred_path = pred_dir / f"{case}.nrrd"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction for fold {fold}: {pred_path}"
            )

        gt_path = find_gt(case)

        pred, _ = nrrd.read(str(pred_path))
        gt, _ = nrrd.read(str(gt_path))

        pred_mt = pred == MT_LABEL
        gt_mt = gt == MT_LABEL

        dice, precision, recall = metrics(gt_mt, pred_mt)

        rows.append({
            "fold": fold,
            "case": case,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "gt_mt_voxels": int(gt_mt.sum()),
            "pred_mt_voxels": int(pred_mt.sum()),
        })

        fold_dices.append(dice)
        print(f"{case:15s} Dice={dice:.4f}  P={precision:.4f}  R={recall:.4f}")

    print(
        f"Fold {fold}: N={len(fold_dices)}, "
        f"Mean Dice={np.mean(fold_dices):.4f}, "
        f"Median Dice={np.median(fold_dices):.4f}"
    )

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

dices = np.array([r["dice"] for r in rows], dtype=float)
precisions = np.array([r["precision"] for r in rows], dtype=float)
recalls = np.array([r["recall"] for r in rows], dtype=float)

print("\n========== 5-FOLD OUT-OF-FOLD SUMMARY ==========")
print(f"N cases          = {len(rows)}")
print(f"Mean MT Dice     = {dices.mean():.4f}")
print(f"Median MT Dice   = {np.median(dices):.4f}")
print(f"Min MT Dice      = {dices.min():.4f}")
print(f"Max MT Dice      = {dices.max():.4f}")
print(f"Mean precision   = {precisions.mean():.4f}")
print(f"Mean recall      = {recalls.mean():.4f}")
print(f"\nSaved CSV:\n{OUT_CSV}")
