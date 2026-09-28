from pathlib import Path
import json
import csv
import numpy as np
import nrrd

ROOT = Path(__file__).resolve().parent

SPLIT_PATH = ROOT / "nnUNet_work" / "preprocessed" / "Dataset501_SinusMT" / "splits_final.json"
RESULT_ROOT = (
    ROOT / "nnUNet_work" / "results" / "Dataset501_SinusMT"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)

# Original project GT locations
GT_DIRS = [
    ROOT / "MT" / "data" / "segment",
    ROOT / "MT" / "data" / "segment_valid",
]

OUT_CSV = ROOT / "mt_5fold_1000epoch_oof_metrics.csv"
MT_LABEL = 1


def find_gt(case):
    for d in GT_DIRS:
        p = d / f"{case}_GT.nrrd"
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find GT for {case}")


def calc_metrics(gt_mt, pred_mt):
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
        raise FileNotFoundError(f"Missing validation folder:\n{pred_dir}")

    fold_dices = []
    print(f"\n===== FOLD {fold} =====")

    for case in splits[fold]["val"]:
        pred_path = pred_dir / f"{case}.nrrd"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction:\n{pred_path}")

        gt_path = find_gt(case)

        pred, _ = nrrd.read(str(pred_path))
        gt, _ = nrrd.read(str(gt_path))

        pred_mt = pred == MT_LABEL
        gt_mt = gt == MT_LABEL

        d, p, r = calc_metrics(gt_mt, pred_mt)
        fold_dices.append(d)

        rows.append({
            "fold": fold,
            "case": case,
            "dice": d,
            "precision": p,
            "recall": r,
            "gt_mt_voxels": int(gt_mt.sum()),
            "pred_mt_voxels": int(pred_mt.sum()),
        })

        print(f"{case:15s} Dice={d:.4f}  P={p:.4f}  R={r:.4f}")

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

print("\n========== 5-FOLD 1000-EPOCH OUT-OF-FOLD SUMMARY ==========")
print(f"N cases        = {len(rows)}")
print(f"Mean MT Dice   = {dices.mean():.4f}")
print(f"Median MT Dice = {np.median(dices):.4f}")
print(f"Min MT Dice    = {dices.min():.4f}")
print(f"Max MT Dice    = {dices.max():.4f}")
print(f"Mean precision = {precisions.mean():.4f}")
print(f"Mean recall    = {recalls.mean():.4f}")
print(f"\nSaved:\n{OUT_CSV}")
