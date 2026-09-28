from pathlib import Path
import numpy as np
import nrrd
import csv
from scipy.ndimage import label

ROOT = Path(".")
GT_DIR = ROOT / "MT/data/segment_valid"
PRED_DIR = ROOT / "nnUNet_work/results/Dataset501_SinusMT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
OUT_DIR = ROOT / "mt_component_analysis"
OUT_DIR.mkdir(exist_ok=True)

thresholds = [0, 5, 10, 20, 30, 50, 75, 100]

def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / denom if denom else 1.0

pred_files = sorted(PRED_DIR.glob("*.nrrd"))
case_names = [p.stem for p in pred_files]

structure = np.ones((3, 3, 3), dtype=np.uint8)

case_rows = []
threshold_scores = {t: [] for t in thresholds}

for pred_path in pred_files:
    case = pred_path.stem
    gt_path = GT_DIR / f"{case}_GT.nrrd"

    if not gt_path.exists():
        print(f"Skipping {case} (missing GT)")
        continue

    pred, _ = nrrd.read(str(pred_path))
    gt, _ = nrrd.read(str(gt_path))

    pred_mt = pred == 1
    gt_mt = gt == 1

    base_dice = dice(gt_mt, pred_mt)

    labeled, ncomp = label(pred_mt, structure=structure)
    counts = np.bincount(labeled.ravel())[1:]  # ignore label 0
    counts_sorted = sorted(counts.tolist(), reverse=True)

    largest = counts_sorted[0] if len(counts_sorted) > 0 else 0
    second = counts_sorted[1] if len(counts_sorted) > 1 else 0
    third = counts_sorted[2] if len(counts_sorted) > 2 else 0

    row = {
        "case": case,
        "base_dice": base_dice,
        "n_components": int(ncomp),
        "largest_comp": int(largest),
        "second_comp": int(second),
        "third_comp": int(third),
        "pred_voxels": int(pred_mt.sum()),
        "gt_voxels": int(gt_mt.sum()),
    }

    for t in thresholds:
        if t == 0:
            cleaned = pred_mt.copy()
        else:
            keep = np.zeros_like(pred_mt, dtype=bool)
            for comp_id, size in enumerate(counts, start=1):
                if size >= t:
                    keep |= (labeled == comp_id)
            cleaned = keep

        d = dice(gt_mt, cleaned)
        row[f"dice_keep_ge_{t}"] = d
        threshold_scores[t].append(d)

    case_rows.append(row)

# Save per-case CSV
case_csv = OUT_DIR / "component_analysis_per_case.csv"
fieldnames = list(case_rows[0].keys()) if case_rows else []
with open(case_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(case_rows)

# Save threshold summary CSV
summary_csv = OUT_DIR / "component_analysis_threshold_summary.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold_keep_ge_voxels", "mean_dice", "median_dice"])
    for t in thresholds:
        vals = np.array(threshold_scores[t], dtype=float)
        writer.writerow([t, vals.mean(), np.median(vals)])

print("===== PER CASE =====")
for row in case_rows:
    print(
        f"{row['case']:15s} "
        f"BaseDice={row['base_dice']:.4f}  "
        f"nComp={row['n_components']:2d}  "
        f"Largest={row['largest_comp']:4d}  "
        f"Second={row['second_comp']:4d}  "
        f"Third={row['third_comp']:4d}"
    )

print()
print("===== THRESHOLD SUMMARY =====")
best_t = None
best_mean = -1
for t in thresholds:
    vals = np.array(threshold_scores[t], dtype=float)
    mean_d = vals.mean()
    med_d = np.median(vals)
    print(f"Keep components >= {t:3d} voxels: Mean Dice={mean_d:.4f}  Median Dice={med_d:.4f}")
    if mean_d > best_mean:
        best_mean = mean_d
        best_t = t

print()
print(f"Best mean Dice threshold: keep components >= {best_t} voxels")
print(f"Best mean Dice: {best_mean:.4f}")
print()
print(f"Saved: {case_csv}")
print(f"Saved: {summary_csv}")
