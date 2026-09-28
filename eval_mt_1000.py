from pathlib import Path
import numpy as np
import nrrd

pred_dir = Path(r"nnUNet_work/results/Dataset501_SinusMT/nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation")
gt_dir = Path(r"MT/data/segment_valid")

rows = []

for pred_path in sorted(pred_dir.glob("*.nrrd")):
    name = pred_path.stem
    gt_path = gt_dir / f"{name}_GT.nrrd"

    if not gt_path.exists():
        print(f"MISSING GT: {gt_path}")
        continue

    pred, _ = nrrd.read(str(pred_path))
    gt, _ = nrrd.read(str(gt_path))

    pred_mt = pred == 1
    gt_mt = gt == 1

    intersection = np.logical_and(pred_mt, gt_mt).sum()
    pred_n = pred_mt.sum()
    gt_n = gt_mt.sum()

    dice = 2 * intersection / (pred_n + gt_n) if (pred_n + gt_n) else 1.0
    precision = intersection / pred_n if pred_n else 0.0
    recall = intersection / gt_n if gt_n else 0.0

    rows.append((name, dice, precision, recall))

for name, dice, precision, recall in rows:
    print(f"{name:15s} Dice={dice:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

dices = np.array([r[1] for r in rows])

print()
print(f"N = {len(dices)}")
print(f"Mean MT Dice   = {dices.mean():.4f}")
print(f"Median MT Dice = {np.median(dices):.4f}")
print(f"Min MT Dice    = {dices.min():.4f}")
print(f"Max MT Dice    = {dices.max():.4f}")
