from pathlib import Path
import csv
import numpy as np
import nrrd
from scipy.ndimage import binary_erosion, distance_transform_edt

# Put this file in the repository root:
# sinus-cbct-segmentation/
#   evaluate_mt_surface_metrics.py
#   MT/
#   nnUNet_work/

ROOT = Path(__file__).resolve().parent

GT_DIR = ROOT / "MT" / "data" / "segment_valid"
PRED_DIR = (
    ROOT
    / "nnUNet_work"
    / "results"
    / "Dataset501_SinusMT"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)
OUT_CSV = ROOT / "mt_1000_surface_metrics.csv"

MT_LABEL = 1


def get_spacing_mm(header):
    """
    Try to read voxel spacing from the NRRD header.
    Falls back to 1 mm isotropic if unavailable.
    """
    if "space directions" in header:
        dirs = np.asarray(header["space directions"], dtype=float)
        if dirs.ndim == 2 and dirs.shape[0] >= 3:
            spacing = []
            for v in dirs[:3]:
                if np.all(np.isfinite(v)):
                    spacing.append(float(np.linalg.norm(v)))
                else:
                    spacing.append(1.0)
            return tuple(spacing)

    if "spacings" in header:
        vals = np.asarray(header["spacings"], dtype=float).ravel()
        if len(vals) >= 3 and np.all(np.isfinite(vals[:3])):
            return tuple(float(x) for x in vals[:3])

    print("WARNING: Could not find spacing in NRRD header; assuming 1x1x1 mm.")
    return (1.0, 1.0, 1.0)


def dice_score(gt, pred):
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(gt, pred).sum() / denom


def surface(mask):
    if not mask.any():
        return mask.copy()
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances_mm(gt, pred, spacing):
    """
    Returns directed surface distances:
      gt_to_pred and pred_to_gt
    in millimeters.
    """
    gt_s = surface(gt)
    pred_s = surface(pred)

    if not gt_s.any() or not pred_s.any():
        return np.array([]), np.array([])

    # EDT measures distance to nearest zero voxel.
    # ~surface => the surface itself is zero.
    dist_to_pred = distance_transform_edt(~pred_s, sampling=spacing)
    dist_to_gt = distance_transform_edt(~gt_s, sampling=spacing)

    gt_to_pred = dist_to_pred[gt_s]
    pred_to_gt = dist_to_gt[pred_s]
    return gt_to_pred, pred_to_gt


def surface_dice(gt_to_pred, pred_to_gt, tolerance_mm):
    if len(gt_to_pred) == 0 or len(pred_to_gt) == 0:
        return np.nan

    close_gt = np.count_nonzero(gt_to_pred <= tolerance_mm)
    close_pred = np.count_nonzero(pred_to_gt <= tolerance_mm)
    return (close_gt + close_pred) / (len(gt_to_pred) + len(pred_to_gt))


rows = []

pred_files = sorted(PRED_DIR.glob("*.nrrd"))
if not pred_files:
    raise FileNotFoundError(f"No prediction NRRDs found in:\n{PRED_DIR}")

for pred_path in pred_files:
    case = pred_path.stem
    gt_path = GT_DIR / f"{case}_GT.nrrd"

    if not gt_path.exists():
        print(f"SKIP - missing GT: {gt_path}")
        continue

    pred, pred_header = nrrd.read(str(pred_path))
    gt, gt_header = nrrd.read(str(gt_path))

    gt_mt = gt == MT_LABEL
    pred_mt = pred == MT_LABEL

    spacing = get_spacing_mm(gt_header)

    dice = float(dice_score(gt_mt, pred_mt))

    gt_to_pred, pred_to_gt = surface_distances_mm(gt_mt, pred_mt, spacing)

    if len(gt_to_pred) and len(pred_to_gt):
        all_distances = np.concatenate([gt_to_pred, pred_to_gt])
        assd = float(
            (gt_to_pred.mean() + pred_to_gt.mean()) / 2.0
        )
        hd95 = float(np.percentile(all_distances, 95))
        sd1 = float(surface_dice(gt_to_pred, pred_to_gt, 1.0))
        sd2 = float(surface_dice(gt_to_pred, pred_to_gt, 2.0))
    else:
        assd = np.nan
        hd95 = np.nan
        sd1 = np.nan
        sd2 = np.nan

    gt_vox = int(gt_mt.sum())
    pred_vox = int(pred_mt.sum())
    vol_error_pct = (
        100.0 * (pred_vox - gt_vox) / gt_vox
        if gt_vox > 0 else np.nan
    )

    row = {
        "case": case,
        "dice": dice,
        "assd_mm": assd,
        "hd95_mm": hd95,
        "surface_dice_1mm": sd1,
        "surface_dice_2mm": sd2,
        "gt_mt_voxels": gt_vox,
        "pred_mt_voxels": pred_vox,
        "volume_error_pct": vol_error_pct,
        "spacing_mm": "x".join(f"{x:.3f}" for x in spacing),
    }
    rows.append(row)

    print(
        f"{case:15s} "
        f"Dice={dice:.4f}  "
        f"ASSD={assd:.3f} mm  "
        f"HD95={hd95:.3f} mm  "
        f"SD@1mm={sd1:.4f}  "
        f"SD@2mm={sd2:.4f}  "
        f"VolErr={vol_error_pct:+.1f}%"
    )

if not rows:
    raise RuntimeError("No cases were evaluated.")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

def mean(key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(vals))

def median(key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmedian(vals))

print("\n========== SUMMARY ==========")
print(f"N cases                = {len(rows)}")
print(f"Mean Dice              = {mean('dice'):.4f}")
print(f"Median Dice            = {median('dice'):.4f}")
print(f"Mean ASSD               = {mean('assd_mm'):.3f} mm")
print(f"Median ASSD             = {median('assd_mm'):.3f} mm")
print(f"Mean HD95               = {mean('hd95_mm'):.3f} mm")
print(f"Median HD95             = {median('hd95_mm'):.3f} mm")
print(f"Mean Surface Dice @1mm  = {mean('surface_dice_1mm'):.4f}")
print(f"Mean Surface Dice @2mm  = {mean('surface_dice_2mm'):.4f}")
print(f"Mean volume error       = {mean('volume_error_pct'):+.1f}%")
print(f"Median volume error     = {median('volume_error_pct'):+.1f}%")
print(f"\nSaved CSV:\n{OUT_CSV}")
