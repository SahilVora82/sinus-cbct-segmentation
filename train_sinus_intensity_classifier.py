#!/usr/bin/env python
"""
train_sinus_intensity_classifier.py

Fast, interpretable healthy-vs-MT classifier using intensity statistics inside
the sinus ROI instead of a deep 3D CNN.

Labels:
  0 = healthy
  1 = mucosal thickening (MT)

IMPORTANT:
- Uses patient-level StratifiedGroupKFold.
- Uses only intensity-distribution features for classification.
- ROI volume / crop size are saved for sanity checking but NOT used as model features,
  to reduce the risk of learning mask-definition differences.
- For MT, labels > 0 are used as the sinus ROI.
- Do NOT point this script at the locked MT test set.

Expected:
Healthy image: File10_L.nrrd
Healthy mask:  File10_L_GT.nrrd
MT image:      FileB11_MT_L_0000.nrrd
MT label:      FileB11_MT_L.nrrd
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import nrrd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# IDs / pairing
# ---------------------------------------------------------------------
def healthy_patient_id(path: Path) -> str:
    stem = re.sub(r'_[LR]$', '', path.stem)
    return "H_" + stem


def mt_patient_id(path: Path) -> str:
    stem = re.sub(r'_0000$', '', path.stem)
    stem = re.sub(r'_MT_[LR]$', '', stem)
    return "M_" + stem


def discover_samples(healthy_dir, healthy_mask_dir, mt_dir, mt_label_dir):
    healthy_dir = Path(healthy_dir)
    healthy_mask_dir = Path(healthy_mask_dir)
    mt_dir = Path(mt_dir)
    mt_label_dir = Path(mt_label_dir)

    for d, name in [
        (healthy_dir, "healthy_dir"),
        (healthy_mask_dir, "healthy_mask_dir"),
        (mt_dir, "mt_dir"),
        (mt_label_dir, "mt_label_dir"),
    ]:
        if not d.exists():
            raise SystemExit(f"{name} does not exist: {d.resolve()}")

    healthy_imgs = sorted(
        p for p in healthy_dir.glob("*.nrrd")
        if "_GT" not in p.name
    )
    mt_imgs = sorted(mt_dir.glob("*_0000.nrrd"))

    samples = []
    missing = []

    for img in healthy_imgs:
        mask = healthy_mask_dir / f"{img.stem}_GT.nrrd"
        if not mask.exists():
            missing.append(str(mask))
            continue
        samples.append({
            "path": img.resolve(),
            "mask_path": mask.resolve(),
            "label": 0,
            "class_name": "healthy",
            "patient_id": healthy_patient_id(img),
            "scan_id": img.stem,
        })

    for img in mt_imgs:
        base = re.sub(r'_0000$', '', img.stem)
        label = mt_label_dir / f"{base}.nrrd"
        if not label.exists():
            missing.append(str(label))
            continue
        samples.append({
            "path": img.resolve(),
            "mask_path": label.resolve(),
            "label": 1,
            "class_name": "MT",
            "patient_id": mt_patient_id(img),
            "scan_id": base,
        })

    if missing:
        print("Missing pairs:")
        for x in missing[:20]:
            print(" ", x)
        raise SystemExit(f"Missing {len(missing)} masks/labels.")

    return samples


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
THRESHOLDS = [-975, -950, -925, -900, -850, -800, -750, -700,
              -650, -600, -550, -500, -400, -300, -200, 0]


def read_arr(path):
    arr, _ = nrrd.read(str(path))
    return np.squeeze(np.asarray(arr))


def extract_features(sample):
    img = read_arr(sample["path"]).astype(np.float32)
    mask_raw = read_arr(sample["mask_path"])

    if img.ndim != 3 or mask_raw.ndim != 3:
        raise ValueError(
            f"{sample['scan_id']}: expected 3D arrays, "
            f"got image={img.shape}, mask={mask_raw.shape}"
        )

    if img.shape != mask_raw.shape:
        raise ValueError(
            f"{sample['scan_id']}: shape mismatch "
            f"image={img.shape}, mask={mask_raw.shape}"
        )

    roi = mask_raw > 0
    if roi.sum() == 0:
        raise ValueError(f"{sample['scan_id']}: empty ROI mask")

    vals = img[roi]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError(f"{sample['scan_id']}: ROI has no finite intensities")

    feats = {}

    # Distribution moments
    feats["mean"] = float(np.mean(vals))
    feats["std"] = float(np.std(vals))
    feats["median"] = float(np.median(vals))
    feats["iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    feats["mad"] = float(np.median(np.abs(vals - np.median(vals))))

    # Percentiles
    for p in PCTS:
        feats[f"p{p:02d}"] = float(np.percentile(vals, p))

    # Fractions above fixed air/soft-tissue thresholds
    for t in THRESHOLDS:
        feats[f"frac_gt_{t}"] = float(np.mean(vals > t))

    # Broad air / transition / soft tissue fractions
    feats["frac_air_le_-900"] = float(np.mean(vals <= -900))
    feats["frac_mid_-900_-500"] = float(np.mean((vals > -900) & (vals <= -500)))
    feats["frac_soft_gt_-500"] = float(np.mean(vals > -500))
    feats["frac_soft_gt_-300"] = float(np.mean(vals > -300))
    feats["frac_nonair_gt_-800"] = float(np.mean(vals > -800))

    # Sanity-only fields: saved but NOT used for modeling
    sanity = {
        "roi_voxels": int(roi.sum()),
        "raw_dim0": int(img.shape[0]),
        "raw_dim1": int(img.shape[1]),
        "raw_dim2": int(img.shape[2]),
    }

    return feats, sanity


# ---------------------------------------------------------------------
# Metrics / calibration
# ---------------------------------------------------------------------
def metric_dict(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, prob))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_mt": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity_mt": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_healthy": float(tn / max(tn + fp, 1)),
        "f1_mt": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(y_true, prob):
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    j = tpr - fpr
    k = int(np.nanargmax(j))
    t = float(thresholds[k])
    if not np.isfinite(t):
        t = 0.5
    return float(np.clip(t, 0.01, 0.99))


# ---------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------
def evaluate_model(name, estimator, X, y, groups, samples, folds, seed):
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )

    oof_prob = np.full(len(y), np.nan)
    oof_pred_cal = np.full(len(y), -1, dtype=int)
    oof_thresh = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_results = []

    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        model = clone(estimator)
        model.fit(X[tr], y[tr])

        p_train = model.predict_proba(X[tr])[:, 1]
        threshold = youden_threshold(y[tr], p_train)

        p_val = model.predict_proba(X[va])[:, 1]
        pred_cal = (p_val >= threshold).astype(int)

        oof_prob[va] = p_val
        oof_pred_cal[va] = pred_cal
        oof_thresh[va] = threshold
        fold_id[va] = fold

        m05 = metric_dict(y[va], p_val, 0.5)
        mcal = metric_dict(y[va], p_val, threshold)

        fold_results.append({
            "fold": fold,
            "auc": m05["roc_auc"],
            "threshold_from_train": threshold,
            "accuracy_05": m05["accuracy"],
            "balanced_accuracy_05": m05["balanced_accuracy"],
            "accuracy_calibrated": mcal["accuracy"],
            "balanced_accuracy_calibrated": mcal["balanced_accuracy"],
            "sensitivity_calibrated": mcal["sensitivity_mt"],
            "specificity_calibrated": mcal["specificity_healthy"],
            "f1_calibrated": mcal["f1_mt"],
        })

        print(
            f"{name:10s} fold {fold+1}/{folds} | "
            f"AUC={m05['roc_auc']:.4f} | "
            f"acc@.5={m05['accuracy']:.3f} | "
            f"cal_acc={mcal['accuracy']:.3f} | "
            f"sens={mcal['sensitivity_mt']:.3f} | "
            f"spec={mcal['specificity_healthy']:.3f}"
        )

    fold_aucs = np.array([r["auc"] for r in fold_results], dtype=float)

    # Note: primary AUC is mean fold AUC because independently trained models
    # can have differently calibrated probability scales.
    primary_mean_auc = float(np.mean(fold_aucs))
    primary_sd_auc = float(np.std(fold_aucs, ddof=1))

    pooled_auc = float(roc_auc_score(y, oof_prob))

    tn, fp, fn, tp = confusion_matrix(y, oof_pred_cal, labels=[0, 1]).ravel()
    calibrated = {
        "accuracy": float(accuracy_score(y, oof_pred_cal)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred_cal)),
        "precision_mt": float(precision_score(y, oof_pred_cal, zero_division=0)),
        "sensitivity_mt": float(recall_score(y, oof_pred_cal, zero_division=0)),
        "specificity_healthy": float(tn / max(tn + fp, 1)),
        "f1_mt": float(f1_score(y, oof_pred_cal, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    result = {
        "model": name,
        "primary_mean_fold_auc": primary_mean_auc,
        "fold_auc_sd": primary_sd_auc,
        "fold_aucs": fold_aucs.tolist(),
        "pooled_oof_probability_auc": pooled_auc,
        "foldwise_train_calibrated_metrics": calibrated,
        "folds": fold_results,
    }

    return result, oof_prob, oof_pred_cal, oof_thresh, fold_id


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy_dir", required=True)
    ap.add_argument("--healthy_mask_dir", required=True)
    ap.add_argument("--mt_dir", required=True)
    ap.add_argument("--mt_label_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="classifier_intensity_results")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    samples = discover_samples(
        args.healthy_dir,
        args.healthy_mask_dir,
        args.mt_dir,
        args.mt_label_dir,
    )

    healthy = [s for s in samples if s["label"] == 0]
    mt = [s for s in samples if s["label"] == 1]

    print(f"Healthy scans: {len(healthy)}")
    print(f"MT scans:      {len(mt)}")
    print(f"Total scans:   {len(samples)}")
    print(f"Healthy patients: {len(set(s['patient_id'] for s in healthy))}")
    print(f"MT patients:      {len(set(s['patient_id'] for s in mt))}")

    if len(healthy) != 40 or len(mt) != 57:
        print("\nWARNING: expected 40 healthy and 57 MT development scans.")

    rows = []
    model_feature_names = None

    print("\nExtracting ROI intensity features...")
    for i, s in enumerate(samples, 1):
        feats, sanity = extract_features(s)
        if model_feature_names is None:
            model_feature_names = list(feats.keys())

        row = {
            "scan_id": s["scan_id"],
            "patient_id": s["patient_id"],
            "class_name": s["class_name"],
            "label": s["label"],
            "path": str(s["path"]),
            "mask_path": str(s["mask_path"]),
            **feats,
            **sanity,
        }
        rows.append(row)

        if i == 1 or i % 10 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}")

    # Print the easiest sanity signals for first sample of each class
    for class_name in ["healthy", "MT"]:
        r = next(x for x in rows if x["class_name"] == class_name)
        print(
            f"\n{class_name} example {r['scan_id']}: "
            f"median={r['median']:.1f}, p90={r['p90']:.1f}, "
            f"frac>-800={r['frac_gt_-800']:.3f}, "
            f"frac>-500={r['frac_gt_-500']:.3f}, "
            f"ROI voxels={r['roi_voxels']} (sanity only)"
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save features
    with open(outdir / "features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if args.dry_run:
        print("\nDRY RUN PASSED.")
        print(f"Model features: {len(model_feature_names)}")
        print("ROI voxel count / dimensions are NOT used as classifier inputs.")
        print(f"Saved features -> {(outdir / 'features.csv').resolve()}")
        return

    X = np.array(
        [[r[k] for k in model_feature_names] for r in rows],
        dtype=np.float64
    )
    y = np.array([r["label"] for r in rows], dtype=int)
    groups = np.array([r["patient_id"] for r in rows])

    models = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
                random_state=args.seed,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            max_features="sqrt",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    all_results = {}
    prediction_rows = []

    for name, model in models.items():
        print("\n" + "=" * 78)
        print(name.upper())
        print("=" * 78)

        result, oof_prob, oof_pred_cal, oof_thresh, fold_id = evaluate_model(
            name, model, X, y, groups, samples, args.folds, args.seed
        )
        all_results[name] = result

        for i, r in enumerate(rows):
            prediction_rows.append({
                "model": name,
                "scan_id": r["scan_id"],
                "patient_id": r["patient_id"],
                "true_label": r["label"],
                "true_class": r["class_name"],
                "fold": int(fold_id[i]),
                "prob_mt": float(oof_prob[i]),
                "train_selected_threshold": float(oof_thresh[i]),
                "pred_calibrated": int(oof_pred_cal[i]),
            })

        print(
            f"\n{name} PRIMARY mean fold AUC: "
            f"{result['primary_mean_fold_auc']:.4f} "
            f"± {result['fold_auc_sd']:.4f}"
        )
        print(
            f"fold AUCs: "
            + ", ".join(f"{x:.4f}" for x in result["fold_aucs"])
        )
        print(
            "calibrated OOF: "
            + json.dumps(result["foldwise_train_calibrated_metrics"])
        )

    # Save results
    with open(outdir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(outdir / "oof_predictions.csv", "w", newline="") as f:
        fields = [
            "model", "scan_id", "patient_id", "true_label", "true_class",
            "fold", "prob_mt", "train_selected_threshold", "pred_calibrated"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(prediction_rows)

    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    ranked = sorted(
        all_results.values(),
        key=lambda z: z["primary_mean_fold_auc"],
        reverse=True
    )
    for r in ranked:
        m = r["foldwise_train_calibrated_metrics"]
        print(
            f"{r['model']:14s} "
            f"mean fold AUC={r['primary_mean_fold_auc']:.4f} "
            f"| bal_acc={m['balanced_accuracy']:.3f} "
            f"| sens={m['sensitivity_mt']:.3f} "
            f"| spec={m['specificity_healthy']:.3f}"
        )

    print(f"\nSaved -> {outdir.resolve()}")
    print("  features.csv")
    print("  metrics.json")
    print("  oof_predictions.csv")


if __name__ == "__main__":
    main()
