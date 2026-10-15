#!/usr/bin/env python
"""
train_sinus_classifier_roi_v2.py

ROI-focused binary 3D CBCT classifier:
    0 = healthy maxillary sinus
    1 = mucosal thickening (MT)

Why this V2 differs from the first classifier:
- Uses anatomical sinus masks ONLY to crop tightly around the sinus.
- Preserves absolute CBCT intensity contrast with a fixed window instead of
  per-volume z-scoring.
- Removes anatomically unrealistic 3D flips.
- Uses avg + max pooling so focal soft-tissue abnormalities are not averaged away.
- Uses patient-level StratifiedGroupKFold.
- Reports mean fold ROC-AUC (primary), plus 0.5-threshold and train-calibrated metrics.

Expected filenames:
Healthy image: File10_L.nrrd
Healthy mask:  File10_L_GT.nrrd

MT image:      FileB11_MT_L_0000.nrrd
MT label:      FileB11_MT_L.nrrd

IMPORTANT:
For MT labels, the union (label > 0) is used ONLY as an anatomical ROI.
The classifier never receives class-1-vs-class-2 label identities as input.
Do not point this script at the locked MT test set.
"""

import argparse
import csv
import json
import math
import random
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import nrrd
except Exception as e:
    raise SystemExit(
        "pynrrd is required. Install it with:\n"
        "  pip install pynrrd\n"
        f"Original error: {e}"
    )

try:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix,
        roc_curve
    )
except Exception as e:
    raise SystemExit(
        "scikit-learn is required. Install it with:\n"
        "  conda install scikit-learn -y\n"
        f"Original error: {e}"
    )


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Patient IDs
# ---------------------------------------------------------------------
def healthy_patient_id(path: Path) -> str:
    # File10_L.nrrd -> H_File10
    stem = re.sub(r'_[LR]$', '', path.stem)
    return "H_" + stem


def mt_patient_id(path: Path) -> str:
    # FileB11_MT_L_0000.nrrd -> M_FileB11
    stem = re.sub(r'_0000$', '', path.stem)
    stem = re.sub(r'_MT_[LR]$', '', stem)
    return "M_" + stem


# ---------------------------------------------------------------------
# Cohort discovery
# ---------------------------------------------------------------------
def discover_samples(healthy_dir: Path, healthy_mask_dir: Path,
                     mt_dir: Path, mt_label_dir: Path):
    for d, name in [
        (healthy_dir, "healthy_dir"),
        (healthy_mask_dir, "healthy_mask_dir"),
        (mt_dir, "mt_dir"),
        (mt_label_dir, "mt_label_dir"),
    ]:
        if not d.exists():
            raise SystemExit(f"{name} does not exist: {d}")

    healthy_images = sorted(
        p for p in healthy_dir.glob("*.nrrd")
        if "_GT" not in p.name
    )
    mt_images = sorted(mt_dir.glob("*_0000.nrrd"))
    if not mt_images:
        raise SystemExit(
            f"No nnU-Net MT images matching *_0000.nrrd found in {mt_dir}"
        )

    samples = []
    missing = []

    for img in healthy_images:
        mask = healthy_mask_dir / f"{img.stem}_GT.nrrd"
        if not mask.exists():
            missing.append(str(mask))
            continue
        samples.append({
            "path": img,
            "mask_path": mask,
            "label": 0,
            "class_name": "healthy",
            "patient_id": healthy_patient_id(img),
            "scan_id": img.stem,
        })

    for img in mt_images:
        base = re.sub(r'_0000$', '', img.stem)
        label_path = mt_label_dir / f"{base}.nrrd"
        if not label_path.exists():
            missing.append(str(label_path))
            continue
        samples.append({
            "path": img,
            "mask_path": label_path,
            "label": 1,
            "class_name": "MT",
            "patient_id": mt_patient_id(img),
            "scan_id": base,
        })

    if missing:
        print("\nMissing image-mask pairs (first 20):")
        for p in missing[:20]:
            print("  ", p)
        print(f"Total missing: {len(missing)}")
        raise SystemExit("Fix the missing masks/labels before training.")

    healthy = [s for s in samples if s["label"] == 0]
    mt = [s for s in samples if s["label"] == 1]
    return samples, healthy, mt


# ---------------------------------------------------------------------
# NRRD + ROI preprocessing
# ---------------------------------------------------------------------
def read_nrrd(path: Path):
    arr, _ = nrrd.read(str(path))
    return np.asarray(arr)


def bbox_from_mask(mask: np.ndarray, margin: int):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        raise ValueError("ROI mask is empty")

    lo = pts.min(axis=0)
    hi = pts.max(axis=0) + 1

    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.array(mask.shape))

    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def preprocess_pair(image_path: Path, mask_path: Path,
                    target_size: int, margin: int,
                    window_low: float, window_high: float):
    img = np.asarray(read_nrrd(image_path), dtype=np.float32)
    mask_raw = np.asarray(read_nrrd(mask_path))

    img = np.squeeze(img)
    mask_raw = np.squeeze(mask_raw)

    if img.ndim != 3 or mask_raw.ndim != 3:
        raise ValueError(
            f"Expected 3D image/mask, got image={img.shape}, mask={mask_raw.shape}"
        )
    if img.shape != mask_raw.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image={img.shape}, mask={mask_raw.shape}"
        )

    # Healthy mask: foreground > 0.
    # MT label: union of MT and air = labels > 0.
    # This removes the disease-label identity and retains only the sinus ROI.
    roi = mask_raw > 0

    sl = bbox_from_mask(roi, margin=margin)
    crop = img[sl].astype(np.float32)
    roi_crop = roi[sl]

    # Fixed intensity window preserves absolute air/tissue contrast.
    crop = np.clip(crop, window_low, window_high)
    crop = 2.0 * (crop - window_low) / (window_high - window_low) - 1.0

    # We use the mask ONLY for localization/cropping, not as a CNN channel.
    x = torch.from_numpy(crop)[None, None].float()
    x = F.interpolate(
        x,
        size=(target_size, target_size, target_size),
        mode="trilinear",
        align_corners=False,
    )[0]

    # Useful sanity features (not fed to CNN).
    roi_vox = img[roi].astype(np.float32)
    stats = {
        "raw_shape": tuple(int(v) for v in img.shape),
        "crop_shape": tuple(int(v) for v in crop.shape),
        "roi_voxels": int(roi.sum()),
        "roi_mean": float(np.mean(roi_vox)),
        "roi_std": float(np.std(roi_vox)),
        "roi_p10": float(np.percentile(roi_vox, 10)),
        "roi_p50": float(np.percentile(roi_vox, 50)),
        "roi_p90": float(np.percentile(roi_vox, 90)),
    }
    return x, stats


def preload(samples, target_size, margin, window_low, window_high):
    cache = {}
    stats = {}
    print(f"\nPreloading {len(samples)} ROI crops -> {target_size}^3 ...")
    for i, s in enumerate(samples, 1):
        x, st = preprocess_pair(
            s["path"], s["mask_path"],
            target_size=target_size,
            margin=margin,
            window_low=window_low,
            window_high=window_high,
        )
        cache[str(s["path"])] = x
        stats[str(s["path"])] = st
        if i == 1 or i % 10 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}")
    return cache, stats


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class SinusROIDataset(Dataset):
    def __init__(self, samples, cache, augment=False):
        self.samples = samples
        self.cache = cache
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = self.cache[str(s["path"])].clone()

        if self.augment:
            # Mild, anatomy-preserving augmentation only.
            if random.random() < 0.5:
                gain = random.uniform(0.92, 1.08)
                offset = random.uniform(-0.04, 0.04)
                x = torch.clamp(x * gain + offset, -1.25, 1.25)

            if random.random() < 0.35:
                sigma = random.uniform(0.005, 0.025)
                x = x + torch.randn_like(x) * sigma

            if random.random() < 0.35:
                shifts = [
                    random.randint(-2, 2),
                    random.randint(-2, 2),
                    random.randint(-2, 2),
                ]
                x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))

        y = torch.tensor(float(s["label"]), dtype=torch.float32)
        return x, y, idx


# ---------------------------------------------------------------------
# 3D residual classifier
# ---------------------------------------------------------------------
class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        g1 = min(8, cout)
        while cout % g1:
            g1 -= 1

        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g1, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g1, cout)
        self.act = nn.GELU()

        if cin != cout or stride != 1:
            self.skip = nn.Conv3d(cin, cout, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + identity)


class ROIResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            ResBlock3D(16, 16),
            ResBlock3D(16, 32, stride=2),
            ResBlock3D(32, 32),
            ResBlock3D(32, 64, stride=2),
            ResBlock3D(64, 64),
            ResBlock3D(64, 96, stride=2),
            ResBlock3D(96, 96),
        )
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.mx = nn.AdaptiveMaxPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.body(self.stem(x))
        a = self.avg(x).flatten(1)
        m = self.mx(x).flatten(1)
        x = torch.cat([a, m], dim=1)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------
# Metrics / threshold calibration
# ---------------------------------------------------------------------
def metric_dict(y_true, probs, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    auc = (
        float(roc_auc_score(y_true, probs))
        if len(np.unique(y_true)) == 2 else float("nan")
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

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


def youden_threshold(y_true, probs):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    k = int(np.nanargmax(j))
    t = float(thresholds[k])
    if not np.isfinite(t):
        t = 0.5
    return float(np.clip(t, 0.01, 0.99))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, local_idx = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(p.tolist())
        local_idx.extend(idx.numpy().tolist())
    return np.asarray(ys), np.asarray(ps), np.asarray(local_idx)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy_dir", required=True)
    ap.add_argument("--healthy_mask_dir", required=True)
    ap.add_argument("--mt_dir", required=True)
    ap.add_argument("--mt_label_dir", required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--target_size", type=int, default=96)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--window_low", type=float, default=-1000.0)
    ap.add_argument("--window_high", type=float, default=2000.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--output_dir", default="classifier_roi_v2_results")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    healthy_dir = Path(args.healthy_dir).resolve()
    healthy_mask_dir = Path(args.healthy_mask_dir).resolve()
    mt_dir = Path(args.mt_dir).resolve()
    mt_label_dir = Path(args.mt_label_dir).resolve()

    samples, healthy, mt = discover_samples(
        healthy_dir, healthy_mask_dir, mt_dir, mt_label_dir
    )

    print(f"\nHealthy scans: {len(healthy)}")
    print(f"MT scans:      {len(mt)}")
    print(f"Total scans:   {len(samples)}")
    print(f"Healthy patients: {len(set(s['patient_id'] for s in healthy))}")
    print(f"MT patients:      {len(set(s['patient_id'] for s in mt))}")

    if not samples:
        raise SystemExit("No paired samples found.")

    print("\nFirst healthy pair:")
    print(" image:", healthy[0]["path"])
    print(" mask: ", healthy[0]["mask_path"])
    print("First MT pair:")
    print(" image:", mt[0]["path"])
    print(" label:", mt[0]["mask_path"])

    # Read and preprocess one from each class in dry-run to validate masks.
    for name, s in [("healthy", healthy[0]), ("MT", mt[0])]:
        x, st = preprocess_pair(
            s["path"], s["mask_path"],
            args.target_size, args.margin,
            args.window_low, args.window_high
        )
        print(f"\n{name} sanity:")
        print(f" tensor shape: {tuple(x.shape)}")
        print(f" raw shape:    {st['raw_shape']}")
        print(f" crop shape:   {st['crop_shape']}")
        print(f" ROI voxels:   {st['roi_voxels']}")
        print(f" ROI intensity p10/p50/p90: "
              f"{st['roi_p10']:.1f} / {st['roi_p50']:.1f} / {st['roi_p90']:.1f}")

    if args.dry_run:
        print("\nDRY RUN PASSED.")
        print("This V2 is ROI-cropping correctly and is ready to train.")
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache, prep_stats = preload(
        samples,
        args.target_size,
        args.margin,
        args.window_low,
        args.window_high,
    )

    # Save cohort + ROI sanity stats.
    with open(outdir / "cohort_manifest.csv", "w", newline="") as f:
        fields = [
            "scan_id", "patient_id", "class_name", "label",
            "path", "mask_path", "raw_shape", "crop_shape",
            "roi_voxels", "roi_mean", "roi_std", "roi_p10", "roi_p50", "roi_p90"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            st = prep_stats[str(s["path"])]
            row = {k: s[k] for k in [
                "scan_id", "patient_id", "class_name", "label",
                "path", "mask_path"
            ]}
            row.update(st)
            w.writerow(row)

    labels = np.asarray([s["label"] for s in samples], dtype=int)
    groups = np.asarray([s["patient_id"] for s in samples])
    all_idx = np.arange(len(samples))

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    oof_prob = np.full(len(samples), np.nan, dtype=float)
    oof_pred_cal = np.full(len(samples), -1, dtype=int)
    oof_threshold = np.full(len(samples), np.nan, dtype=float)
    fold_of = np.full(len(samples), -1, dtype=int)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(
        splitter.split(all_idx, labels, groups)
    ):
        print("\n" + "=" * 78)
        print(f"FOLD {fold + 1}/{args.folds}")
        print("=" * 78)

        tr_samples = [samples[i] for i in tr_idx]
        va_samples = [samples[i] for i in va_idx]

        tr_groups = {s["patient_id"] for s in tr_samples}
        va_groups = {s["patient_id"] for s in va_samples}
        overlap = tr_groups & va_groups
        if overlap:
            raise RuntimeError(f"PATIENT LEAKAGE: {sorted(overlap)}")

        tr_y = labels[tr_idx]
        va_y = labels[va_idx]

        print(
            f"Train {len(tr_idx)} | H={(tr_y==0).sum()} MT={(tr_y==1).sum()} "
            f"| patients={len(tr_groups)}"
        )
        print(
            f"Val   {len(va_idx)} | H={(va_y==0).sum()} MT={(va_y==1).sum()} "
            f"| patients={len(va_groups)}"
        )

        train_ds = SinusROIDataset(tr_samples, cache, augment=True)
        train_eval_ds = SinusROIDataset(tr_samples, cache, augment=False)
        val_ds = SinusROIDataset(va_samples, cache, augment=False)

        # Balanced sampling instead of changing the loss probability scale.
        class_counts = np.bincount(tr_y, minlength=2)
        weights = np.asarray([
            1.0 / max(class_counts[y], 1) for y in tr_y
        ], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = ROIResNet3D().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )

        best_auc = -1.0
        best_epoch = -1
        best_state = None
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for x, y, _ in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))

            scheduler.step()

            vy, vp, _ = predict(model, val_loader, device)
            vm = metric_dict(vy, vp, 0.5)
            val_auc = vm["roc_auc"]
            score = -999.0 if math.isnan(val_auc) else val_auc

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"loss={np.mean(losses):.4f} "
                f"AUC={val_auc:.4f} "
                f"acc@.5={vm['accuracy']:.3f} "
                f"sens={vm['sensitivity_mt']:.3f} "
                f"spec={vm['specificity_healthy']:.3f}"
            )

            if score > best_auc + 1e-5:
                best_auc = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                torch.save({
                    "model_state_dict": best_state,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "val_auc": best_auc,
                    "target_size": args.target_size,
                    "margin": args.margin,
                    "window_low": args.window_low,
                    "window_high": args.window_high,
                }, outdir / f"fold{fold}_best.pt")
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch >= 12 and bad_epochs >= args.patience:
                print(
                    f"Early stopping: epoch={epoch}, "
                    f"best_epoch={best_epoch}, best_AUC={best_auc:.4f}"
                )
                break

        model.load_state_dict(best_state)

        # Threshold is chosen on TRAIN data only, then applied to held-out fold.
        ty, tp, _ = predict(model, train_eval_loader, device)
        threshold = youden_threshold(ty, tp)

        vy, vp, local_idx = predict(model, val_loader, device)
        global_idx = va_idx[local_idx]

        oof_prob[global_idx] = vp
        fold_of[global_idx] = fold
        oof_threshold[global_idx] = threshold
        oof_pred_cal[global_idx] = (vp >= threshold).astype(int)

        m05 = metric_dict(vy, vp, 0.5)
        mcal = metric_dict(vy, vp, threshold)

        fm = {
            "fold": fold,
            "best_epoch": best_epoch,
            "train_selected_threshold": threshold,
            "auc": m05["roc_auc"],
            "accuracy_05": m05["accuracy"],
            "balanced_accuracy_05": m05["balanced_accuracy"],
            "accuracy_calibrated": mcal["accuracy"],
            "balanced_accuracy_calibrated": mcal["balanced_accuracy"],
            "sensitivity_calibrated": mcal["sensitivity_mt"],
            "specificity_calibrated": mcal["specificity_healthy"],
            "f1_calibrated": mcal["f1_mt"],
        }
        fold_metrics.append(fm)

        print("\nBEST FOLD:")
        print(json.dumps(fm, indent=2))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_prob).any():
        raise RuntimeError("Some scans did not receive an OOF prediction.")

    fold_aucs = np.asarray([m["auc"] for m in fold_metrics], dtype=float)
    mean_fold_auc = float(np.nanmean(fold_aucs))
    sd_fold_auc = float(np.nanstd(fold_aucs, ddof=1))

    pooled_05 = metric_dict(labels, oof_prob, 0.5)

    # Metrics using each fold's threshold chosen from that fold's training set.
    tn, fp, fn, tp = confusion_matrix(
        labels, oof_pred_cal, labels=[0, 1]
    ).ravel()
    calibrated = {
        "accuracy": float(accuracy_score(labels, oof_pred_cal)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, oof_pred_cal)
        ),
        "precision_mt": float(
            precision_score(labels, oof_pred_cal, zero_division=0)
        ),
        "sensitivity_mt": float(
            recall_score(labels, oof_pred_cal, zero_division=0)
        ),
        "specificity_healthy": float(tn / max(tn + fp, 1)),
        "f1_mt": float(f1_score(labels, oof_pred_cal, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    summary = {
        "PRIMARY_mean_fold_roc_auc": mean_fold_auc,
        "fold_roc_auc_sd": sd_fold_auc,
        "fold_roc_aucs": fold_aucs.tolist(),
        "pooled_oof_probability_auc": float(
            roc_auc_score(labels, oof_prob)
        ),
        "pooled_metrics_threshold_0.5": pooled_05,
        "pooled_metrics_foldwise_train_calibrated": calibrated,
        "n_scans": int(len(samples)),
        "n_patients": int(len(set(groups))),
        "folds": fold_metrics,
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "oof_predictions.csv", "w", newline="") as f:
        fields = [
            "scan_id", "patient_id", "true_label", "true_class",
            "fold", "prob_mt", "train_selected_threshold",
            "pred_calibrated", "path"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(samples):
            w.writerow({
                "scan_id": s["scan_id"],
                "patient_id": s["patient_id"],
                "true_label": s["label"],
                "true_class": s["class_name"],
                "fold": int(fold_of[i]),
                "prob_mt": float(oof_prob[i]),
                "train_selected_threshold": float(oof_threshold[i]),
                "pred_calibrated": int(oof_pred_cal[i]),
                "path": str(s["path"]),
            })

    print("\n" + "=" * 78)
    print("ROI V2 FINAL RESULTS")
    print("=" * 78)
    print(f"PRIMARY mean fold ROC-AUC: {mean_fold_auc:.4f} ± {sd_fold_auc:.4f}")
    print("Fold AUCs:", ", ".join(f"{x:.4f}" for x in fold_aucs))
    print(f"Pooled OOF probability AUC: {summary['pooled_oof_probability_auc']:.4f}")
    print("\nFoldwise TRAIN-calibrated held-out predictions:")
    for k, v in calibrated.items():
        print(f"{k}: {v}")

    print(f"\nSaved results -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
