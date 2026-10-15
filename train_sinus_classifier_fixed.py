#!/usr/bin/env python
"""
train_sinus_classifier_fixed.py

Binary 3D CBCT classifier:
    0 = healthy maxillary sinus
    1 = mucosal thickening (MT)

Designed for:
  Healthy scans: classifier_healthy_scans/*.nrrd
  MT scans: nnUNet_work/raw/Dataset501_SinusMT/imagesTr/*_0000.nrrd

Important:
- Uses ONLY the directory you explicitly pass for MT.
- Do not point --mt_dir at the locked test set.
- Patient-level stratified 5-fold CV.
- All volumes are resampled to the same 96^3 tensor before training.
"""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import nrrd
except Exception:
    nrrd = None

try:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix
    )
except Exception as e:
    raise SystemExit(
        "scikit-learn is required. Install with:\n"
        "  conda install scikit-learn -y\n"
        f"Original error: {e}"
    )

# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# File/patient handling
# ----------------------------
def healthy_patient_id(path: Path) -> str:
    # File10_L.nrrd -> H_File10
    stem = path.stem
    stem = re.sub(r'_[LR]$', '', stem)
    return "H_" + stem

def mt_patient_id(path: Path) -> str:
    # FileB14_MT_L_0000.nrrd -> M_FileB14
    # FileU8_MT_R_0000.nrrd  -> M_FileU8
    stem = path.stem
    stem = re.sub(r'_0000$', '', stem)
    stem = re.sub(r'_MT_[LR]$', '', stem)
    return "M_" + stem

def discover_samples(healthy_dir: Path, mt_dir: Path):
    if not healthy_dir.exists():
        raise SystemExit(f"Healthy directory does not exist: {healthy_dir}")
    if not mt_dir.exists():
        raise SystemExit(f"MT directory does not exist: {mt_dir}")

    healthy = sorted(
        p for p in healthy_dir.glob("*.nrrd")
        if "_GT" not in p.name and not p.name.endswith("_0000.nrrd")
    )
    # Explicitly accept nnU-Net naming, but fall back to all NRRDs if needed.
    mt = sorted(mt_dir.glob("*_0000.nrrd"))
    if not mt:
        mt = sorted(
            p for p in mt_dir.glob("*.nrrd")
            if "_GT" not in p.name
        )

    samples = []
    for p in healthy:
        samples.append({
            "path": p,
            "label": 0,
            "class_name": "healthy",
            "patient_id": healthy_patient_id(p),
            "scan_id": p.stem,
        })
    for p in mt:
        samples.append({
            "path": p,
            "label": 1,
            "class_name": "MT",
            "patient_id": mt_patient_id(p),
            "scan_id": re.sub(r'_0000$', '', p.stem),
        })
    return samples, healthy, mt

# ----------------------------
# NRRD loading / preprocessing
# ----------------------------
def read_nrrd(path: Path):
    if nrrd is not None:
        arr, header = nrrd.read(str(path))
        return np.asarray(arr, dtype=np.float32)
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return arr
    except Exception as e:
        raise RuntimeError(
            "Could not read NRRD. Install pynrrd with:\n"
            "  pip install pynrrd\n"
            f"File: {path}\nError: {e}"
        )

def preprocess_volume(arr: np.ndarray, target_size=96):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    # Robust clipping prevents extreme CBCT outliers from dominating.
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        arr = np.clip(arr, lo, hi)

    # Per-volume z-score: reduces scanner/global intensity shortcut learning.
    mean = float(arr.mean())
    std = float(arr.std())
    arr = (arr - mean) / max(std, 1e-6)

    x = torch.from_numpy(arr).float()[None, None]  # 1,1,D,H,W
    if tuple(arr.shape) != (target_size, target_size, target_size):
        x = F.interpolate(
            x,
            size=(target_size, target_size, target_size),
            mode="trilinear",
            align_corners=False,
        )
    return x[0]  # 1,D,H,W

def preload(samples, target_size=96):
    cache = {}
    print(f"\nPreloading/resampling {len(samples)} scans to {target_size}^3 ...")
    for i, s in enumerate(samples, 1):
        try:
            cache[str(s["path"])] = preprocess_volume(
                read_nrrd(s["path"]), target_size
            )
        except Exception as e:
            raise RuntimeError(f"Failed on {s['path']}: {e}") from e
        if i == 1 or i % 10 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}")
    return cache

# ----------------------------
# Dataset / augmentation
# ----------------------------
class SinusDataset(Dataset):
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
            # Safe geometric augmentation for local sinus crops.
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
            if random.random() < 0.35:
                x = x * random.uniform(0.90, 1.10)
            if random.random() < 0.35:
                x = x + torch.randn_like(x) * random.uniform(0.0, 0.04)

        y = torch.tensor(float(s["label"]), dtype=torch.float32)
        return x, y, idx

# ----------------------------
# Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = min(8, cout)
        while cout % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.MaxPool3d(2),
        )
    def forward(self, x):
        return self.net(x)

class Small3DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),   # 96 -> 48
            ConvBlock(16, 32),  # 48 -> 24
            ConvBlock(32, 64),  # 24 -> 12
            ConvBlock(64, 96),  # 12 -> 6
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

# ----------------------------
# Metrics
# ----------------------------
def metrics_from_probs(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)

    return {
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_mt": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity_mt": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_healthy": float(specificity),
        "f1_mt": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, probs, local_indices = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()
        ys.extend(y.numpy().tolist())
        probs.extend(p.tolist())
        local_indices.extend(idx.numpy().tolist())
    return np.array(ys), np.array(probs), np.array(local_indices)

# ----------------------------
# Main training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy_dir", required=True)
    ap.add_argument("--mt_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--target_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--output_dir", default="classifier_results")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    healthy_dir = Path(args.healthy_dir).resolve()
    mt_dir = Path(args.mt_dir).resolve()

    samples, healthy, mt = discover_samples(healthy_dir, mt_dir)

    print(f"\nHealthy directory: {healthy_dir}")
    print(f"MT directory:      {mt_dir}")
    print(f"Healthy scans: {len(healthy)}")
    print(f"MT scans:      {len(mt)}")
    print(f"Total scans:   {len(samples)}")

    if len(samples) == 0:
        raise SystemExit("No samples found.")

    h_patients = sorted({s["patient_id"] for s in samples if s["label"] == 0})
    m_patients = sorted({s["patient_id"] for s in samples if s["label"] == 1})
    print(f"Healthy patients: {len(h_patients)}")
    print(f"MT patients:      {len(m_patients)}")
    print("\nFirst healthy:", [p.name for p in healthy[:5]])
    print("First MT:     ", [p.name for p in mt[:5]])

    # Read first file from each class now so dry-run catches bad LFS pointers.
    for label_name, files in [("healthy", healthy), ("MT", mt)]:
        if files:
            a = read_nrrd(files[0])
            print(f"{label_name} first raw shape: {tuple(a.shape)}, dtype={a.dtype}, "
                  f"min={float(np.min(a)):.2f}, max={float(np.max(a)):.2f}")

    if args.dry_run:
        print("\nDRY RUN PASSED. No training performed.")
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cohort manifest
    with open(outdir / "cohort_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scan_id","patient_id","class_name","label","path"])
        w.writeheader()
        for s in samples:
            w.writerow({k: str(s[k]) for k in w.fieldnames})

    cache = preload(samples, args.target_size)

    labels = np.array([s["label"] for s in samples], dtype=int)
    groups = np.array([s["patient_id"] for s in samples])
    indices = np.arange(len(samples))

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    oof_prob = np.full(len(samples), np.nan, dtype=float)
    fold_of = np.full(len(samples), -1, dtype=int)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(indices, labels, groups)):
        print("\n" + "="*72)
        print(f"FOLD {fold+1}/{args.folds}")
        print("="*72)

        tr_samples = [samples[i] for i in tr_idx]
        va_samples = [samples[i] for i in va_idx]

        tr_groups = {s["patient_id"] for s in tr_samples}
        va_groups = {s["patient_id"] for s in va_samples}
        overlap = tr_groups & va_groups
        if overlap:
            raise RuntimeError(f"Patient leakage in fold {fold}: {sorted(overlap)}")

        tr_y = labels[tr_idx]
        va_y = labels[va_idx]
        print(f"Train: {len(tr_idx)} scans | healthy={(tr_y==0).sum()} MT={(tr_y==1).sum()} | patients={len(tr_groups)}")
        print(f"Val:   {len(va_idx)} scans | healthy={(va_y==0).sum()} MT={(va_y==1).sum()} | patients={len(va_groups)}")

        train_ds = SinusDataset(tr_samples, cache, augment=True)
        val_ds = SinusDataset(va_samples, cache, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

        model = Small3DClassifier().to(device)

        n_neg = max(int((tr_y == 0).sum()), 1)
        n_pos = max(int((tr_y == 1).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_auc = -1.0
        best_state = None
        best_epoch = -1
        patience = 8
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for x, y, _ in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            vy, vp, _ = evaluate(model, val_loader, device)
            vm = metrics_from_probs(vy, vp)
            val_auc = vm["roc_auc"]
            score = -999 if math.isnan(val_auc) else val_auc

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"loss={np.mean(losses):.4f} | "
                f"val AUC={val_auc:.4f} | "
                f"acc={vm['accuracy']:.3f} | "
                f"sens={vm['recall_sensitivity_mt']:.3f} | "
                f"spec={vm['specificity_healthy']:.3f}"
            )

            if score > best_auc + 1e-5:
                best_auc = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "fold": fold,
                        "epoch": epoch,
                        "val_auc": val_auc,
                        "target_size": args.target_size,
                    },
                    outdir / f"fold{fold}_best.pt",
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, AUC={best_auc:.4f}")
                    break

        model.load_state_dict(best_state)
        vy, vp, local_idx = evaluate(model, val_loader, device)

        # local_idx refers to val_ds indices; map back to global indices.
        global_idx = va_idx[local_idx]
        oof_prob[global_idx] = vp
        fold_of[global_idx] = fold

        fm = metrics_from_probs(vy, vp)
        fm["fold"] = fold
        fm["best_epoch"] = best_epoch
        fold_metrics.append(fm)
        print("BEST FOLD METRICS:", json.dumps(fm, indent=2))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_prob).any():
        missing = np.where(np.isnan(oof_prob))[0].tolist()
        raise RuntimeError(f"Missing OOF predictions for indices: {missing}")

    overall = metrics_from_probs(labels, oof_prob)
    overall["n_scans"] = int(len(samples))
    overall["n_patients"] = int(len(set(groups)))

    with open(outdir / "oof_predictions.csv", "w", newline="") as f:
        fieldnames = [
            "scan_id","patient_id","true_label","true_class",
            "fold","prob_mt","pred_label","pred_class","path"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, s in enumerate(samples):
            pred = int(oof_prob[i] >= 0.5)
            w.writerow({
                "scan_id": s["scan_id"],
                "patient_id": s["patient_id"],
                "true_label": s["label"],
                "true_class": s["class_name"],
                "fold": int(fold_of[i]),
                "prob_mt": float(oof_prob[i]),
                "pred_label": pred,
                "pred_class": "MT" if pred else "healthy",
                "path": str(s["path"]),
            })

    with open(outdir / "metrics.json", "w") as f:
        json.dump({"overall_oof": overall, "folds": fold_metrics}, f, indent=2)

    print("\n" + "="*72)
    print("PATIENT-LEVEL OUT-OF-FOLD RESULT")
    print("="*72)
    for k, v in overall.items():
        print(f"{k}: {v}")

    print(f"\nSaved to: {outdir.resolve()}")
    print("  metrics.json")
    print("  oof_predictions.csv")
    print("  cohort_manifest.csv")
    print("  fold0_best.pt ...")

if __name__ == "__main__":
    main()
