#!/usr/bin/env python3
"""
End-to-end 3D CBCT classifier for the Garcia sinus project.

Task
----
Binary classification of individual sinus CBCT crops:
    0 = healthy / normal
    1 = mucosal thickening (MT)

Design choices for this project
-------------------------------
- Uses patient-level stratified cross-validation (left/right sinuses from the same
  patient never split across train/validation).
- Uses ONLY Dataset501 imagesTr for MT by default, so the locked MT test set is
  not touched.
- Handles the project's different crop sizes by converting every volume to a
  common 96x96x96 tensor.
- Per-volume intensity normalization prevents using the previously different
  healthy/MT global normalization constants as an accidental class cue.
- Saves out-of-fold predictions and metrics plus one model per fold.
- In prediction mode, averages the fold models as an ensemble.

Expected packages (already commonly present in the nnU-Net environment):
    torch numpy SimpleITK scikit-learn

Typical run from repository root:
    python train_sinus_classifier.py --healthy_dir PATH_TO_HEALTHY_SCANS --dry_run
    python train_sinus_classifier.py --healthy_dir PATH_TO_HEALTHY_SCANS --epochs 25 --folds 5

If your healthy scans are in one of the common project folders, --healthy_dir can
be omitted and the script will try to find them automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except Exception as e:
    raise SystemExit(
        "SimpleITK is required. Activate the sinus-unet environment first.\n"
        f"Original import error: {e}"
    )

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise SystemExit(f"PyTorch is required. Original import error: {e}")

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedGroupKFold
except Exception as e:
    raise SystemExit(
        "scikit-learn is required. Install it with: pip install scikit-learn\n"
        f"Original import error: {e}"
    )


# ----------------------------- reproducibility ----------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ------------------------------ file finding ------------------------------- #

IMAGE_EXTENSIONS = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd")


def is_image_file(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(ext) for ext in IMAGE_EXTENSIONS)


def looks_like_mask(p: Path) -> bool:
    s = str(p).lower()
    bad = (
        "segment",
        "labels",
        "labelstr",
        "mask",
        "prediction",
        "predictions",
        "validation",
        "probabilit",
    )
    return any(x in s for x in bad)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and is_image_file(p)]
    files = [p for p in files if not looks_like_mask(p)]
    return sorted(files)


def find_first_existing(root: Path, candidates: Sequence[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.exists() and list_images(p):
            return p
    return None


def auto_healthy_dir(root: Path) -> Path | None:
    # Intentionally conservative. We do not search the whole repo and guess labels.
    candidates = [
        "Healthy/data/scan",
        "healthy/data/scan",
        "HEALTHY/data/scan",
        "Healthy/scan",
        "healthy/scan",
        "data/healthy",
        "data/Healthy",
        "Healthy",
        "healthy",
    ]
    return find_first_existing(root, candidates)


def auto_mt_dir(root: Path) -> Path | None:
    # SAFE DEFAULT: development/training cases only. Never auto-use imagesTs.
    candidates = [
        "nnUNet_work/raw/Dataset501_SinusMT/imagesTr",
        "nnUNet_work/raw/Dataset501_SinusMT/imagesTr_57",
        "MT/data/scan_train",
        "MT/data/scan_dev",
    ]
    return find_first_existing(root, candidates)


def case_name_from_path(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    name = re.sub(r"_0000$", "", name, flags=re.IGNORECASE)
    return name


def patient_id_from_case(case: str) -> str:
    """Best-effort grouping for project names such as FileB15_MT_R -> FileB15."""
    s = case
    patterns = [
        r"_MT_[LR]$",
        r"_MT$",
        r"_[LR]$",
        r"_(healthy|normal)$",
        r"_(left|right)$",
    ]
    for pat in patterns:
        s2 = re.sub(pat, "", s, flags=re.IGNORECASE)
        if s2 != s:
            s = s2
            break
    return s


@dataclass
class Sample:
    path: Path
    label: int
    case: str
    patient: str


def collect_samples(healthy_dir: Path, mt_dir: Path) -> List[Sample]:
    healthy = list_images(healthy_dir)
    mt = list_images(mt_dir)

    # nnU-Net imagesTr filenames are modality images and are safe; other duplicate
    # modality files are not expected here.
    out: List[Sample] = []
    for p in healthy:
        case = case_name_from_path(p)
        out.append(Sample(p, 0, case, patient_id_from_case(case)))
    for p in mt:
        case = case_name_from_path(p)
        out.append(Sample(p, 1, case, patient_id_from_case(case)))
    return out


# ------------------------------ preprocessing ------------------------------ #

def center_crop_or_pad(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center crop/pad without changing physical scale when shapes are close."""
    out = arr
    # Crop each axis around its center.
    slices = []
    for dim, t in zip(out.shape, target):
        if dim > t:
            start = (dim - t) // 2
            slices.append(slice(start, start + t))
        else:
            slices.append(slice(0, dim))
    out = out[tuple(slices)]

    # Symmetric zero padding where needed.
    pads = []
    for dim, t in zip(out.shape, target):
        if dim < t:
            total = t - dim
            before = total // 2
            after = total - before
            pads.append((before, after))
        else:
            pads.append((0, 0))
    if any(a or b for a, b in pads):
        out = np.pad(out, pads, mode="constant", constant_values=0)
    return out


def resize_volume(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    x = torch.from_numpy(arr.astype(np.float32))[None, None]
    x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
    return x[0, 0].numpy()


def load_and_preprocess(path: Path, target_size: int) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {arr.shape} for {path}")

    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    # Robust per-volume normalization. This matters because the original healthy
    # and MT pipelines used different global mean/std constants.
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = np.clip(arr, lo, hi)
    mean = float(arr.mean())
    std = float(arr.std())
    arr = (arr - mean) / max(std, 1e-6)

    target = (target_size, target_size, target_size)
    # Project crops are typically 96^3 or 112^3. Prefer crop/pad to avoid
    # stretching one class differently. Resize only for substantially different
    # shapes.
    if all(abs(int(d) - target_size) <= 32 for d in arr.shape):
        arr = center_crop_or_pad(arr, target)
    else:
        arr = resize_volume(arr, target)

    return arr.astype(np.float32, copy=False)


class VolumeDataset(Dataset):
    def __init__(self, arrays: Sequence[np.ndarray], labels: Sequence[int], indices: Sequence[int], augment: bool):
        self.arrays = arrays
        self.labels = labels
        self.indices = list(indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = self.indices[k]
        x = torch.from_numpy(self.arrays[i].copy()).float().unsqueeze(0)
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)

        if self.augment:
            # Axis flips are conservative for sinus crops and cheap.
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    x = torch.flip(x, dims=(axis,))
            if random.random() < 0.35:
                x = x + 0.02 * torch.randn_like(x)
            if random.random() < 0.25:
                scale = random.uniform(0.95, 1.05)
                shift = random.uniform(-0.05, 0.05)
                x = x * scale + shift

        return x, y, i


# --------------------------------- model ----------------------------------- #

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        groups = 8 if cout >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SinusClassifier3D(nn.Module):
    def __init__(self, base: int = 16, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, base, 2),      # 96 -> 48
            ConvBlock(base, base * 2, 2),   # 48 -> 24
            ConvBlock(base * 2, base * 4, 2),  # 24 -> 12
            ConvBlock(base * 4, base * 8, 2),  # 12 -> 6
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 8, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


# -------------------------------- metrics ---------------------------------- #

def compute_metrics(y_true: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)

    return {
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_mt": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity_mt": float(recall_score(y, pred, zero_division=0)),
        "specificity_healthy": float(specificity),
        "f1_mt": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def safe_score_for_selection(metrics: dict) -> float:
    auc = metrics.get("roc_auc", float("nan"))
    if not math.isnan(auc):
        return float(auc)
    return float(metrics["balanced_accuracy"])


# ------------------------------- train/eval -------------------------------- #

def predict_loader(model, loader, device) -> Tuple[List[int], List[float], List[int]]:
    model.eval()
    ys, probs, ids = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pr = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(pr.tolist())
            ys.extend(y.numpy().astype(int).tolist())
            ids.extend(idx.numpy().astype(int).tolist())
    return ys, probs, ids


def train_one_fold(
    fold: int,
    arrays: Sequence[np.ndarray],
    samples: Sequence[Sample],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    outdir: Path,
    device: torch.device,
    args,
):
    train_ds = VolumeDataset(arrays, [s.label for s in samples], train_idx, augment=True)
    val_ds = VolumeDataset(arrays, [s.label for s in samples], val_idx, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model = SinusClassifier3D(base=args.base_channels, dropout=args.dropout).to(device)

    y_train = np.array([samples[i].label for i in train_idx])
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -1.0
    best_epoch = -1
    patience_left = args.patience
    ckpt = outdir / f"fold{fold}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        yv, pv, _ = predict_loader(model, val_loader, device)
        m = compute_metrics(yv, pv)
        score = safe_score_for_selection(m)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"fold {fold} | epoch {epoch:03d}/{args.epochs} | "
            f"loss {mean_loss:.4f} | AUC {m['roc_auc']:.4f} | "
            f"bal_acc {m['balanced_accuracy']:.4f}"
        )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "target_size": args.target_size,
                    "base_channels": args.base_channels,
                    "dropout": args.dropout,
                    "class_names": ["healthy", "MT"],
                },
                ckpt,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"fold {fold}: early stopping at epoch {epoch}; best epoch = {best_epoch}")
                break

    saved = torch.load(ckpt, map_location=device)
    model.load_state_dict(saved["model_state"])
    yv, pv, ids = predict_loader(model, val_loader, device)
    final_m = compute_metrics(yv, pv)
    return final_m, yv, pv, ids, best_epoch


# ------------------------------- prediction -------------------------------- #

def load_fold_model(path: Path, device: torch.device):
    c = torch.load(path, map_location=device)
    model = SinusClassifier3D(base=int(c.get("base_channels", 16)), dropout=float(c.get("dropout", 0.30)))
    model.load_state_dict(c["model_state"])
    model.to(device).eval()
    return model, int(c.get("target_size", 96))


def run_prediction(input_path: Path, model_dir: Path, device: torch.device):
    model_paths = sorted(model_dir.glob("fold*_best.pt"))
    if not model_paths:
        raise SystemExit(f"No fold*_best.pt models found in {model_dir}")

    files = [input_path] if input_path.is_file() else list_images(input_path)
    if not files:
        raise SystemExit(f"No input images found at {input_path}")

    loaded = [load_fold_model(p, device) for p in model_paths]
    target_size = loaded[0][1]

    print(f"Using {len(loaded)} fold models as an ensemble.")
    for p in files:
        arr = load_and_preprocess(p, target_size)
        x = torch.from_numpy(arr)[None, None].float().to(device)
        ps = []
        with torch.no_grad():
            for model, _ in loaded:
                ps.append(float(torch.sigmoid(model(x)).item()))
        prob = float(np.mean(ps))
        label = "MT" if prob >= 0.5 else "healthy"
        print(f"{p.name}\tP(MT)={prob:.4f}\t{label}")


# ---------------------------------- main ----------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="3D healthy-vs-MT CBCT classifier")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current folder)")
    p.add_argument("--healthy_dir", type=Path, default=None, help="Folder containing healthy sinus CBCT crops")
    p.add_argument("--mt_dir", type=Path, default=None, help="Folder containing MT CBCT crops. Default uses Dataset501 imagesTr only.")
    p.add_argument("--outdir", type=Path, default=Path("classifier_results"))
    p.add_argument("--target_size", type=int, default=96)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Only inspect data and patient grouping; do not train")
    p.add_argument("--predict", type=Path, default=None, help="Predict one image or a folder using saved fold models")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.predict is not None:
        run_prediction(args.predict.resolve(), outdir, device)
        return

    healthy_dir = args.healthy_dir.resolve() if args.healthy_dir else auto_healthy_dir(root)
    mt_dir = args.mt_dir.resolve() if args.mt_dir else auto_mt_dir(root)

    if healthy_dir is None:
        raise SystemExit(
            "Could not auto-find the healthy scan folder. Re-run with:\n"
            "  python train_sinus_classifier.py --healthy_dir \"C:\\path\\to\\healthy\\scans\" --dry_run"
        )
    if mt_dir is None:
        raise SystemExit(
            "Could not auto-find MT development images. Re-run with --mt_dir pointing to the 57 development scans.\n"
            "Do NOT point it at the locked test set."
        )

    samples = collect_samples(healthy_dir, mt_dir)
    if not samples:
        raise SystemExit("No samples found.")

    n_h = sum(s.label == 0 for s in samples)
    n_mt = sum(s.label == 1 for s in samples)
    patients_h = {s.patient for s in samples if s.label == 0}
    patients_mt = {s.patient for s in samples if s.label == 1}
    shared = patients_h & patients_mt

    print(f"Healthy directory: {healthy_dir}")
    print(f"MT directory:      {mt_dir}")
    print(f"Healthy scans: {n_h} | patients/groups: {len(patients_h)}")
    print(f"MT scans:      {n_mt} | patients/groups: {len(patients_mt)}")
    print(f"Total:         {len(samples)} scans")

    if n_h < 5 or n_mt < 5:
        raise SystemExit("Too few cases in one class; check the selected folders.")

    if shared:
        print(f"NOTE: {len(shared)} patient IDs appear in both classes. They will remain grouped together.")

    # Verify each file has a unique case path and print first few mappings.
    print("\nFirst 10 samples:")
    for s in samples[:10]:
        print(f"  label={s.label} patient={s.patient:<20} case={s.case:<25} file={s.path.name}")

    if args.dry_run:
        print("\nDry run complete. If the folders and patient IDs above look right, run again without --dry_run.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # Preload once. This is faster than re-reading ~100 NRRDs every epoch and is
    # usually well under 1 GB RAM for 96^3 float32 volumes.
    arrays: List[np.ndarray] = []
    print("\nPreloading and normalizing volumes...")
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            arrays.append(load_and_preprocess(s.path, args.target_size))
        except Exception as e:
            raise RuntimeError(f"Failed on {s.path}: {e}") from e
        if i == 1 or i % 10 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}")
    print(f"Preload finished in {(time.time()-t0):.1f}s")

    labels = np.array([s.label for s in samples], dtype=int)
    groups = np.array([s.patient for s in samples])
    indices = np.arange(len(samples))

    n_groups = len(set(groups.tolist()))
    folds = min(args.folds, n_groups)
    if folds < 2:
        raise SystemExit("Need at least two patient groups for validation.")

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)

    # Save the exact cohort manifest before training.
    manifest_path = outdir / "cohort_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "case", "patient", "label", "class", "path"])
        for i, s in enumerate(samples):
            w.writerow([i, s.case, s.patient, s.label, "MT" if s.label else "healthy", str(s.path)])

    oof_probs = np.full(len(samples), np.nan, dtype=float)
    oof_folds = np.full(len(samples), -1, dtype=int)
    fold_metrics = []

    for fold, (tr, va) in enumerate(splitter.split(indices, labels, groups)):
        # Hard leakage check.
        tr_groups = set(groups[tr].tolist())
        va_groups = set(groups[va].tolist())
        leak = tr_groups & va_groups
        if leak:
            raise RuntimeError(f"Patient leakage in fold {fold}: {sorted(leak)[:10]}")

        print("\n" + "=" * 78)
        print(
            f"FOLD {fold}: train={len(tr)} (healthy={(labels[tr]==0).sum()}, MT={(labels[tr]==1).sum()}) | "
            f"val={len(va)} (healthy={(labels[va]==0).sum()}, MT={(labels[va]==1).sum()})"
        )
        print("=" * 78)

        m, yv, pv, ids, best_epoch = train_one_fold(
            fold, arrays, samples, tr, va, outdir, device, args
        )
        for idx, prob in zip(ids, pv):
            oof_probs[idx] = prob
            oof_folds[idx] = fold
        m["fold"] = fold
        m["best_epoch"] = best_epoch
        m["n_train"] = int(len(tr))
        m["n_val"] = int(len(va))
        fold_metrics.append(m)
        print(f"fold {fold} best: {json.dumps(m, indent=2)}")

    if np.isnan(oof_probs).any():
        missing = np.where(np.isnan(oof_probs))[0].tolist()
        raise RuntimeError(f"Missing OOF predictions for indices: {missing}")

    overall = compute_metrics(labels.tolist(), oof_probs.tolist())
    overall["n_scans"] = int(len(samples))
    overall["n_healthy"] = int(n_h)
    overall["n_mt"] = int(n_mt)
    overall["n_patient_groups"] = int(n_groups)
    overall["folds"] = int(folds)

    pred_path = outdir / "oof_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "patient", "true_label", "true_class", "fold", "p_mt", "pred_label", "pred_class"])
        for i, s in enumerate(samples):
            pr = float(oof_probs[i])
            pred = int(pr >= 0.5)
            w.writerow([
                s.case, s.patient, s.label, "MT" if s.label else "healthy",
                int(oof_folds[i]), f"{pr:.8f}", pred, "MT" if pred else "healthy",
            ])

    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"overall_oof": overall, "folds": fold_metrics}, f, indent=2)

    print("\n" + "#" * 78)
    print("PATIENT-LEVEL OUT-OF-FOLD RESULT")
    print("#" * 78)
    print(json.dumps(overall, indent=2))
    print(f"\nSaved models/results to: {outdir}")
    print(f"OOF predictions: {pred_path}")
    print(f"Metrics:         {metrics_path}")
    print("\nFor a new crop later:")
    print(f'  python {Path(__file__).name} --predict "PATH_TO_SCAN.nrrd" --outdir "{outdir}"')

    # Important methodological warning in the output so it is not forgotten.
    print("\nIMPORTANT: A healthy-vs-MT classifier is only scientifically interpretable if the")
    print("two classes were cropped/preprocessed comparably. Different acquisition sites,")
    print("scanner protocols, or systematically different fields of view can become shortcuts.")


if __name__ == "__main__":
    main()
