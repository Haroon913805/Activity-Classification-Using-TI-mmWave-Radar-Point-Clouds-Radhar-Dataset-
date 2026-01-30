
# ============================================
# Micro-Doppler from TI mmWave point-cloud .txt
# ============================================

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, medfilt  # FIXED
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import seaborn as sns
# ---------------------------
# 0) CONFIG — EDIT THIS PATH
# ---------------------------
CLASSES = ["boxing", "jump", "squats", "jack", "walk"]
ROOT = r"E:\Data sceince\Micro_Doppler_Signatures\Micro Doppler Signatures\Micro Doppler Signatures\radhar\RadHAR-master\RadHAR-master/Data\Train"

# Convert ROOT to a proper Path object even if it's a string
from pathlib import Path
ROOT = Path(ROOT).expanduser().resolve()


# If your data is under Train:
# ROOT = Path(r"H:\Micro Doppler Signatures\radhar\RadHAR-master\RadHAR-master\Data\Train").resolve()

# If you want to restrict to specific classes, list them; else it auto-detects subfolders as classes.
CLASSES = None  # e.g., ["boxing", "jump", "squats", "jack", "walk"] or None to auto-discover
ASSUME_FPS_IF_UNKNOWN = 10.0  # Hz, used if timestamps are missing/invalid

# ------- NEW: Visualization & export config -------
SAVE_DIR = (ROOT.parent / "Spectrograms").resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 5              # how many examples to save per class
SPEC_SIZE_MODEL = (64, 64)   # size used by the CNN (keep as-is for your current model)
# Higher STFT resolution for prettier pictures (no resize used for saving)
NPERSEG_VIZ = 128
NOVERLAP_VIZ = 96
CMAP = "magma"               # or "turbo", "viridis", "jet"
# -------------------------------------------------


# ---------------------------
# 1) Parsing helper functions
# ---------------------------
def parse_txt_file(file_path: Path):
    """
    Parse a TI mmWave point-cloud style .txt where each 'header:' starts a new frame.
    Returns:
      frames_bins: list of list[int]     -> doppler_bin values per frame
      frames_w:    list of list[float]   -> corresponding intensity weights per frame
      frame_times: list[float]           -> timestamp (seconds) per frame (if parsed), else []
    """
    frames_bins = []
    frames_w = []
    frame_times = []

    current_bins = []
    current_w = []
    current_secs = None
    current_nsecs = None
    inside_frame = False

    # Regex helpers
    rgx_int = re.compile(r"[-]?\d+")
    rgx_float = re.compile(r"[-]?\d+\.\d+|[-]?\d+")

    # local "last bin" per line pairing (avoid global side-effects)
    last_bin_local = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # New frame starts when we hit 'header:'
            if line.startswith("header:"):
                # flush previous frame if any
                if inside_frame:
                    if len(current_bins) > 0:
                        frames_bins.append(current_bins)
                        frames_w.append(current_w)
                        if (current_secs is not None) and (current_nsecs is not None):
                            frame_times.append(float(current_secs) + float(current_nsecs) * 1e-9)
                        else:
                            frame_times.append(np.nan)
                    current_bins = []
                    current_w = []
                    current_secs = None
                    current_nsecs = None
                inside_frame = True
                last_bin_local = None
                continue

            # Timestamp fields inside header block
            if line.startswith("secs:"):
                ints = rgx_int.findall(line)
                if ints:
                    current_secs = int(ints[0])
                continue
            if line.startswith("nsecs:"):
                ints = rgx_int.findall(line)
                if ints:
                    current_nsecs = int(ints[0])
                continue

            # Point fields
            if line.startswith("doppler_bin"):
                ints = rgx_int.findall(line)
                doppler_bin = int(ints[0]) if ints else 0
                last_bin_local = doppler_bin
                continue

            if line.startswith("intensity"):
                fls = rgx_float.findall(line)
                # accept integers or floats; default 1.0
                intensity = float(fls[0]) if fls else 1.0
                if last_bin_local is not None:
                    current_bins.append(last_bin_local)
                    current_w.append(max(float(intensity), 1e-6))  # avoid zero weights
                continue

        # flush final frame if file didn't end with new header
        if inside_frame:
            if len(current_bins) > 0:
                frames_bins.append(current_bins)
                frames_w.append(current_w)
                if (current_secs is not None) and (current_nsecs is not None):
                    frame_times.append(float(current_secs) + float(current_nsecs) * 1e-9)
                else:
                    frame_times.append(np.nan)

    return frames_bins, frames_w, frame_times


def estimate_fps_from_timestamps(frame_times):
    """Estimate FPS from list of timestamps (seconds). Falls back to ASSUME_FPS_IF_UNKNOWN."""
    times = np.array([t for t in frame_times if not np.isnan(t)], dtype=float)
    if len(times) < 2:
        return ASSUME_FPS_IF_UNKNOWN
    dt = np.diff(times)
    dt = dt[dt > 0]  # filter any zeros/negatives if same header repeated
    if len(dt) == 0:
        return ASSUME_FPS_IF_UNKNOWN
    fps = 1.0 / np.median(dt)
    # sanity clamp
    if not np.isfinite(fps) or fps < 1 or fps > 1000:
        return ASSUME_FPS_IF_UNKNOWN
    return float(fps)


def aggregate_frame_doppler(bins, weights, method="weighted_mean"):
    """
    Aggregate a frame's multiple doppler_bin values to a single scalar.
    method: 'weighted_mean' | 'mean' | 'median' | 'max_mag'
    """
    b = np.array(bins, dtype=float)
    if len(b) == 0:
        return 0.0
    if method == "weighted_mean" and weights is not None and len(weights) == len(b):
        w = np.array(weights, dtype=float)
        w = np.clip(w, 1e-6, None)
        return float(np.sum(b * w) / np.sum(w))
    if method == "median":
        return float(np.median(b))
    if method == "max_mag":
        # choose bin with highest |bin|*weight (emphasize fast movers)
        if weights is not None and len(weights) == len(b):
            idx = np.argmax(np.abs(b) * np.array(weights))
        else:
            idx = np.argmax(np.abs(b))
        return float(b[idx])
    return float(np.mean(b))


def build_doppler_curve(frames_bins, frames_w, agg="weighted_mean"):
    """frames_bins: list[list[int]], frames_w: list[list[float]] -> 1D doppler series per frame."""
    series = []
    for bins, w in zip(frames_bins, frames_w):
        series.append(aggregate_frame_doppler(bins, w, method=agg))
    return np.array(series, dtype=float)


# ---------------------------
# 2) Signal cleanup & STFT
# ---------------------------
def clean_curve(curve, medfilt_ks=3, zscore=True):
    c = np.array(curve, dtype=float)
    # Remove DC, mild median filtering to suppress spikes
    c = c - np.median(c)
    if medfilt_ks and medfilt_ks > 1:
        c = medfilt(c, kernel_size=medfilt_ks)
    if zscore:
        c = (c - np.mean(c)) / (np.std(c) + 1e-6)
    return c


def micro_doppler_spectrogram(curve, fps, nperseg=64, noverlap=None, eps=1e-8, log_scale=False, percentile_clip=(1, 99)):
    """
    STFT of doppler-time curve -> (normalized) spectrogram.
    Returns S (H x W) in [0,1].
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if len(curve) < 8:
        # too short for STFT; pad a bit
        pad = max(0, 8 - len(curve))
        curve = np.pad(curve, (0, pad))

    f, t, Z = stft(curve, fs=fps, nperseg=min(nperseg, len(curve)), noverlap=noverlap, boundary=None)
    S = np.abs(Z)

    if log_scale:
        S = np.log1p(S)  # log(1 + |Z|) for contrast

    # Percentile clipping to remove outliers
    lo, hi = np.percentile(S, percentile_clip)
    S = np.clip(S, lo, hi)
    S = (S - S.min()) / (S.max() - S.min() + eps)
    return S, f, t


def resize_spec_to(spec, out_hw=(64, 64)):
    """Resize spectrogram to fixed HxW via torch interpolate (no extra deps)."""
    ten = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    ten = F.interpolate(ten, size=out_hw, mode='bilinear', align_corners=False)
    return ten.squeeze(0).squeeze(0).numpy()


# ---------------------------
# 3) Dataset building (from your PC)
# ---------------------------
def discover_classes(root: Path):
    if CLASSES is not None:
        return CLASSES
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def collect_samples(root: Path, classes):
    """
    Scans ROOT/<class>/*.txt, parses files, computes doppler curve, FPS, spectrogram.
    Returns: specs (list[np.ndarray]), labels (list[int]), class_names (list[str]),
             meta (list[dict]) with file paths and a high-res spec for saving
    """
    specs, labels, meta = [], [], []
    for ci, cname in enumerate(classes):
        class_dir = root / cname
        if not class_dir.is_dir():
            print(f"[WARN] Class folder missing: {class_dir}")
            continue
        txt_files = list(class_dir.glob("*.txt"))
        if not txt_files:
            print(f"[WARN] No .txt in {class_dir}")
            continue

        print(f"[INFO] Processing {cname}: {len(txt_files)} files")
        for fp in txt_files:
            frames_bins, frames_w, frame_times = parse_txt_file(fp)
            if len(frames_bins) == 0:
                continue

            # Build doppler-time curve
            dcurve = build_doppler_curve(frames_bins, frames_w, agg="weighted_mean")
            dcurve = clean_curve(dcurve, medfilt_ks=3, zscore=True)

            fps = estimate_fps_from_timestamps(frame_times)
            # Model-sized STFT (kept at 64x64 for training)
            S_model, f_m, t_m = micro_doppler_spectrogram(dcurve, fps, nperseg=64, noverlap=32, log_scale=True)
            S_model = resize_spec_to(S_model, out_hw=SPEC_SIZE_MODEL)

            # High-res STFT for visualization (no resize, higher nperseg and overlap)
            S_viz, f_v, t_v = micro_doppler_spectrogram(
                dcurve, fps, nperseg=NPERSEG_VIZ, noverlap=NOVERLAP_VIZ, log_scale=True
            )

            specs.append(S_model)
            labels.append(ci)
            meta.append({
                "class_index": ci,
                "class_name": cname,
                "file_path": fp,
                "spec_viz": S_viz,
                "f_v": f_v,
                "t_v": t_v,
            })

    return specs, labels, classes, meta


# ---------------------------
# 3b) NEW: Saving spectrograms
# ---------------------------
def save_spectrogram_image(S, f, t, out_path: Path, title=None, cmap=CMAP, dpi=300):
    plt.figure(figsize=(6, 4), dpi=dpi)
    # Use extent so axes represent approximate time/frequency bins
    extent = [t.min() if len(t) else 0, t.max() if len(t) else S.shape[1],
              f.min() if len(f) else 0, f.max() if len(f) else S.shape[0]]
    plt.imshow(S, cmap=cmap, aspect='auto', origin='lower', extent=extent)
    if title:
        plt.title(title, fontsize=10)
    plt.xlabel("Time")
    plt.ylabel("Doppler (relative bins)")
    cbar = plt.colorbar()
    cbar.set_label("Normalized magnitude", rotation=90)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def save_topN_per_class(meta, N=5):
    """
    Save up to N individual high-res spectrograms per class and also a grid per class.
    """
    # Group by class
    by_class = {}
    for m in meta:
        by_class.setdefault(m["class_name"], []).append(m)

    for cname, items in by_class.items():
        # Take first N (or random N if you prefer; here deterministic)
        chosen = items[:N]
        out_dir = SAVE_DIR / cname
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save individual images
        for i, m in enumerate(chosen, 1):
            src_name = m["file_path"].stem
            out_path = out_dir / f"{i:02d}_{src_name}.png"
            save_spectrogram_image(
                m["spec_viz"], m["f_v"], m["t_v"], out_path,
                title=f"{cname} – {src_name}"
            )

        # Save a grid for the class
        cols = len(chosen)
        if cols == 0:
            continue
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4), dpi=200)
        if cols == 1:
            axes = [axes]
        for ax, m in zip(axes, chosen):
            S, f, t = m["spec_viz"], m["f_v"], m["t_v"]
            extent = [t.min() if len(t) else 0, t.max() if len(t) else S.shape[1],
                      f.min() if len(f) else 0, f.max() if len(f) else S.shape[0]]
            im = ax.imshow(S, cmap=CMAP, aspect='auto', origin='lower', extent=extent)
            ax.set_title(m["file_path"].stem, fontsize=9)
            ax.set_xlabel("Time")
            ax.set_ylabel("Doppler")
        fig.suptitle(f"{cname} — up to {N} examples", fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        grid_out = SAVE_DIR / cname / f"{cname}_grid_{min(N, len(chosen))}.png"
        fig.savefig(grid_out, bbox_inches='tight')
        plt.close(fig)


def save_overview_grid(meta, classes, N=5):
    """
    Save an overview grid of shape (len(classes) x N) if enough samples exist.
    """
    # Build per-class lists
    class_to_items = {c: [] for c in classes}
    for m in meta:
        class_to_items[m["class_name"]].append(m)

    rows = len(classes)
    cols = N
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), dpi=200)

    for r, cname in enumerate(classes):
        items = class_to_items.get(cname, [])
        # pad to N with repeats or blanks to keep grid rectangular
        show = items[:N]
        for c in range(cols):
            ax = axes[r, c] if rows > 1 else axes[c]
            if c < len(show):
                m = show[c]
                S, f, t = m["spec_viz"], m["f_v"], m["t_v"]
                extent = [t.min() if len(t) else 0, t.max() if len(t) else S.shape[1],
                          f.min() if len(f) else 0, f.max() if len(f) else S.shape[0]]
                ax.imshow(S, cmap=CMAP, aspect='auto', origin='lower', extent=extent)
                ax.set_title(f"{cname} | {m['file_path'].stem}", fontsize=8)
            else:
                ax.axis('off')
            if c == 0:
                ax.set_ylabel("Doppler")
            ax.set_xlabel("Time")

    fig.suptitle(f"Overview grid — up to {N} per class", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    grid_out = SAVE_DIR / f"OVERVIEW_{rows}x{cols}.png"
    fig.savefig(grid_out, bbox_inches='tight')
    plt.close(fig)


# ---------------------------
# 4) Torch Dataset & Model
# ---------------------------
class SpecDataset(Dataset):
    def __init__(self, specs, labels):
        self.X = specs
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = self.X[idx]
        # (H,W) -> (1,H,W)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return spec, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        running += float(loss.item())
        pred = out.argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return correct / max(total, 1), running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    preds_all, y_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(1)
        preds_all.append(pred.cpu().numpy())
        y_all.append(yb.cpu().numpy())
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    acc = correct / max(total, 1)
    preds = np.concatenate(preds_all) if preds_all else np.array([])
    ys = np.concatenate(y_all) if y_all else np.array([])
    return acc, preds, ys


# ---------------------------
# 5) MAIN
# ---------------------------
def main():
    assert ROOT.is_dir(), f"Root not found: {ROOT}"
    classes = discover_classes(ROOT)
    print(f"[INFO] Classes: {classes}")

    specs, labels, class_names, meta = collect_samples(ROOT, classes)
    assert len(specs) > 0, "No spectrograms built. Check paths and .txt format."
    X = np.stack(specs, axis=0)  # (N,H,W)
    y = np.array(labels, dtype=int)

    # ------- NEW: Save up to N_PER_CLASS high-res spectrograms per class -------
    save_topN_per_class(meta, N=N_PER_CLASS)
    save_overview_grid(meta, classes=class_names, N=N_PER_CLASS)
    print(f"[INFO] Saved spectrograms to: {SAVE_DIR}")

    # Optional: quick visualization of 1 sample (keep or remove)
    plt.figure(figsize=(5,4), dpi=150)
    plt.imshow(X[0], cmap='jet', aspect='auto', origin='lower')
    plt.title(f"Sample spectrogram — class: {class_names[y[0]]}")
    plt.xlabel("Time (STFT bins)"); plt.ylabel("Relative Doppler (bins)")
    plt.colorbar(); plt.tight_layout(); plt.show()

    # Split and build loaders
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    train_ds = SpecDataset(X_train, y_train)
    test_ds  = SpecDataset(X_test, y_test)
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_ld  = DataLoader(test_ds, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes=len(class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 50
    for ep in range(1, EPOCHS+1):
        tr_acc, tr_loss = train_one_epoch(model, train_ld, opt, loss_fn, device)
        te_acc, preds, ys = evaluate(model, test_ld, device)
        print(f"Epoch {ep:02d} | Train Acc {tr_acc:.3f} Loss {tr_loss:.3f} | Test Acc {te_acc:.3f}")

    # Detailed metrics
    if ys.size > 0:
        import seaborn as sns  # Ensure this is at the top of your script as well

        print("\n📊 [Classification Report]")
        print(classification_report(ys, preds, target_names=class_names))
        
        # 1. Generate the Confusion Matrix data
        cm = confusion_matrix(ys, preds)
        
        # 2. Create the Figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,      # Show the actual numbers in the boxes
            fmt='d',         # Use decimal format for integers
            cmap='Blues',    # Color theme
            xticklabels=class_names, 
            yticklabels=class_names
        )
        
        # 3. Add Labels and Title
        plt.title('Micro-Doppler Activity Confusion Matrix', fontsize=14)
        plt.ylabel('Actual Activity (True Label)', fontsize=12)
        plt.xlabel('Predicted Activity (Model Output)', fontsize=12)
        
        # 4. Save and Show
        # NOTE: Always save BEFORE calling plt.show(), otherwise the saved image will be blank
        plt.savefig("confusion_matrix_final.png", bbox_inches='tight', dpi=300)
        print(f"\n[INFO] Confusion matrix figure saved as 'confusion_matrix_final.png'")
        plt.show()

if __name__ == "__main__":
    main()

