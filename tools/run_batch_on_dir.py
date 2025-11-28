"""Run batch edge-detection evaluation (ODS/OIS/PR) on a directory using project helpers.

Usage:
  .venv\Scripts\python.exe tools/run_batch_on_dir.py <images_dir> [algorithm]

Saves outputs to output/<timestamp>/ containing per-image *_edge.png, CSV report and PR curve.
"""
import sys
from pathlib import Path
import time
import os
import numpy as np

if len(sys.argv) < 2:
    print('Usage: run_batch_on_dir.py <images_dir> [algorithm] [--bsds-root <BSDS_DATA_DIR>] [--tolerance <pixels>]')
    sys.exit(1)

images_dir = Path(sys.argv[1])
if not images_dir.exists():
    print('Provided path does not exist:', images_dir)
    sys.exit(2)

algorithm = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'HED'

# optional: --bsds-root <dir>
bsds_root = None
if '--bsds-root' in sys.argv:
    try:
        idx = sys.argv.index('--bsds-root')
        bsds_root = Path(sys.argv[idx + 1])
        if not bsds_root.exists():
            print('Provided --bsds-root does not exist:', bsds_root)
            sys.exit(2)
    except Exception:
        print('Usage: run_batch_on_dir.py <images_dir> [algorithm] [--bsds-root <BSDS_DATA_DIR>] [--tolerance <pixels>]')
        sys.exit(1)

# optional: --tolerance <pixels> (integer, default 2)
tolerance = 2
if '--tolerance' in sys.argv:
    try:
        idx = sys.argv.index('--tolerance')
        tolerance = int(sys.argv[idx + 1])
        if tolerance < 0 or tolerance > 20:
            print('Warning: tolerance out of typical range (0-20). Using:', tolerance)
    except Exception:
        print('Usage: run_batch_on_dir.py <images_dir> [algorithm] [--bsds-root <BSDS_DATA_DIR>] [--tolerance <pixels>]')
        sys.exit(1)

repo_root = Path(__file__).resolve().parents[1]
import cv2
from datetime import datetime
from PIL import Image

# Import helpers from project
sys.path.insert(0, str(repo_root))
try:
    from edge_detection import compute_edge_strength, get_reference_edge, frameWidth, frameHeight
except Exception as e:
    print('Failed to import project helpers:', e)
    sys.exit(3)

out_root = repo_root / 'output'
out_root.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_output_dir = out_root / timestamp
run_output_dir.mkdir(parents=True, exist_ok=True)

exts = ('.jpg', '.jpeg', '.png', '.bmp')
files = [p for p in sorted(images_dir.rglob('*')) if p.suffix.lower() in exts]
if not files:
    print('No images found in', images_dir)
    sys.exit(4)

print(f'Found {len(files)} images, algorithm={algorithm}, output -> {run_output_dir}')
if bsds_root:
    print(f'Using BSDS groundTruth from: {bsds_root}')
print(f'Pixel tolerance (for boundary matching): {tolerance}px')

# helper: load BSDS groundTruth for a given image name; returns uint8 0/255 map resized to HxW
def load_bsds_reference(image_path: Path, H: int, W: int):
    # image name without extension
    base = image_path.stem
    import scipy.io as sio
    # try in train/val/test
    for sub in ['train', 'val', 'test']:
        mat_path = bsds_root / 'groundTruth' / sub / f'{base}.mat'
        if mat_path.exists():
            try:
                mat = sio.loadmat(str(mat_path))
                gt = mat.get('groundTruth', None)
                if gt is None or gt.size == 0:
                    break
                union = np.zeros((H, W), dtype=np.float32)
                for i in range(gt.size):
                    entry = gt[0, i]
                    try:
                        bmap = entry['Boundaries'][0, 0]
                    except Exception:
                        bmap = entry[0, 0]['Boundaries'][0, 0]
                    if bmap.dtype != np.float32:
                        bmap = bmap.astype(np.float32)
                    if bmap.shape[0] != H or bmap.shape[1] != W:
                        bmap = cv2.resize(bmap, (W, H), interpolation=cv2.INTER_NEAREST)
                    union = np.maximum(union, bmap)
                ref = (union > 0.5).astype(np.uint8) * 255
                return ref
            except Exception as _e:
                print('Failed to load BSDS GT for', image_path.name, '->', _e)
                break
    return None

# parameters (use defaults reasonable for batch)
blur_ksize = 7
sobel_ksize = 3

thresholds = list(range(0, 256, 1))
thr_TP_sum = np.zeros(len(thresholds), dtype=np.float64)
thr_FP_sum = np.zeros(len(thresholds), dtype=np.float64)
thr_FN_sum = np.zeros(len(thresholds), dtype=np.float64)

per_image_best = []

start = time.time()
for idx, fp in enumerate(files, 1):
    img = cv2.imread(str(fp))
    if img is None:
        print('skip', fp)
        continue
    img = cv2.resize(img, (frameWidth, frameHeight))
    if bsds_root:
        reference = load_bsds_reference(fp, frameHeight, frameWidth)
        if reference is None:
            # fallback to Canny pseudo reference
            reference = get_reference_edge(img)
    else:
        reference = get_reference_edge(img)
    edge_strength = compute_edge_strength(img, algorithm, blur_ksize, sobel_ksize)

    best_f1 = -1.0
    best_prec = best_rec = best_thr = 0

    # prepare tolerance kernel and reference masks
    ref_mask = (reference > 0)
    if tolerance > 0:
        ksize = 2 * tolerance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        ref_dilated = cv2.dilate(ref_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        ref_dilated = ref_mask

    for i, t in enumerate(thresholds):
        _, detected = cv2.threshold(edge_strength, t, 255, cv2.THRESH_BINARY)
        det_mask = (detected > 0)
        if tolerance > 0:
            # symmetric tolerant matching:
            # TP: detected that hit dilated GT
            # FP: detected that do not hit dilated GT
            # FN: GT that are not hit by dilated detected
            det_dilated = cv2.dilate(det_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            TP = int(np.logical_and(det_mask, ref_dilated).sum())
            FP = int(np.logical_and(det_mask, np.logical_not(ref_dilated)).sum())
            FN = int(np.logical_and(ref_mask, np.logical_not(det_dilated)).sum())
        else:
            TP = int(np.logical_and(det_mask, ref_mask).sum())
            FP = int(np.logical_and(det_mask, np.logical_not(ref_mask)).sum())
            FN = int(np.logical_and(np.logical_not(det_mask), ref_mask).sum())

        thr_TP_sum[i] += TP
        thr_FP_sum[i] += FP
        thr_FN_sum[i] += FN

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = t

    per_image_best.append((fp.name, best_thr, best_prec, best_rec, best_f1))
    try:
        _, detected_final = cv2.threshold(edge_strength, best_thr, 255, cv2.THRESH_BINARY)
        save_path = run_output_dir / f"{fp.stem}_edge.png"
        cv2.imwrite(str(save_path), detected_final)
    except Exception:
        pass

    if idx % 50 == 0 or idx == len(files):
        print(f'Processed {idx}/{len(files)}')

# compute dataset-level precision/recall/f1 per threshold
denom_prec = thr_TP_sum + thr_FP_sum
precision_mean = np.zeros_like(thr_TP_sum)
nonzero = denom_prec > 0
precision_mean[nonzero] = thr_TP_sum[nonzero] / denom_prec[nonzero]

denom_rec = thr_TP_sum + thr_FN_sum
recall_mean = np.zeros_like(thr_TP_sum)
nonzero_r = denom_rec > 0
recall_mean[nonzero_r] = thr_TP_sum[nonzero_r] / denom_rec[nonzero_r]

mean_f1_per_thr = np.zeros_like(precision_mean)
nonzero_f = (precision_mean + recall_mean) > 0
mean_f1_per_thr[nonzero_f] = 2 * precision_mean[nonzero_f] * recall_mean[nonzero_f] / (
    precision_mean[nonzero_f] + recall_mean[nonzero_f]
)

best_idx = int(np.argmax(mean_f1_per_thr))
ods_thr = thresholds[best_idx]
ods_prec = float(precision_mean[best_idx])
ods_rec = float(recall_mean[best_idx])
ods_f1 = float(mean_f1_per_thr[best_idx])

ois_prec = np.mean([x[2] for x in per_image_best])
ois_rec = np.mean([x[3] for x in per_image_best])
ois_f1 = np.mean([x[4] for x in per_image_best])

elapsed = time.time() - start
print(f'Done. Processed {len(files)} images in {elapsed:.1f}s')
print(f'ODS_thr={ods_thr}, ODS_F1={ods_f1:.3f}, OIS_F1={ois_f1:.3f}')

# save CSV
import csv
out_csv = run_output_dir / f'batch_metrics_{algorithm}.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.writer(cf)
    writer.writerow(["filename", "best_threshold", "best_precision", "best_recall", "best_f1"])
    for row in per_image_best:
        writer.writerow(row)
    writer.writerow([])
    writer.writerow(["ODS_threshold", ods_thr])
    writer.writerow(["ODS_precision", round(ods_prec, 3)])
    writer.writerow(["ODS_recall", round(ods_rec, 3)])
    writer.writerow(["ODS_f1", round(ods_f1, 3)])
    writer.writerow(["OIS_precision", round(ois_prec, 3)])
    writer.writerow(["OIS_recall", round(ois_rec, 3)])
    writer.writerow(["OIS_f1", round(ois_f1, 3)])

# plot PR curve
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    precision_mean = np.clip(precision_mean, 0.0, 1.0)
    recall_mean = np.clip(recall_mean, 0.0, 1.0)
    idxs = np.argsort(recall_mean)
    recall_sorted = recall_mean[idxs]
    precision_sorted = precision_mean[idxs]

    unique_recalls = []
    max_precisions = []
    for r, p in zip(recall_sorted, precision_sorted):
        if len(unique_recalls) == 0 or r != unique_recalls[-1]:
            unique_recalls.append(r)
            max_precisions.append(p)
        else:
            if p > max_precisions[-1]:
                max_precisions[-1] = p

    recall_sorted = np.array(unique_recalls)
    precision_sorted = np.array(max_precisions)

    if recall_sorted.max() - recall_sorted.min() < 1e-6:
        recall_fine = recall_sorted
        precision_fine = precision_sorted
    else:
        recall_fine = np.linspace(recall_sorted.min(), recall_sorted.max(), 512)
        precision_fine = np.interp(recall_fine, recall_sorted, precision_sorted)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(recall_fine, precision_fine, '-', linewidth=1.5)
    ax.plot(recall_sorted, precision_sorted, 'o', markersize=3, alpha=0.6)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve ({algorithm})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.4)
    pr_path = run_output_dir / f'PR_curve_{algorithm}.png'
    fig.tight_layout()
    fig.savefig(str(pr_path), dpi=150)
    plt.close(fig)
    print('Saved PR curve to', pr_path)
except Exception as e:
    print('Failed to plot PR curve:', e)

print('Saved CSV to', out_csv)
print('Saved edges to', run_output_dir)
