"""Run HED via OpenCV DNN on a user-specified BSDS500 images directory.

Usage:
.venv\Scripts\python.exe tools/run_hed_on_user_bsds.py "F:\path\to\BSDS500\data\images"
"""
import sys
from pathlib import Path
import time

if len(sys.argv) < 2:
    print('Usage: run_hed_on_user_bsds.py <bsds_images_dir>')
    sys.exit(1)

bsds_dir = Path(sys.argv[1])
if not bsds_dir.exists():
    print('Provided path does not exist:', bsds_dir)
    sys.exit(2)

repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / 'models'
prototxt_path = models_dir / 'hed_deploy.prototxt'
# prefer caffemodel in models
caffemodel_candidates = list(models_dir.glob('*.caffemodel'))
if not caffemodel_candidates:
    # fallback to repo root
    caffemodel_candidates = list(repo_root.glob('*.caffemodel'))
if not caffemodel_candidates:
    print('No .caffemodel found in models/ or repo root. Place hed_pretrained_bsds.caffemodel into models/ and rerun.')
    sys.exit(3)

caffemodel_path = caffemodel_candidates[0]
print('Using prototxt:', prototxt_path)
print('Using caffemodel:', caffemodel_path)

try:
    import cv2
except Exception as e:
    print('OpenCV import failed:', e)
    sys.exit(4)

images = [p for p in sorted(bsds_dir.iterdir()) if p.suffix.lower() in ('.jpg', '.png', '.jpeg')][:5]
if not images:
    print('No images found in provided folder')
    sys.exit(5)

# Use cached net from edge_detection if available to avoid repeated loads
try:
    # import local module
    import sys
    sys.path.insert(0, str(repo_root))
    from edge_detection import get_hed_net
    net = get_hed_net()
    if net is None:
        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
except Exception:
    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

out_dir = repo_root / 'output' / 'hed_caffe_test_user'
out_dir.mkdir(parents=True, exist_ok=True)

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print('Failed to read', img_path)
        continue
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(500,500), mean=(104.00698793,116.66876762,122.67891434), swapRB=False, crop=False)
    net.setInput(blob)
    t0 = time.time()
    out = net.forward()
    dur = time.time() - t0
    out_map = out[0,0,:,:]
    out_map = (out_map * 255.0).clip(0,255).astype('uint8')
    out_map_resized = cv2.resize(out_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    save_path = out_dir / f'hed_user_{img_path.name}.png'
    cv2.imwrite(str(save_path), out_map_resized)
    print(f'Processed {img_path.name} in {dur:.3f}s -> saved {save_path} min={int(out_map_resized.min())} max={int(out_map_resized.max())} mean={float(out_map_resized.mean()):.2f}')

print('Done. Outputs in', out_dir)
