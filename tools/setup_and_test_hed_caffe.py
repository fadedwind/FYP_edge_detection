"""Setup HED Caffe deploy prototxt and run OpenCV DNN inference using a Caffe model.

This script will:
- download the HED deploy prototxt from the original repo into `models/hed_deploy.prototxt` if missing
- look for a Caffe model named `hed_pretrained_bsds.caffemodel` under `models/` or repo root
- if found, use OpenCV DNN to run HED on a few BSDS500 images and save outputs to output/hed_caffe_test

Run with the project's Python environment, e.g.:
.venv\Scripts\python.exe tools/setup_and_test_hed_caffe.py
"""
import os
import sys
from pathlib import Path
import urllib.request
import time

repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / 'models'
models_dir.mkdir(exist_ok=True)

prototxt_url = 'https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt'
prototxt_path = models_dir / 'hed_deploy.prototxt'

print('Project root:', repo_root)
print('Ensuring prototxt at:', prototxt_path)
if not prototxt_path.exists():
    try:
        print('Downloading prototxt from', prototxt_url)
        urllib.request.urlretrieve(prototxt_url, str(prototxt_path))
        print('Saved prototxt.')
    except Exception as e:
        print('Failed to download prototxt:', e)
        sys.exit(2)
else:
    print('Prototxt already present.')

# locate caffemodel
caffe_candidates = list(models_dir.glob('*.caffemodel'))
if not caffe_candidates:
    # also check repo root
    caffe_candidates = list(repo_root.glob('*.caffemodel'))

if not caffe_candidates:
    print('No .caffemodel found in models/ or repo root. Please place hed_pretrained_bsds.caffemodel into models/ and rerun.')
    print('Original official download: https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel')
    sys.exit(3)

caffemodel_path = caffe_candidates[0]
print('Using Caffe model:', caffemodel_path)

try:
    import cv2
except Exception as e:
    print('OpenCV not available or import failed:', e)
    sys.exit(4)

# prepare BSDS images path (same as earlier)
bsds_rel = Path('..') / 'BSR_bsds500' / 'BSR' / 'BSDS500' / 'data' / 'images'
bsds_dir = (repo_root / bsds_rel).resolve()
if not bsds_dir.exists():
    print('BSDS images path not found at', bsds_dir)
    sys.exit(5)

images = [p for p in sorted(bsds_dir.iterdir()) if p.suffix.lower() in ('.jpg', '.png', '.jpeg')][:5]
if not images:
    print('No images found in BSDS folder')
    sys.exit(6)

# load net
print('Loading network via OpenCV DNN...')
net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

out_dir = repo_root / 'output' / 'hed_caffe_test'
out_dir.mkdir(parents=True, exist_ok=True)

results = []
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print('Failed to read', img_path)
        continue
    # prepare blob: size 500x500 as in prototxt
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(500, 500), mean=(104.00698793,116.66876762,122.67891434), swapRB=False, crop=False)
    net.setInput(blob)
    t0 = time.time()
    out = net.forward()
    dur = time.time() - t0
    # out shape: (1,1,H,W) or (1,1,500,500)
    out_map = out[0,0,:,:]
    # normalize to 0-255
    out_map = (out_map * 255.0).clip(0,255).astype('uint8')
    # resize back to original image size
    out_map_resized = cv2.resize(out_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    save_path = out_dir / f'hed_caffe_{img_path.name}.png'
    cv2.imwrite(str(save_path), out_map_resized)
    results.append((img_path.name, dur, int(out_map_resized.min()), int(out_map_resized.max()), float(out_map_resized.mean())))
    print(f'Processed {img_path.name} in {dur:.3f}s -> saved {save_path}')

print('\nSummary:')
for r in results:
    print(r)
print('Outputs saved to', out_dir)
print('Done.')
