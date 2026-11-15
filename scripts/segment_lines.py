"""Kraken baseline/polygon line segmentation for unlabeled pages, one JSON per page (resumable).

Runs in the separate kraken env (external/envs/kraken), not the uktr env:
  python scripts/segment_lines.py --pages ../unlabel-data/handwritten_essays --shard 0/3
"""
import argparse, glob, json, os, time
from PIL import Image
from kraken import blla

ap = argparse.ArgumentParser()
ap.add_argument('--pages', required=True)
ap.add_argument('--out', default='data/interim/kraken_seg')
ap.add_argument('--shard', default='0/1')
ap.add_argument('--device', default='cuda:0')
a = ap.parse_args()
i, n = map(int, a.shard.split('/'))
files = sorted(glob.glob(os.path.join(a.pages, '*.jpg')))[i::n]
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
for k, f in enumerate(files):
    dst = os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + '.json')
    if os.path.exists(dst):
        continue
    im = Image.open(f).convert('RGB')
    seg = blla.segment(im, device=a.device)
    lines = [{'boundary': [list(map(int, p)) for p in l.boundary], 'baseline': [list(map(int, p)) for p in l.baseline]}
             for l in seg.lines]
    tmp = dst + '.tmp'
    json.dump({'image': os.path.basename(f), 'width': im.width, 'height': im.height, 'lines': lines}, open(tmp, 'w'))
    os.replace(tmp, dst)
    print(f'[{i}] {k+1}/{len(files)} {os.path.basename(f)} {len(lines)} lines {time.time()-t0:.0f}s', flush=True)
print(f'[{i}] done', flush=True)
