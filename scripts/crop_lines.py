"""Cut Kraken line polygons (scripts/segment_lines.py) into flattened line crops for pseudo_label.py.

Per line: drop short fragments, grow the tight Kraken polygon so Khmer upper vowels and subscripts survive,
remove the neighbouring lines' polygons, fill everything else with the paper colour, then shift every
column so the baseline is straight. One folder per page (= writer_id for pseudo_label.py).

  python scripts/crop_lines.py --pages ../unlabel-data/handwritten_essays
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np


def paper(img):
    """Background estimate: max-filter removes the (dark) ink, blur smooths it."""
    return cv2.GaussianBlur(cv2.dilate(img, np.ones((15, 15), np.uint8)), (31, 31), 0)


def flatten(img, mask, baseline, x0):
    """Shift each column so the baseline sits on one row; return (crop, mask) trimmed to the mask."""
    bl = np.array(sorted(baseline), float)
    xs = np.arange(x0, x0 + img.shape[1])
    by = np.interp(xs, bl[:, 0], bl[:, 1])                      # constant beyond the ends
    rows = np.where(mask.any(1))[0]
    up = int(np.ceil(np.max(by) - rows[0])) + 1                 # covers the mask in every column
    down = int(np.ceil(rows[-1] - np.min(by))) + 1
    H = up + down
    map_x = np.tile(np.arange(img.shape[1], dtype=np.float32), (H, 1))
    map_y = (by[None, :] - up + np.arange(H)[:, None]).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    r, c = np.where(m)
    return out[r.min():r.max() + 1, c.min():c.max() + 1], m[r.min():r.max() + 1, c.min():c.max() + 1]


def crop_page(img, lines, min_frac=0.15, grow=0.3, min_h=12):
    """-> list of (line_index, crop). Lines with a baseline shorter than min_frac of the page width are dropped."""
    H, W = img.shape[:2]
    keep = [i for i, l in enumerate(lines) if np.ptp(np.array(l['baseline'])[:, 0]) >= min_frac * W]
    if not keep:
        return []
    polys = [np.int32(l['boundary']) for l in lines]
    # thickness = area / baseline length (the polygon's y-extent would include the line's slope)
    line_h = np.median([cv2.contourArea(polys[i]) / max(1, np.ptp(np.array(lines[i]['baseline'])[:, 0])) for i in keep])
    ky, kx = max(3, int(grow * line_h)) | 1, max(3, int(0.1 * line_h)) | 1
    others = np.zeros((H, W), np.uint8)
    for i in keep:
        cv2.fillPoly(others, [polys[i]], 1)
    bg = paper(img)
    out = []
    for i in keep:
        own = np.zeros((H, W), np.uint8)
        cv2.fillPoly(own, [polys[i]], 1)
        m = cv2.dilate(own, np.ones((ky, kx), np.uint8)) & ~(others & ~own)   # grow, minus other lines
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.where(m[y0:y1, x0:x1, None].astype(bool), img[y0:y1, x0:x1], bg[y0:y1, x0:x1])
        bl = [(x - x0, y - y0) for x, y in lines[i]['baseline']]
        crop, _ = flatten(sub, m[y0:y1, x0:x1], bl, 0)
        if crop.shape[0] >= min_h and crop.shape[1] >= 2 * crop.shape[0]:
            out.append((i, crop))
    return out


def demo():
    """Self-check: a sloped dark bar cut from a page comes out flat and short."""
    img = np.full((200, 400, 3), 230, np.uint8)
    base = [[20, 60], [380, 140]]                              # slope of ~12.5 degrees
    poly = [[20, 45], [380, 125], [380, 145], [20, 65]]
    cv2.fillPoly(img, [np.int32(poly)], (20, 20, 20))
    [(i, crop)] = crop_page(img, [{'baseline': base, 'boundary': poly}])
    assert crop.shape[0] < 40, crop.shape                       # unflattened bbox would be ~100 px tall
    dark = (crop[..., 0] < 100).mean(1)
    assert dark.max() > 0.9, dark                               # the bar fills whole rows once flattened


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pages', help='folder with the page images')
    ap.add_argument('--seg', default='data/interim/kraken_seg')
    ap.add_argument('--out', default='data/raw/unlabeled')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo:
        demo()
        print('ok')
        raise SystemExit
    rows = []
    for j in sorted(glob.glob(os.path.join(a.seg, '*.json'))):
        seg = json.load(open(j))
        page = os.path.splitext(seg['image'])[0]
        img = cv2.imread(os.path.join(a.pages, seg['image']))
        d = os.path.join(a.out, page)
        os.makedirs(d, exist_ok=True)
        for i, crop in crop_page(img, seg['lines']):
            p = os.path.join(d, f'{page}_L{i:03d}.png')
            cv2.imwrite(p, crop)
            rows.append([os.path.relpath(p, a.out).replace(os.sep, '/'), page, i, crop.shape[1], crop.shape[0]])
    with open(os.path.join(a.out, 'lines.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['image_path', 'page', 'line', 'width', 'height']] + rows)
    print(f'{len(rows)} lines from {len({r[1] for r in rows})} pages -> {a.out}')
