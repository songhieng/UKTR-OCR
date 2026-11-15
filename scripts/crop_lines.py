"""Cut Kraken line polygons (scripts/segment_lines.py) into flattened line crops for pseudo_label.py.

Kraken's polygons hug the consonant row, but Khmer upper vowels and subscripts (coeng) reach into the
gap between lines - and so do the next line's vowels and the previous line's coeng. So ink is assigned
per stroke, not per area: every connected ink component goes to the line whose core (consonant) band is
nearest; a component touching two cores is split pixel by pixel. A line crop keeps its polygon plus all
of its own strokes, wherever they hang, and paints out every other line's strokes with the paper colour.
Then every column is shifted so the baseline is straight. One folder per page (= writer_id).

  python scripts/crop_lines.py --pages ../unlabel-data/handwritten_essays
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np


def paper(img, k=15):
    """Background estimate: max-filter removes the (dark) ink, blur smooths it. k must exceed the stroke width."""
    k = int(k) | 1
    return cv2.GaussianBlur(cv2.dilate(img, np.ones((k, k), np.uint8)), (2 * k + 1, 2 * k + 1), 0)


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


def ink_mask(img, bg):
    """Ink = clearly darker than the local paper. Long ruled/grid lines are not ink (they belong to no line)."""
    diff = cv2.subtract(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ink = (diff > max(25, 0.45 * np.percentile(diff, 99.5))).astype(np.uint8)
    L = max(15, img.shape[1] // 8)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) \
        | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    return ink & ~rules


def core_band(ink, bases, S):
    """(top, bottom) offsets of the consonant row relative to the baseline, measured on this page: the dense
    run of ink rows around the baseline, averaged over all lines (vowels above / coeng below are sparser)."""
    offs = np.arange(-int(S), int(0.6 * S) + 1)
    prof = np.zeros(len(offs))
    for bx, by in bases:
        xs = np.arange(int(bx[0]), int(bx[-1]) + 1, 2)
        ys = np.round(np.interp(xs, bx, by)[None, :] + offs[:, None]).astype(int)
        ok = (ys >= 0) & (ys < ink.shape[0])
        prof += np.where(ok, ink[ys.clip(0, ink.shape[0] - 1), xs[None, :].clip(0, ink.shape[1] - 1)], 0).sum(1)
    prof = np.convolve(prof, np.ones(3) / 3, 'same')
    near = (offs >= -0.5 * S) & (offs <= 0.15 * S)              # the row of letters sits on the baseline
    if prof[near].max() == 0:
        return -0.4 * S, 0.0
    peak = int(np.flatnonzero(near)[np.argmax(prof[near])])
    lo = hi = peak
    while lo > 0 and prof[lo - 1] >= 0.5 * prof[peak]:
        lo -= 1
    while hi < len(offs) - 1 and prof[hi + 1] >= 0.5 * prof[peak]:
        hi += 1
    top, bot = float(offs[lo]), float(offs[hi])
    if not (-0.7 * S <= top < bot and -0.25 * S <= bot <= 0.3 * S):   # implausible band: typical Khmer proportions
        return -0.4 * S, 0.0
    return top, bot


def assign_ink(ink, bases, xr, top, bot, S):
    """-> int32 owner map: index into bases for every ink pixel, -1 for no ink / no nearby line."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    owner = np.full(ink.shape, -1, np.int32)
    for c in range(1, n):
        x0, y0, w, h = st[c, :4]
        if h > 2.5 * S:                                         # box frames, margin rules: not a letter
            continue
        cand, dist = [], []
        for k, ((bx, by), (a, b)) in enumerate(zip(bases, xr)):
            if x0 + w < a - 0.5 * S or x0 > b + 0.5 * S:
                continue
            yb = np.interp(np.clip(cen[c][0], a, b), bx, by)
            cand.append(k)
            dist.append(max(0.0, yb + top - (y0 + h), y0 - (yb + bot)))   # gap between stroke and core band
        if not cand or min(dist) > 1.2 * S:
            continue                                            # stray mark far from every line
        touching = [k for k, d in zip(cand, dist) if d <= 0.15 * S]   # reaches (nearly) into this core
        ys, xs = np.where(lab[y0:y0 + h, x0:x0 + w] == c)
        ys, xs = ys + y0, xs + x0
        if len(touching) < 2:
            owner[ys, xs] = cand[int(np.argmin(dist))]
            continue
        # one blob bridging two lines (a coeng touching the next line's vowel): split pixel by pixel
        d = [np.maximum(0, np.maximum(np.interp(xs, *bases[k]) + top - ys, ys - np.interp(xs, *bases[k]) - bot))
             for k in touching]
        owner[ys, xs] = np.array(touching)[np.argmin(np.stack(d), 0)]
    return owner


def crop_page(img, lines, min_frac=0.15, min_h=12):
    """-> list of (line_index, crop). Lines with a baseline shorter than min_frac of the page width are dropped."""
    H, W = img.shape[:2]
    keep = [i for i, l in enumerate(lines) if np.ptp(np.array(l['baseline'])[:, 0]) >= min_frac * W]
    if not keep:
        return []
    bases, xr = [], []
    for i in keep:
        bl = np.array(sorted(lines[i]['baseline']), float)
        bases.append((bl[:, 0], bl[:, 1])); xr.append((bl[0, 0], bl[-1, 0]))
    polys = [np.int32(lines[i]['boundary']) for i in keep]
    # line spacing S: median gap between neighbouring baselines; thickness fallback for one-line pages
    thick = np.median([cv2.contourArea(p) / max(1, b - a) for p, (a, b) in zip(polys, xr)])
    ymid = np.sort([np.interp((a + b) / 2, bx, by) for (bx, by), (a, b) in zip(bases, xr)])
    gaps = np.diff(ymid)
    gaps = gaps[gaps > 0.5 * thick]
    S = float(np.median(gaps)) if len(gaps) else 2.5 * thick
    bg = paper(img, max(15, 0.3 * S))                           # wider than any pen stroke at this line size
    ink = ink_mask(img, bg)
    top, bot = core_band(ink, bases, S)
    owner = assign_ink(ink, bases, xr, top, bot, S)
    pad = max(3, int(0.08 * S)) | 1
    halo = np.ones((5, 5), np.uint8)
    out = []
    for j, i in enumerate(keep):
        region = (owner == j).astype(np.uint8)
        cv2.fillPoly(region, [polys[j]], 1)
        region = cv2.dilate(region, np.ones((pad, pad), np.uint8))
        region &= ~cv2.dilate(((owner >= 0) & (owner != j)).astype(np.uint8), halo)   # other lines' strokes
        ys, xs = np.where(region)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = region[y0:y1, x0:x1]
        sub = np.where(m[..., None].astype(bool), img[y0:y1, x0:x1], bg[y0:y1, x0:x1])
        bl = [(x - x0, y - y0) for x, y in lines[i]['baseline']]
        crop, _ = flatten(sub, m, bl, 0)
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

    # Two lines with tight (Kraken-like) polygons around their consonant rows (separate glyph blocks).
    # A's coeng hangs below A's polygon, B's upper vowel rises above B's polygon - both in the gap - and
    # one coeng of A touches a vowel of B (one blob that has to be split).
    img = np.full((300, 600, 3), 230, np.uint8)
    parts = {k: np.zeros((300, 600), np.uint8) for k in ('A', 'B', 'coengA', 'vowelB', 'touchA', 'touchB')}
    D = lambda k, x0, y0, x1, y1: cv2.rectangle(parts[k], (x0, y0), (x1, y1), 1, -1)
    for x in range(50, 540, 30):
        D('A', x, 80, x + 20, 100); D('B', x, 160, x + 20, 180)
    D('coengA', 200, 104, 216, 124)                              # starts 4 px below A's baseline
    D('vowelB', 290, 136, 306, 154)                              # ends 6 px above B's consonants
    D('touchA', 410, 104, 418, 126); D('touchB', 410, 127, 418, 154)
    for m in parts.values():
        img[m > 0] = 20
    poly = lambda y: [[50, y - 22], [545, y - 22], [545, y + 2], [50, y + 2]]
    lines = [{'baseline': [[50, 100], [545, 100]], 'boundary': poly(100)},
             {'baseline': [[50, 180], [545, 180]], 'boundary': poly(180)}]
    (_, A), (_, B) = crop_page(img, lines)
    n = lambda c: int((c[..., 0] < 100).sum())                  # dark pixels in a crop
    s = {k: int(m.sum()) for k, m in parts.items()}
    want_a, want_b = s['A'] + s['coengA'] + s['touchA'], s['B'] + s['vowelB'] + s['touchB']
    assert abs(n(A) - want_a) <= 0.03 * want_a, ('A should keep its coeng, not B vowel', n(A), want_a)
    assert abs(n(B) - want_b) <= 0.03 * want_b, ('B should keep its vowel, not A coeng', n(B), want_b)


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
