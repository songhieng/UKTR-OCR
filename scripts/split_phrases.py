"""Split line crops (scripts/crop_lines.py) into short phrases at the widest ink gaps.

The real handwriting we train on (real_hw, kiteaether) is 1-4 words, about 3-4x wider than tall; full essay
lines are ~13x. Cuts go at the widest whitespace gap inside each [min_aspect, max_aspect] window, so a
phrase never ends inside a stroke. Ruled lines and grid lines are removed before measuring gaps.

  python scripts/split_phrases.py                 # data/raw/unlabeled -> data/raw/unlabeled_phrases
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np


def ink_columns(gray):
    """Per-column ink mask, ignoring long horizontal rules and full-height vertical grid lines."""
    h = gray.shape[0]
    k = max(3, h // 3)
    bg = cv2.GaussianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), (0, 0), k / 2)   # paper, ink removed
    diff = cv2.subtract(bg, gray)                    # how much darker than the local paper
    ink = (diff > max(25, 0.45 * np.percentile(diff, 99.5))).astype(np.uint8)   # mask seams/texture stay below
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(3, int(1.5 * h))), np.uint8))
    grid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(3, int(0.85 * h)), 1), np.uint8))
    ink = ink & ~rules & ~grid
    core = ink[int(0.15 * h):int(0.85 * h)]          # the crop_lines mask edge lives in the outer rows
    return core.sum(0) >= max(2, int(0.03 * h))


def cut_points(inked, h, min_aspect=2.5, max_aspect=7.0, min_gap=0.12):
    """Column indices to cut at (gap centres). Greedy: from the current start, cut at the widest gap whose
    centre lies in [start + min_w, start + max_w]; if there is none, at the first gap after that window."""
    gaps, x, W = [], 0, len(inked)
    while x < W:                                                # runs of empty columns between ink
        if inked[x]:
            x += 1; continue
        s = x
        while x < W and not inked[x]:
            x += 1
        if s > 0 and x < W and x - s >= min_gap * h:
            gaps.append(((s + x) // 2, x - s))
    min_w, max_w = min_aspect * h, max_aspect * h
    cuts, start = [], 0
    while True:
        cand = [g for g in gaps if start + min_w <= g[0] <= start + max_w]
        if not cand:
            cand = [g for g in gaps if g[0] > start + max_w][:1]
        if not cand:
            return cuts
        c = max(cand, key=lambda g: g[1])[0]
        if W - c < min_w / 2:                                   # tail too short: leave it on this phrase
            return cuts
        cuts.append(c); start = c


def split_line(img, pad=0.1):
    """-> list of phrase crops, trimmed to their ink plus a little padding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, W = gray.shape
    inked = ink_columns(gray)
    edges = [0] + cut_points(inked, h) + [W]
    out, p = [], int(pad * h)
    for a, b in zip(edges, edges[1:]):
        cols = np.where(inked[a:b])[0]
        if len(cols) == 0:
            continue
        x0, x1 = max(a, a + cols[0] - p), min(b, a + cols[-1] + 1 + p)
        if x1 - x0 >= 0.8 * h:
            out.append(img[:, x0:x1])
    return out


def demo():
    """Self-check: 10 ink blobs, h=40, separated by 8 px gaps except two 30 px gaps -> cuts land in the big gaps."""
    img = np.full((40, 600, 3), 235, np.uint8)
    x, big = 10, {3, 7}
    for i in range(10):
        cv2.rectangle(img, (x, 12), (x + 45, 28), (30, 30, 30), -1)
        x += 45 + (30 if i in big else 8)
    phrases = split_line(img)
    assert len(phrases) == 3, [p.shape for p in phrases]          # cut in both 30 px gaps, nowhere else
    assert all(2.5 * 40 * 0.8 <= p.shape[1] <= 7 * 40 for p in phrases), [p.shape for p in phrases]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines', default='data/raw/unlabeled')
    ap.add_argument('--out', default='data/raw/unlabeled_phrases')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo:
        demo()
        print('ok')
        raise SystemExit
    rows = []
    for f in sorted(glob.glob(os.path.join(a.lines, '*', '*.png'))):
        page = os.path.basename(os.path.dirname(f))
        d = os.path.join(a.out, page)
        os.makedirs(d, exist_ok=True)
        for k, ph in enumerate(split_line(cv2.imread(f))):
            p = os.path.join(d, f'{os.path.splitext(os.path.basename(f))[0]}_P{k:02d}.png')
            cv2.imwrite(p, ph)
            rows.append([os.path.relpath(p, a.out).replace(os.sep, '/'), page, os.path.basename(f), ph.shape[1], ph.shape[0]])
    with open(os.path.join(a.out, 'phrases.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows([['image_path', 'page', 'line_image', 'width', 'height']] + rows)
    print(f'{len(rows)} phrases from {len({r[2] for r in rows})} lines -> {a.out}')
