"""Pick essay phrases the model can NOT read confidently, for hand correction (= real TRAIN labels).

Self-training only learns from lines the model already reads (pseudo_label.py keeps CTC == TR and low NLL).
This picks from the rest: rejected phrases, spread over pages (writers), minus the least confident 10%
(mostly printed headers, scribbles, cut-off crops). Writes data/raw/essay_hard/{images/, labels.csv,
label.html}: same page as the test set, own browser storage, exports split=train.

  python scripts/make_hard_set.py --checkpoint checkpoints/essay_r2/uktr_stage2_best_infer.pt
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path


def pick(rows, n, per_page, seed=0):
    """rows: dicts with page. -> at most n rows, at most per_page from any page, pages as even as possible."""
    rnd = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r['page'], []).append(r)
    for v in by.values():
        rnd.shuffle(v)
    out, k = [], 0
    while len(out) < n and k < per_page:                         # round-robin: 1 per page, then 2, ...
        layer = [v[k] for v in by.values() if len(v) > k]
        rnd.shuffle(layer)
        out += layer[:n - len(out)]
        k += 1
    return out


def demo():
    rows = [{'page': p, 'i': i} for p, m in (('a', 10), ('b', 1), ('c', 3)) for i in range(m)]
    got = pick(rows, 5, per_page=2)
    assert len(got) == 5 and sum(r['page'] == 'b' for r in got) == 1     # every page before seconds
    assert max(sum(r['page'] == p for r in got) for p in 'ac') == 2
    assert len(pick(rows, 100, per_page=2)) == 5                         # 2 + 1 + 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--images', default='data/raw/unlabeled_phrases')
    ap.add_argument('--out', default='data/raw/essay_hard')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--per-page', type=int, default=2)
    ap.add_argument('--max-nll', type=float, default=0.05, help='the pseudo-label threshold: above it = rejected')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo:
        demo(); print('ok'); raise SystemExit
    out = Path(a.out)
    if (out / 'labels.csv').exists():                            # re-picking would orphan saved browser labels
        sys.exit(f'{out}/labels.csv exists - delete {out} to pick a new set')

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
    from make_test_set import page
    from pseudo_label import read_all
    from src.models.checkpoint import load_stage1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tok, ck = load_stage1(Path(a.checkpoint).resolve(), device=dev); model.eval()
    root = Path(a.images)
    paths = sorted(root.rglob('*.png'))
    df = read_all(model, tok, ck, paths, dev)
    rej = df[~(df.agree & (df.nll_per_tok <= a.max_nll)) & (df.text.str.strip() != '')]
    rej = rej[rej.nll_per_tok <= rej.nll_per_tok.quantile(0.9)]
    rows = [{'src': Path(p), 'page': Path(p).parent.name, 'guess': t} for p, t in zip(rej.path, rej.text)]
    chosen = sorted(pick(rows, a.n, a.per_page), key=lambda r: (r['page'], r['src'].name))
    for r in chosen:
        d = out / 'images' / r['page']
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(r['src'], d / r['src'].name)
    items = [{'image_path': f"{r['page']}/{r['src'].name}", 'page': r['page'], 'writer_id': r['page'],
              'guess': r['guess']} for r in chosen]
    with open(out / 'labels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image_path', 'text', 'writer_id', 'split', 'model_guess'])
        w.writerows([[r['image_path'], '', r['writer_id'], 'train', r['guess']] for r in items])
    (out / 'label.html').write_text(page(items, 'Hard essay phrases', 'essay_hard_labels_v1', 'train'), encoding='utf-8')
    print(f'{len(df)} phrases, {len(rej)} rejected and readable -> {len(items)} picked from '
          f'{len({r["page"] for r in items})} pages -> {out}/label.html')
