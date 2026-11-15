"""Hold out essay pages as a hand-transcribed test set, before any pseudo-labelling.

Moves the line crops of TEST pages to data/raw/essay_test/images/<page>/ and of BUFFER pages (rest of the
same post, or the gallery neighbours of a test page = likely the same writer) to data/raw/heldout_unlabeled/,
so neither reaches pseudo_label.py. Pre-fills each line with the current model's reading and writes
label.html: confirm/correct each line, then export labels.csv (only confirmed lines get text).

  python scripts/make_test_set.py --checkpoint ../external/checkpoints/thai_b2/uktr_stage2_best_infer.pt
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Pages to transcribe, picked for variety (lined/grid/exam form/scan/photo, every source).
TEST = ['tg_543', 'tg_115', 'mfy1_0148_122148447561111440', 'mfy2_122134110561111440', 'kh400_1000671922875369',
        'ess1_essay_777_page_01', 'ess1_essay_777_page_02', 'ess2_exam_answer_sheets_post_931_932',
        'ess2_exam_answer_sheets_post_931_935', 'ess3_notebook_writing_page_02']
# Same post as a test page -> same writer; held out whole.
POSTS = ['ess1_essay_777', 'ess2_exam_answer_sheets_post_931', 'ess3_notebook_writing']
# Pages with lines already transcribed in data/raw/essay_test_extra (labelled on the 2026-11-15 demo page):
# held out with their neighbours like TEST, but not put on label.html again.
EXTRA = ['mfy1_0047_122152710309111440', 'mfy1_0138_122148590961111440', 'mfy1_0235_122145472887111440',
         'mfy1_0594_122137691487111440', 'mfy2_122131512033111440', 'mfy2_122135800635111440',
         'tg_116', 'tg_281', 'tg_524']


def writer(page):
    return next((p for p in POSTS if page.startswith(p)), page)


def heldout(pages):
    """-> set of buffer pages: the rest of each test post + gallery neighbours (+-1) of other test pages."""
    def key(p):                                                  # tg_99 < tg_105: sort tg numerically
        return (p.split('_')[0], int(p[3:]) if p.startswith('tg_') else 0, p)
    pages = set(pages) | set(TEST) | set(EXTRA)                  # a duplicate would shift the +-1 slice
    buf ={p for p in pages if any(p.startswith(q) for q in POSTS)} | set(EXTRA)
    for t in TEST + EXTRA:
        if writer(t) != t:
            continue
        same = sorted((p for p in pages if p.split('_')[0] == t.split('_')[0]), key=key)
        i = same.index(t)
        buf |= set(same[max(0, i - 1):i + 2])
    return buf - set(TEST)


def demo():
    pages = ['tg_99', 'tg_105', 'tg_115', 'tg_116', 'tg_543', 'tg_600', 'ess1_essay_777_page_03', 'ess1_essay_700_page_01',
             'mfy1_0147_x', 'mfy1_0148_122148447561111440', 'mfy1_0149_y', 'kh400_1000671922875369']
    b = heldout(pages + TEST)
    assert {'tg_105', 'tg_116', 'tg_600', 'mfy1_0147_x', 'mfy1_0149_y', 'ess1_essay_777_page_03'} <= b, b
    assert not b & {'tg_99', 'ess1_essay_700_page_01'} and not b & set(TEST), b
    assert heldout(pages) == b                                  # re-run after test pages were moved: same buffer
    assert set(EXTRA) <= b                                      # already-labelled pages never get pseudo-labels


HTML = r"""<!doctype html>
<html lang="km"><head><meta charset="utf-8"><title>__TITLE__ labelling</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--ok:#1f7a44;--line:#dcdcd6;--cur:#2f6fd6;--zh:110px}
body{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
header b{font-size:17px} .bar{flex:1;min-width:120px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)} button{font:inherit;padding:5px 11px;border:1px solid var(--line);background:var(--card);border-radius:6px;cursor:pointer}
label{display:flex;gap:6px;align-items:center;cursor:pointer} #saved{font-size:12px;color:var(--ok)}
main{max-width:1400px;margin:0 auto;padding:12px 16px} h2{font-size:14px;color:var(--mute);margin:22px 0 6px}
.row{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:6px;padding:8px;margin:6px 0}
.row.ok{border-left-color:var(--ok)} .row.skip{border-left-color:var(--mute);opacity:.55} .row:focus-within{box-shadow:0 0 0 2px var(--cur)}
.hide{display:none} .pic{overflow-x:auto} .row img{display:block;height:var(--zh);width:auto;max-width:none}
.ed{display:flex;gap:8px;margin-top:6px;align-items:center}
.ed input{flex:1;font:26px "Khmer UI","Khmer OS Siemreap",Siemreap,"Noto Sans Khmer",sans-serif;padding:4px 8px;border:1px solid var(--line);border-radius:4px}
.hint{font-size:12px;color:var(--mute)} kbd{font-size:12px;border:1px solid var(--line);border-radius:3px;padding:0 4px}
</style></head><body>
<header><b>__TITLE__</b><span id="count"></span><div class="bar"><i id="prog"></i></div>
<label><input type="checkbox" id="todo"> unfinished only</label>
<span>zoom <button id="zo">−</button> <button id="zi">+</button></span>
<button id="file" hidden>Autosave to labels.csv…</button><span id="saved"></span>
<button id="exp">Export labels.csv</button>
<span class="hint" style="flex-basis:100%"><kbd>Enter</kbd> confirm &amp; next · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> skip
 (unreadable / not text) · <kbd>↑</kbd><kbd>↓</kbd> move · <kbd>Esc</kbd> back to model guess · <kbd>Ctrl</kbd>+<kbd>=</kbd>/<kbd>−</kbd> zoom</span></header>
<main id="m"></main>
<script>
const ROWS = __ROWS__;
const KEY = '__KEY__';
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const $ = id => document.getElementById(id), m = $('m'), rows = [];
const pref = (k, v) => { try { if (v === undefined) return localStorage.getItem('label_pref_' + k); localStorage.setItem('label_pref_' + k, v); } catch (e) {} };
let page = null;
ROWS.forEach((r, i) => {
  if (r.page !== page) { page = r.page; const h = document.createElement('h2'); h.textContent = page; m.appendChild(h); }
  const d = document.createElement('div'); d.className = 'row';
  d.innerHTML = `<div class="pic"><img loading="lazy" src="images/${r.image_path}" alt="line ${i + 1}"></div><div class="ed">
    <input lang="km"><button data-a="ok">✓</button><button data-a="skip">skip</button></div>`;
  const inp = d.querySelector('input'); inp.value = (st[r.image_path] || {}).text ?? r.guess;
  const done = () => !!(st[r.image_path] || {}).status;
  const paint = () => { const t = (st[r.image_path] || {}).status; d.classList.toggle('ok', t === 'ok'); d.classList.toggle('skip', t === 'skip'); };
  const mark = a => { st[r.image_path] = { text: inp.value.trim(), status: a }; save(); paint(); count(); persist(); };
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); mark(e.ctrlKey ? 'skip' : 'ok'); go(inp, 1); }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); go(inp, e.key === 'ArrowDown' ? 1 : -1); }
    else if (e.key === 'Escape') { inp.value = r.guess; inp.dispatchEvent(new Event('input')); }
  });
  inp.addEventListener('input', () => { if (st[r.image_path]) { st[r.image_path].status = ''; save(); paint(); count(); } });
  d.querySelector('[data-a=ok]').onclick = () => mark('ok'); d.querySelector('[data-a=skip]').onclick = () => mark('skip');
  m.appendChild(d); paint(); rows.push({ d, inp, done });
});
function visible() { return rows.filter(x => !x.d.classList.contains('hide')); }
function go(inp, step) {                                   // next / previous visible line, centred
  const v = visible(), n = v[v.findIndex(x => x.inp === inp) + step];
  if (n) { n.inp.focus(); n.d.scrollIntoView({block: 'center'}); } filter();
}
function filter() { const on = $('todo').checked; rows.forEach(x => x.d.classList.toggle('hide', on && x.done() && x.inp !== document.activeElement)); }
function count() {
  const done = rows.filter(x => x.done()).length;
  $('count').textContent = `${done} / ${ROWS.length} lines`;
  $('prog').style.width = (100 * done / ROWS.length) + '%';
}
function csv() {
  const q = v => '"' + String(v).replace(/"/g, '""') + '"';
  return '﻿' + ['image_path,text,writer_id,split,model_guess'].concat(ROWS.map(r => {
    const s = st[r.image_path] || {};
    return [r.image_path, s.status === 'ok' ? s.text : '', r.writer_id, '__SPLIT__', r.guess].map(q).join(',');
  })).join('\n');
}
$('exp').onclick = () => { const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv()], {type: 'text/csv'})); a.download = 'labels.csv'; a.click(); };
let fh = null, busy = false, again = false;                // autosave: File System Access API (Chrome / Edge)
async function persist() {
  if (!fh) return; if (busy) { again = true; return; } busy = true;
  try { const w = await fh.createWritable(); await w.write(csv()); await w.close();
    $('saved').textContent = 'saved ' + new Date().toLocaleTimeString(); }
  catch (e) { $('saved').textContent = 'autosave failed - use Export'; fh = null; }
  busy = false; if (again) { again = false; persist(); }
}
if (window.showSaveFilePicker) { $('file').hidden = false; $('file').onclick = async () => {
  try { fh = await showSaveFilePicker({ suggestedName: 'labels.csv', types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }] });
    $('file').textContent = 'Autosaving: ' + fh.name; persist(); } catch (e) {} }; }
let zh = +(pref('zoom') || 110);
const zoom = f => { zh = Math.max(40, Math.min(400, Math.round(zh * f))); document.documentElement.style.setProperty('--zh', zh + 'px'); pref('zoom', zh); };
$('zi').onclick = () => zoom(1.2); $('zo').onclick = () => zoom(1 / 1.2); zoom(1);
document.addEventListener('keydown', e => { if (e.ctrlKey && (e.key === '=' || e.key === '+')) { e.preventDefault(); zoom(1.2); }
  if (e.ctrlKey && e.key === '-') { e.preventDefault(); zoom(1 / 1.2); } });
$('todo').checked = pref('todo') === '1'; $('todo').onchange = () => { pref('todo', $('todo').checked ? '1' : '0'); filter(); };
count(); filter();
const first = rows.find(x => !x.done()); if (first) { first.inp.focus(); first.d.scrollIntoView({block: 'center'}); }
</script></body></html>
"""


def page(rows, title, key, split, exact_key=False):
    """label.html for rows; split = exported split. Browser storage is per key AND row list: every file:// page
    shares one localStorage, so a page with the same line names (the demo) must not show its answers here.
    exact_key=True keeps `key` as given (rebuilding a page someone is labelling: same slot, same progress)."""
    data = json.dumps(rows, ensure_ascii=False)
    if not exact_key:
        key = f"{key}_{hashlib.md5(data.encode('utf-8')).hexdigest()[:8]}"
    return HTML.replace('__TITLE__', title).replace('__KEY__', key).replace('__SPLIT__', split).replace('__ROWS__', data)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--unlabeled', default='data/raw/unlabeled')
    ap.add_argument('--out', default='data/raw/essay_test')
    ap.add_argument('--buffer', default='data/raw/heldout_unlabeled')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo:
        demo(); print('ok'); raise SystemExit
    unl, out, buf = Path(a.unlabeled), Path(a.out), Path(a.buffer)
    pages = [p.name for p in unl.iterdir() if p.is_dir()]
    missing = [t for t in TEST if t not in pages and not (out / 'images' / t).exists()]
    assert not missing, f'test pages without crops: {missing}'
    moved = [p.name for p in buf.iterdir() if p.is_dir()] if buf.exists() else []
    for p in sorted(heldout(pages + moved)):                     # counting moved pages keeps re-runs idempotent
        if (unl / p).exists():
            buf.mkdir(parents=True, exist_ok=True); shutil.move(str(unl / p), str(buf / p))
    for t in TEST:
        if (unl / t).exists():
            (out / 'images').mkdir(parents=True, exist_ok=True); shutil.move(str(unl / t), str(out / 'images' / t))

    imgs = sorted((out / 'images').glob('*/*.png'), key=lambda p: (TEST.index(p.parent.name), p.name))
    guess = {p: '' for p in imgs}
    if a.checkpoint:                                             # model's reading as a starting point
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
        from pseudo_label import read_all
        from src.models.checkpoint import load_stage1
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, tok, ck = load_stage1(Path(a.checkpoint).resolve(), device=dev); model.eval()
        df = read_all(model, tok, ck, imgs, dev)
        guess = dict(zip(imgs, df.text))
    rows = [{'image_path': p.relative_to(out / 'images').as_posix(), 'page': p.parent.name, 'writer_id': writer(p.parent.name),
             'guess': guess[p]} for p in imgs]
    labels = out / 'labels.csv'
    if not labels.exists():                                      # never overwrite typed labels
        with open(labels, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['image_path', 'text', 'writer_id', 'split', 'model_guess'])
            w.writerows([[r['image_path'], '', r['writer_id'], 'test', r['guess']] for r in rows])
    (out / 'label.html').write_text(page(rows, 'Essay test set', 'essay_test_labels_v1', 'test'), encoding='utf-8')
    print(f'{len(rows)} test lines from {len(TEST)} pages -> {out}/label.html; '
          f'{len(list(buf.glob("*")))} buffer pages -> {buf}')
