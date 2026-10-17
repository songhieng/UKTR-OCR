"""Corpus loading (.txt / .jsonl / .csv) + light filtering."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..text.graphemes import segment_graphemes
from ..text.normalization import looks_corrupted, normalize_khmer


def load_corpus(
    path: str | Path,
    column: str | None = None,
    limit: int = 0,
    min_clusters: int = 1,
    max_clusters: int = 64,
    dedupe: bool = True,
    drop_corrupted: bool = True,
) -> list[str]:
    """Read text lines from ``path``.

    ``.txt``   one sample per line
    ``.jsonl`` one JSON object per line; ``column`` (default "text") is read
    ``.csv``   ``column`` (default: the first column) is read
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    raw: list[str] = []

    if suffix == ".jsonl":
        key = column or "text"
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = obj.get(key) if isinstance(obj, dict) else None
                if isinstance(value, str):
                    raw.append(value)
    elif suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            key = column or reader.fieldnames[0]
            for row in reader:
                value = row.get(key)
                if isinstance(value, str):
                    raw.append(value)
    else:
        with p.open("r", encoding="utf-8") as fh:
            raw = [line.rstrip("\n") for line in fh]

    return filter_corpus(
        raw,
        limit=limit,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        dedupe=dedupe,
        drop_corrupted=drop_corrupted,
    )


def filter_corpus(
    lines: Iterable[str],
    limit: int = 0,
    min_clusters: int = 1,
    max_clusters: int = 64,
    dedupe: bool = True,
    drop_corrupted: bool = True,
) -> list[str]:
    out: list[str] = []
    seen = set()
    for line in lines:
        text = (line or "").strip()
        if not text:
            continue
        norm = normalize_khmer(text)
        if not norm:
            continue
        if drop_corrupted and looks_corrupted(norm):
            continue
        n = len([c for c in segment_graphemes(norm) if c.kind != "space"])
        if n < min_clusters or n > max_clusters:
            continue
        if dedupe:
            if norm in seen:
                continue
            seen.add(norm)
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def grapheme_coverage(lines: Sequence[str]) -> dict:
    """Distinct grapheme clusters in the corpus (dataset statistics)."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for line in lines:
        for c in segment_graphemes(normalize_khmer(line)):
            if c.kind != "space":
                counter[c.text] += 1
    return {
        "unique_clusters": len(counter),
        "total_clusters": int(sum(counter.values())),
        "top": counter.most_common(30),
    }


DEFAULT_CORPUS = [
    "កម្ពុជា",
    "រាជធានីភ្នំពេញ",
    "ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា",
    "សូមអរគុណ",
    "អក្សរសាស្ត្រខ្មែរ",
    "សៀមរាប",
    "បាត់ដំបង",
    "ព្រះវិហារ",
    "ទន្លេសាប",
    "អង្គរវត្ត",
    "សាលារៀន",
    "គ្រូបង្រៀន",
    "សិស្សានុសិស្ស",
    "ថ្ងៃនេះមេឃស្រឡះ",
    "តើអ្នកសុខសប្បាយជាទេ",
    "ខ្ញុំចង់ទៅផ្សារ",
    "លេខ ០១២ ៣៤៥ ៦៧៨",
    "ឆ្នាំ ២០២៥",
    "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញជារាជធានី",
    "ការអប់រំគឺជាគន្លឹះនៃភាពជោគជ័យ",
    "សេចក្តីស្រឡាញ់និងសន្តិភាព",
    "វប្បធម៌ខ្មែរ",
    "សំណេរដោយដៃ",
    "ភាសាខ្មែរ Khmer 123",
]
