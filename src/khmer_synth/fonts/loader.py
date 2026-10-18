"""Font discovery and loading (fontTools + uharfbuzz on the same file).

A :class:`LoadedFont` bundles the two views of a font that the pipeline needs:

* the HarfBuzz view (``hb_font``) used for *shaping* - glyph selection,
  reordering and positioning;
* the fontTools view (``glyph_set``) used for *outline extraction*.

Both are configured with the same variable-font axis location so that the shaped
advances and the extracted outlines can never disagree.

Fonts are cached per process, keyed by ``(path, variations)``; HarfBuzz objects
are not picklable, so multiprocessing workers each build their own library
lazily from the config (which *is* picklable).
"""

from __future__ import annotations

import fnmatch
import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTCollection, TTFont

from ..config import FontConfig
from ..utils.rng import choice_weighted

FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".otc")


@dataclass
class LoadedFont:
    key: str
    path: str
    face_index: int
    upem: int
    variations: dict[str, float]
    hb_font: hb.Font
    tt_font: TTFont
    glyph_set: object
    cmap: dict[int, str]
    ascender: float
    descender: float
    #: glyph id -> glyph name (fontTools order == HarfBuzz gid order for sfnt)
    glyph_order: list[str] = field(default_factory=list)
    #: short content hash of the font file.  ``key`` is only the filename stem and
    #: is therefore NOT unique: MEASURED, ``fonts/Battambang-Regular.ttf``
    #: (md5 ba95e740…, 360 glyphs) and ``fonts/km/Battambang-Regular.ttf``
    #: (md5 971de454…, 302 glyphs) are different files that collide on the stem,
    #: and both produced ``cluster_skeleton`` key a8874b6c0d2a5d91.  Since
    #: ``fonts/km/*`` are symlinks, repointing one was invisible to the cache and
    #: stale skeletons were served forever.  This goes into both cache keys.
    content_hash: str = ""

    @property
    def uid(self) -> str:
        """Stem + content hash: unique per font *file*, unlike ``key``."""
        return f"{self.key}@{self.content_hash}" if self.content_hash else self.key

    def glyph_name(self, gid: int) -> str:
        if 0 <= gid < len(self.glyph_order):
            return self.glyph_order[gid]
        return f"gid{gid}"

    def covers(self, text: str) -> bool:
        return all(ord(c) in self.cmap or c.isspace() for c in text)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<LoadedFont {self.key} upem={self.upem} glyphs={len(self.glyph_order)}>"


_FONT_CACHE: dict[tuple[str, int, tuple[tuple[str, float], ...]], LoadedFont] = {}
_CACHE_LOCK = threading.Lock()
_HASH_CACHE: dict[str, str] = {}


def _content_hash(path: str | Path) -> str:
    """Short, move-stable fingerprint of a font *file*.

    Used to disambiguate two files that share a filename stem in the skeleton /
    plan cache keys - see ``LoadedFont.content_hash`` for the proven collision.
    A content hash rather than (path, size, mtime) so that repointing a symlink
    under ``fonts/km`` invalidates the cache while merely moving the tree does
    not.  Cached per process; ~245 files hash in well under a second.
    """
    p = str(Path(path))
    with _CACHE_LOCK:
        hit = _HASH_CACHE.get(p)
    if hit is not None:
        return hit
    h = hashlib.blake2b(digest_size=8)
    try:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out = h.hexdigest()
    except OSError:  # pragma: no cover - unreadable file
        out = ""
    with _CACHE_LOCK:
        _HASH_CACHE[p] = out
    return out


def load_font(
    path: str | Path,
    variations: dict[str, float] | None = None,
    face_index: int = 0,
) -> LoadedFont:
    """Load (and cache) a font file for both shaping and outline extraction."""
    path = str(Path(path))
    variations = dict(variations or {})
    key = (path, face_index, tuple(sorted(variations.items())))
    with _CACHE_LOCK:
        if key in _FONT_CACHE:
            return _FONT_CACHE[key]

    blob = hb.Blob.from_file_path(path)
    face = hb.Face(blob, face_index) if face_index else hb.Face(blob)
    hb_font = hb.Font(face)
    upem = face.upem
    hb_font.scale = (upem, upem)  # shape in font units -> resolution independent
    if variations:
        try:
            hb_font.set_variations(variations)
        except Exception:  # pragma: no cover - font without fvar
            pass

    if Path(path).suffix.lower() in (".ttc", ".otc"):
        tt = TTCollection(path, lazy=True).fonts[face_index]
    else:
        tt = TTFont(path, fontNumber=face_index, lazy=True)

    location = variations if variations else None
    try:
        glyph_set = tt.getGlyphSet(location=location) if location else tt.getGlyphSet()
    except TypeError:  # older fontTools
        glyph_set = tt.getGlyphSet()

    cmap = {}
    try:
        cmap = {cp: name for cp, name in tt.getBestCmap().items()}
    except Exception:  # pragma: no cover
        cmap = {}

    hhea = tt["hhea"] if "hhea" in tt else None
    ascender = float(getattr(hhea, "ascent", upem * 0.8))
    descender = float(getattr(hhea, "descent", -upem * 0.2))

    font = LoadedFont(
        key=Path(path).stem,
        path=path,
        content_hash=_content_hash(path),
        face_index=face_index,
        upem=int(upem),
        variations=variations,
        hb_font=hb_font,
        tt_font=tt,
        glyph_set=glyph_set,
        cmap=cmap,
        ascender=ascender,
        descender=descender,
        glyph_order=list(tt.getGlyphOrder()),
    )
    with _CACHE_LOCK:
        _FONT_CACHE[key] = font
    return font


def discover_font_files(paths: Sequence[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for suffix in FONT_SUFFIXES:
                out.extend(sorted(p.rglob(f"*{suffix}")))
        elif p.is_file() and p.suffix.lower() in FONT_SUFFIXES:
            out.append(p)
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


_GLOB_CHARS = ("*", "?", "[")


def font_is_excluded(path: Path, patterns: Sequence[str], mode: str = "exact") -> str | None:
    """Return the ``exclude`` entry that bans ``path``, else ``None``.

    See ``FontConfig.exclude_match`` for why the modes exist: the shipped
    exclude lists name Google-style stems while the ``fonts/km`` symlinks carry
    vendor filenames, so 26 decorative faces the config means to drop were being
    sampled as handwriting.
    """
    stem = path.stem
    # the stem plus every directory component, so "Moul" can also ban fonts/km/Moul/*
    parts = [stem] + [p for p in path.parts[:-1]]
    low = [p.casefold() for p in parts]
    smart = mode == "smart"
    for pat in patterns:
        if not pat:
            continue
        if not smart and stem == pat:
            return pat  # v1 semantics, kept exactly
        pl = pat.casefold()
        if any(c in pat for c in _GLOB_CHARS):
            if any(fnmatch.fnmatch(p, pl) for p in low):
                return pat
        elif smart and any(pl in p for p in low):
            return pat
    return None


class FontLibrary:
    """A weighted collection of Khmer-capable fonts."""

    def __init__(self, cfg: FontConfig):
        self.cfg = cfg
        self.fonts: list[LoadedFont] = []
        self.rejected: list[tuple[str, str]] = []
        self._load()

    def _load(self) -> None:
        files = discover_font_files(self.cfg.paths)
        if not files:
            raise FileNotFoundError(
                f"no font files found under {self.cfg.paths!r} "
                "(pass --fonts <dir> or run scripts/fetch_fonts.py)"
            )
        mode = getattr(self.cfg, "exclude_match", "exact")
        for f in files:
            stem = f.stem
            banned = font_is_excluded(f, self.cfg.exclude, mode)
            if banned is not None:
                self.rejected.append((stem, f"excluded by config ({banned!r})"))
                continue
            try:
                font = load_font(f, self.cfg.variations.get(stem))
            except Exception as exc:
                self.rejected.append((stem, f"load error: {exc}"))
                continue
            probe = self.cfg.coverage_probe
            if probe and not font.covers(probe):
                self.rejected.append((stem, "does not cover the Khmer probe string"))
                continue
            self.fonts.append(font)
        if not self.fonts:
            raise RuntimeError(
                "no usable Khmer fonts. Rejected: "
                + "; ".join(f"{k}: {v}" for k, v in self.rejected)
            )

    @property
    def keys(self) -> list[str]:
        return [f.key for f in self.fonts]

    def get(self, key: str) -> LoadedFont:
        """Look up by ``uid`` (unique) or, failing that, by ``key`` (first match).

        Configs, ``preferred_fonts`` and sample metadata all speak stems, so the
        stem lookup has to stay; it is the ``uid`` form that is unambiguous.
        """
        for f in self.fonts:
            if f.uid == key:
                return f
        for f in self.fonts:
            if f.key == key:
                return f
        raise KeyError(f"font {key!r} not in library ({self.keys})")

    def font_weight(self, font: LoadedFont) -> float:
        """Sampling weight of one face (``uid`` first, then ``key``, then default).

        WHY the default is configurable: MEASURED, with unlisted fonts pinned at
        1.0 only 7.3% of `real_match_v2`'s sampling mass reached the 17 faces it
        tunes, because ``fonts.paths: ["fonts"]`` also rglobs the 232 faces in
        ``fonts/km``.  See ``FontConfig.default_weight``.
        """
        w = self.cfg.weights
        if font.uid in w:
            return float(w[font.uid])
        if font.key in w:
            return float(w[font.key])
        return float(getattr(self.cfg, "default_weight", 1.0))

    def sample(self, rng: np.random.Generator, text: str | None = None) -> LoadedFont:
        """Pick a font, honouring configured weights and (optionally) coverage."""
        candidates = self.fonts
        if text:
            covering = [f for f in self.fonts if f.covers(text)]
            if covering:
                candidates = covering
        # Weight by *index*, not by key: two files can share a stem, and keying a
        # dict on the stem silently merged them and then handed back whichever
        # one `get()` found first - which could be a face that does not cover the
        # text at all, rendering .notdef boxes.
        weights = {str(i): max(self.font_weight(f), 0.0) for i, f in enumerate(candidates)}
        if sum(weights.values()) <= 0.0:  # every candidate weighted 0 -> uniform
            weights = {k: 1.0 for k in weights}
        return candidates[int(choice_weighted(rng, weights))]

    def __len__(self) -> int:
        return len(self.fonts)

    def __iter__(self):
        return iter(self.fonts)
