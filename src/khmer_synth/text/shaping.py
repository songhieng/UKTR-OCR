"""HarfBuzz shaping with a preserved text <-> glyph mapping.

Khmer needs a real shaper: dependent vowels reorder around the base consonant
(``េ``, ``ើ`` and friends are typed after but drawn before), coeng sequences turn
into subscript glyph forms, and register shifters move.  HarfBuzz's Khmer shaper
does all of that; we only have to keep the bookkeeping.

Output units are **font units** (HarfBuzz is scaled to ``upem``), so a shaped
line is resolution independent and can be scaled to any em size later.

Mapping preserved by this module::

    original text -> normalised text -> grapheme clusters -> shaped glyphs
                                             ^                    |
                                             +--- cluster_index --+
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import uharfbuzz as hb

from ..config import ShapingConfig
from .graphemes import GraphemeCluster, cluster_index_map, segment_graphemes

if TYPE_CHECKING:  # pragma: no cover
    from ..fonts.loader import LoadedFont


@dataclass
class ShapedGlyph:
    """One positioned glyph, in font units, y-UP (font space)."""

    glyph_id: int
    glyph_name: str
    #: code-point index into the *normalised* string reported by HarfBuzz
    cluster: int
    #: index of the grapheme cluster this glyph belongs to
    cluster_index: int
    x_offset: float
    y_offset: float
    x_advance: float
    y_advance: float
    #: absolute pen position at which this glyph is drawn (offset already added)
    x: float
    y: float

    def to_dict(self) -> dict:
        return {
            "glyph_id": self.glyph_id,
            "glyph_name": self.glyph_name,
            "cluster": self.cluster,
            "cluster_index": self.cluster_index,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "x_advance": self.x_advance,
            "y_advance": self.y_advance,
            "x": self.x,
            "y": self.y,
        }


@dataclass
class ShapedText:
    text: str
    font_key: str
    upem: int
    glyphs: list[ShapedGlyph]
    clusters: list[GraphemeCluster]
    advance: float  # total advance in font units
    #: cluster index -> indices into ``glyphs``
    cluster_glyphs: dict[int, list[int]] = field(default_factory=dict)

    @property
    def has_notdef(self) -> bool:
        return any(g.glyph_id == 0 for g in self.glyphs)

    def cluster_bounds(self, cluster_index: int) -> tuple[float, float]:
        idx = self.cluster_glyphs.get(cluster_index, [])
        if not idx:
            return (0.0, 0.0)
        xs = [self.glyphs[i].x for i in idx]
        return (min(xs), max(xs))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "font": self.font_key,
            "upem": self.upem,
            "advance": self.advance,
            "glyphs": [g.to_dict() for g in self.glyphs],
            "clusters": [
                {"text": c.text, "start": c.start, "end": c.end, "kind": c.kind} for c in self.clusters
            ],
        }


class Shaper:
    """Thin, stateless wrapper over ``hb.shape``."""

    def __init__(self, cfg: ShapingConfig | None = None):
        self.cfg = cfg or ShapingConfig()

    def shape(self, font: LoadedFont, text: str, features: dict[str, bool] | None = None) -> ShapedText:
        clusters = segment_graphemes(text)
        cmap = cluster_index_map(clusters, len(text))

        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = self.cfg.direction
        if self.cfg.script:
            buf.script = self.cfg.script
        if self.cfg.language:
            buf.language = self.cfg.language
        try:
            buf.cluster_level = self.cfg.cluster_level
        except Exception:  # pragma: no cover - older uharfbuzz
            pass
        if not (self.cfg.script and self.cfg.language):
            buf.guess_segment_properties()

        feats = dict(self.cfg.features)
        if features:
            feats.update(features)
        hb.shape(font.hb_font, buf, feats or None)

        infos = buf.glyph_infos
        positions = buf.glyph_positions

        glyphs: list[ShapedGlyph] = []
        pen_x = 0.0
        pen_y = 0.0
        for info, pos in zip(infos, positions):
            cl = int(info.cluster)
            ci = cmap[cl] if 0 <= cl < len(cmap) else (len(clusters) - 1 if clusters else 0)
            g = ShapedGlyph(
                glyph_id=int(info.codepoint),
                glyph_name=font.glyph_name(int(info.codepoint)),
                cluster=cl,
                cluster_index=int(ci),
                x_offset=float(pos.x_offset),
                y_offset=float(pos.y_offset),
                x_advance=float(pos.x_advance),
                y_advance=float(pos.y_advance),
                x=pen_x + float(pos.x_offset),
                y=pen_y + float(pos.y_offset),
            )
            glyphs.append(g)
            pen_x += float(pos.x_advance)
            pen_y += float(pos.y_advance)

        cluster_glyphs: dict[int, list[int]] = {}
        for i, g in enumerate(glyphs):
            cluster_glyphs.setdefault(g.cluster_index, []).append(i)

        return ShapedText(
            text=text,
            font_key=font.key,
            upem=font.upem,
            glyphs=glyphs,
            clusters=clusters,
            advance=pen_x,
            cluster_glyphs=cluster_glyphs,
        )

    def shape_with_spacing(
        self,
        font: LoadedFont,
        text: str,
        cluster_spacing: float = 1.0,
        word_spacing: float = 1.0,
        per_cluster_extra: Sequence[float] | None = None,
    ) -> ShapedText:
        """Shape, then re-lay-out the glyphs with writer-dependent spacing.

        Spacing is applied *between* grapheme clusters (never inside one), so
        subscripts and diacritics stay attached to their base.  ``per_cluster_extra``
        is an optional per-cluster extra advance in font units (irregular spacing).
        """
        shaped = self.shape(font, text)
        if cluster_spacing == 1.0 and word_spacing == 1.0 and not per_cluster_extra:
            return shaped

        # per-cluster advance in shaping order
        order: list[int] = []
        for g in shaped.glyphs:
            if g.cluster_index not in order:
                order.append(g.cluster_index)

        pen = 0.0
        new_glyphs: list[ShapedGlyph] = []
        for k, ci in enumerate(order):
            idx = shaped.cluster_glyphs[ci]
            base_x = min(shaped.glyphs[i].x - shaped.glyphs[i].x_offset for i in idx)
            adv = sum(shaped.glyphs[i].x_advance for i in idx)
            is_space = shaped.clusters[ci].kind == "space" if ci < len(shaped.clusters) else False
            scale = word_spacing if is_space else cluster_spacing
            extra = float(per_cluster_extra[k]) if per_cluster_extra is not None and k < len(per_cluster_extra) else 0.0
            for i in idx:
                g = shaped.glyphs[i]
                new_glyphs.append(
                    ShapedGlyph(
                        glyph_id=g.glyph_id,
                        glyph_name=g.glyph_name,
                        cluster=g.cluster,
                        cluster_index=g.cluster_index,
                        x_offset=g.x_offset,
                        y_offset=g.y_offset,
                        x_advance=g.x_advance,
                        y_advance=g.y_advance,
                        x=pen + (g.x - g.x_offset - base_x) + g.x_offset,
                        y=g.y,
                    )
                )
            pen += adv * scale + extra

        new_glyphs.sort(key=lambda g: (g.cluster_index, g.x))
        cluster_glyphs: dict[int, list[int]] = {}
        for i, g in enumerate(new_glyphs):
            cluster_glyphs.setdefault(g.cluster_index, []).append(i)
        return ShapedText(
            text=shaped.text,
            font_key=shaped.font_key,
            upem=shaped.upem,
            glyphs=new_glyphs,
            clusters=shaped.clusters,
            advance=pen,
            cluster_glyphs=cluster_glyphs,
        )
