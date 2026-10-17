#!/usr/bin/env python
"""Download open-licence Khmer fonts into ./fonts.

    python scripts/fetch_fonts.py                  # body faces (recommended)
    python scripts/fetch_fonts.py --all            # + decorative/display faces
    python scripts/fetch_fonts.py --list           # show the roster, download nothing
    python scripts/fetch_fonts.py --source google  # only one upstream

Four upstreams, all redistributable:

``google``   github.com/google/fonts — every family whose Google Fonts subset is
             ``khmer`` (24 families, SIL Open Font License 1.1).
``khmeros``  KhmerSoftwareInitiative/khmer-unicode-fonts v5.0 (LGPL-3.0).  The
             KhmerOS faces by Danh Hong are the fonts most Cambodian government
             forms, school books and office documents are actually set in, so
             they matter more for real-document OCR than their star count
             suggests.
``khmertype`` danhhong/{KohSantepheap,Khla,HanumanUI} (OFL-1.1).  Koh Santepheap
             is the standard Khmer newspaper face.

Display faces are decorative (hollow, doubled or ornamented outlines).  They
skeletonise badly — the centreline traces the *decoration* rather than the
letter — so they are off by default for trajectory work, but they are valuable
OCR robustness data where only the image matters.  Fetch them with ``--all``
and enable them with ``--set fonts.exclude='[]'``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

GOOGLE = "https://github.com/google/fonts/raw/main/ofl"
KHMEROS = "https://github.com/KhmerSoftwareInitiative/khmer-unicode-fonts/raw/master/fonts/v5.0"
KHMERTYPE = "https://github.com/danhhong"

SFNT_MAGIC = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")


@dataclass(frozen=True)
class Font:
    url: str
    name: str  #: output filename
    kind: str  #: "body" | "display"
    source: str
    licence: str


def _google(family: str, filename: str, kind: str, out: str | None = None) -> Font:
    # '[wght]' etc. must be percent-encoded for the raw endpoint
    return Font(
        url=f"{GOOGLE}/{family}/{urllib.parse.quote(filename)}",
        name=out or filename,
        kind=kind,
        source="google/fonts",
        licence="OFL-1.1",
    )


def _khmeros(filename: str, kind: str) -> Font:
    return Font(
        url=f"{KHMEROS}/{filename}",
        name=filename,
        kind=kind,
        source="KhmerSoftwareInitiative/khmer-unicode-fonts",
        licence="LGPL-3.0",
    )


def _khmertype(repo: str, path: str, kind: str = "body") -> Font:
    return Font(
        url=f"{KHMERTYPE}/{repo}/raw/master/{path}",
        name=path.rsplit("/", 1)[-1],
        kind=kind,
        source=f"danhhong/{repo}",
        licence="OFL-1.1",
    )


FONTS: list[Font] = [
    # ── google/fonts, body ────────────────────────────────────────────────────
    _google("battambang", "Battambang-Thin.ttf", "body"),
    _google("battambang", "Battambang-Light.ttf", "body"),
    _google("battambang", "Battambang-Regular.ttf", "body"),
    _google("battambang", "Battambang-Bold.ttf", "body"),
    _google("battambang", "Battambang-Black.ttf", "body"),
    _google("suwannaphum", "Suwannaphum-Thin.ttf", "body"),
    _google("suwannaphum", "Suwannaphum-Light.ttf", "body"),
    _google("suwannaphum", "Suwannaphum-Regular.ttf", "body"),
    _google("suwannaphum", "Suwannaphum-Bold.ttf", "body"),
    _google("suwannaphum", "Suwannaphum-Black.ttf", "body"),
    _google("content", "Content-Regular.ttf", "body"),
    _google("content", "Content-Bold.ttf", "body"),
    _google("hanuman", "Hanuman[wght].ttf", "body", out="Hanuman-VF.ttf"),
    _google("nokora", "Nokora[wght].ttf", "body", out="Nokora-VF.ttf"),
    _google("kantumruypro", "KantumruyPro[wght].ttf", "body", out="KantumruyPro-VF.ttf"),
    _google("kantumruypro", "KantumruyPro-Italic[wght].ttf", "body", out="KantumruyPro-Italic-VF.ttf"),
    _google("notosanskhmer", "NotoSansKhmer[wdth,wght].ttf", "body", out="NotoSansKhmer-VF.ttf"),
    _google("notoserifkhmer", "NotoSerifKhmer[wdth,wght].ttf", "body", out="NotoSerifKhmer-VF.ttf"),
    _google("khmer", "Khmer.ttf", "body", out="Khmer-Regular.ttf"),
    _google("siemreap", "Siemreap.ttf", "body", out="Siemreap-Regular.ttf"),
    _google("chenla", "Chenla.ttf", "body", out="Chenla-Regular.ttf"),
    _google("freehand", "Freehand-Regular.ttf", "body"),
    _google("fasthand", "Fasthand-Regular.ttf", "body"),
    # ── google/fonts, display ─────────────────────────────────────────────────
    _google("moul", "Moul-Regular.ttf", "display"),
    _google("moulpali", "Moulpali-Regular.ttf", "display"),
    _google("bokor", "Bokor-Regular.ttf", "display"),
    _google("angkor", "Angkor-Regular.ttf", "display"),
    _google("taprom", "Taprom-Regular.ttf", "display"),
    # upstream spells it 'Preahvihear'; kept as 'PreahVihear-Regular' to match
    # the stem already listed in FontConfig.exclude
    _google("preahvihear", "Preahvihear-Regular.ttf", "display", out="PreahVihear-Regular.ttf"),
    _google("bayon", "Bayon-Regular.ttf", "display"),
    _google("dangrek", "Dangrek-Regular.ttf", "display"),
    _google("koulen", "Koulen-Regular.ttf", "display"),
    _google("metal", "Metal-Regular.ttf", "display"),
    _google("odormeanchey", "OdorMeanChey-Regular.ttf", "display"),
    # ── KhmerOS v5.0, body ────────────────────────────────────────────────────
    _khmeros("KhmerOS.ttf", "body"),
    _khmeros("KhmerOS_battambang.ttf", "body"),
    _khmeros("KhmerOS_content.ttf", "body"),
    _khmeros("KhmerOS_fasthand.ttf", "body"),
    _khmeros("KhmerOS_freehand.ttf", "body"),
    _khmeros("KhmerOS_siemreap.ttf", "body"),
    _khmeros("KhmerOS_sys.ttf", "body"),
    # ── KhmerOS v5.0, display ─────────────────────────────────────────────────
    _khmeros("KhmerOS_bokor.ttf", "display"),
    _khmeros("KhmerOS_metalchrieng.ttf", "display"),
    _khmeros("KhmerOS_muol.ttf", "display"),
    _khmeros("KhmerOS_muollight.ttf", "display"),
    _khmeros("KhmerOS_muolpali.ttf", "display"),
    # ── khmertype.org (Danh Hong) ─────────────────────────────────────────────
    _khmertype("KohSantepheap", "Release/ttf/KohSantepheap-Thin.ttf"),
    _khmertype("KohSantepheap", "Release/ttf/KohSantepheap-Light.ttf"),
    _khmertype("KohSantepheap", "Release/ttf/KohSantepheap-Regular.ttf"),
    _khmertype("KohSantepheap", "Release/ttf/KohSantepheap-Bold.ttf"),
    _khmertype("KohSantepheap", "Release/ttf/KohSantepheap-Black.ttf"),
    _khmertype("Khla", "Release/Khla-Light.ttf"),
    _khmertype("Khla", "Release/Khla-Regular.ttf"),
    _khmertype("Khla", "Release/Khla-SimiBold.ttf"),
    _khmertype("Khla", "Release/Khla-Bold.ttf"),
    _khmertype("HanumanUI", "Release/HanumanUI-Regular.ttf"),
    _khmertype("HanumanUI", "Release/HanumanUI-Bold.ttf"),
]

SOURCE_KEYS = {
    "google": "google/fonts",
    "khmeros": "KhmerSoftwareInitiative/khmer-unicode-fonts",
    "khmertype": "danhhong/",
}


def fetch_one(font: Font, out: Path, force: bool, timeout: int) -> tuple[str, str, int]:
    """-> (status, message, bytes).  status in {have, ok, failed}."""
    dst = out / font.name
    if dst.exists() and not force:
        return "have", font.name, dst.stat().st_size
    try:
        req = urllib.request.Request(font.url, headers={"User-Agent": "khmer-synth/fetch_fonts"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001 — report and continue with the rest
        return "failed", f"{font.name}: {exc}", 0
    if not data.startswith(SFNT_MAGIC):
        return "failed", f"{font.name}: not a font file (moved upstream?) {font.url}", 0
    dst.write_bytes(data)
    return "ok", font.name, len(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fonts")
    ap.add_argument("--all", action="store_true", help="also fetch decorative/display faces")
    ap.add_argument("--display-only", action="store_true", help="fetch only the display faces")
    ap.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_KEYS),
        help="restrict to one upstream (repeatable); default: all",
    )
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--list", action="store_true", help="print the roster and exit")
    ap.add_argument("--manifest", default=None, help="write a JSON manifest here (default: <out>/MANIFEST.json)")
    args = ap.parse_args(argv)

    wanted = list(FONTS)
    if args.display_only:
        wanted = [f for f in wanted if f.kind == "display"]
    elif not args.all:
        wanted = [f for f in wanted if f.kind == "body"]
    if args.source:
        prefixes = [SOURCE_KEYS[s] for s in args.source]
        wanted = [f for f in wanted if any(f.source.startswith(p) for p in prefixes)]

    if not wanted:
        print("nothing selected", file=sys.stderr)
        return 2

    if args.list:
        print(f"{len(wanted)} fonts selected ({len(FONTS)} total)\n")
        print(f"  {'file':38s} {'kind':8s} {'licence':10s} source")
        for f in wanted:
            print(f"  {f.name:38s} {f.kind:8s} {f.licence:10s} {f.source}")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"fetching {len(wanted)} fonts -> {out.resolve()}")
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {f: pool.submit(fetch_one, f, out, args.force, args.timeout) for f in wanted}
    # collected in roster order so the log is stable regardless of completion order
    results = [(f, futures[f].result()) for f in wanted]

    ok = have = failed = 0
    for font, (status, message, size) in results:
        if status == "ok":
            ok += 1
            print(f"  fetched {font.name:38s} {size / 1024:6.0f} kB  [{font.licence}]")
        elif status == "have":
            have += 1
            print(f"  have    {font.name:38s} {size / 1024:6.0f} kB")
        else:
            failed += 1
            print(f"  FAILED  {message}", file=sys.stderr)

    manifest_path = Path(args.manifest) if args.manifest else out / "MANIFEST.json"
    present = [f for f, (s, _, _) in results if s in ("ok", "have")]
    manifest_path.write_text(
        json.dumps(
            {
                "fonts": [
                    {"file": f.name, "kind": f.kind, "licence": f.licence, "source": f.source, "url": f.url}
                    for f in present
                ],
                "counts": {
                    "total": len(present),
                    "body": sum(1 for f in present if f.kind == "body"),
                    "display": sum(1 for f in present if f.kind == "display"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    on_disk = sorted(p.name for p in out.glob("*.tt[fc]")) + sorted(p.name for p in out.glob("*.otf"))
    print(f"\n{ok} fetched, {have} already present, {failed} failed")
    print(f"{len(on_disk)} font files now in {out.resolve()}  (manifest: {manifest_path.name})")
    print(
        "Licences: SIL Open Font License 1.1 (google/fonts, khmertype.org) and "
        "LGPL-3.0 (KhmerOS).  Images and trajectories rendered from a font are\n"
        "derivative of its outlines — check the terms before redistributing a dataset."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
