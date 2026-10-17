"""Two-level cache (in-process LRU + optional on-disk pickle).

Rasterising and skeletonising a Khmer grapheme cluster costs milliseconds; a
large corpus repeats the same few thousand clusters millions of times, so this
cache is what makes large-scale generation feasible.  Keys are content hashes of
``(font_key, cluster glyph signature, raster/skeleton config)`` so a config change
automatically invalidates.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .rng import stable_hash


class LRUCache:
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


class DiskCache:
    """Tiny sharded pickle cache; safe for concurrent workers (atomic replace).

    Security note: entries are written by this process into a local, private
    cache directory (``dataset.cache_dir``) and are never shared or downloaded,
    so pickle is used purely as a fast serialiser for numpy-heavy graph objects.
    Point ``cache_dir`` at a directory only you can write to; if you need to
    share a cache across untrusted users, delete it instead of importing it.
    """

    def __init__(self, root: str | Path | None, enabled: bool = True):
        self.enabled = bool(enabled and root is not None)
        self.root = Path(root) if root is not None else None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # corrupt / partially written entry -> treat as a miss
            return None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)
        except Exception:
            pass  # cache failures must never break generation


class TwoLevelCache:
    def __init__(self, capacity: int = 4096, disk_root: str | Path | None = None):
        self.mem = LRUCache(capacity)
        self.disk = DiskCache(disk_root, enabled=disk_root is not None)

    def get_or_compute(self, key_parts: tuple, factory: Callable[[], Any]) -> Any:
        key = f"{stable_hash(*key_parts):016x}"
        value = self.mem.get(key)
        if value is not None:
            return value
        value = self.disk.get(key)
        if value is None:
            value = factory()
            self.disk.put(key, value)
        self.mem.put(key, value)
        return value

    def stats(self) -> dict:
        return self.mem.stats()
