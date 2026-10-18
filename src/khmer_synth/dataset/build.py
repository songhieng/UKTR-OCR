"""Large-scale dataset generation (multiprocessing).

Design notes
------------
* Workers write their own image / trajectory files and return only a small
  metadata dict, so no image data is ever pickled through the queue.
* Each job carries its own seed derived from ``(global seed, index)``: the
  dataset is reproducible regardless of worker count, scheduling or chunking.
* The expensive per-cluster skeletons are shared through the on-disk cache, so
  workers warm each other up instead of duplicating the work.
* GPU is never required.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..utils.rng import stable_hash
from ..writer.style import save_writers, writer_table
from .generator import SampleGenerator

_WORKER: dict[str, Any] = {}


@dataclass
class Job:
    index: int
    text: str
    writer_id: int
    seed: int


def _shard_dir(root: Path, index: int, shard_size: int) -> Path:
    if shard_size <= 0:
        return root
    return root / f"{index // shard_size:05d}"


def _save_image(path: Path, image: np.ndarray, fmt: str) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(image)
    if fmt == "webp":
        img.save(path, "WEBP", quality=92, method=4)
    elif fmt in ("jpg", "jpeg"):
        img.save(path, "JPEG", quality=92)
    else:
        img.save(path, "PNG", optimize=False, compress_level=3)


#: env vars that must be set *before* numpy / OpenCV are imported, otherwise the
#: thread pools are already built.  With N worker processes each spawning one
#: thread per core the machine thrashes; single-threaded workers scale linearly.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_threads(n: int = 1) -> None:
    """Pin this process to ``n`` compute threads (BLAS + OpenCV)."""
    for var in _THREAD_ENV:
        os.environ[var] = str(n)
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:  # pragma: no cover
        pass


def _init_worker(cfg_dict: dict[str, Any], out_dir: str) -> None:  # pragma: no cover - subprocess
    limit_threads(1)
    cfg = PipelineConfig.from_dict(cfg_dict)
    _WORKER["cfg"] = cfg
    _WORKER["gen"] = SampleGenerator(cfg)
    _WORKER["out"] = Path(out_dir)


def _run_job(job: Job) -> dict[str, Any] | None:  # pragma: no cover - subprocess
    cfg: PipelineConfig = _WORKER["cfg"]
    gen: SampleGenerator = _WORKER["gen"]
    out: Path = _WORKER["out"]
    return generate_and_write(gen, cfg, out, job)


def generate_and_write(
    gen: SampleGenerator, cfg: PipelineConfig, out: Path, job: Job
) -> dict[str, Any] | None:
    ds = cfg.dataset
    sample_id = f"sample_{job.index:08d}"
    try:
        result = gen.generate(
            job.text,
            writer_id=job.writer_id,
            seed=job.seed,
            sample_id=sample_id,
            render=ds.save_clean or ds.save_degraded,
            degrade_image=ds.save_degraded,
        )
    except Exception as exc:  # a bad corpus line must not kill the run
        return {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}", "text": job.text}

    if ds.validate and result.problems:
        return {"sample_id": sample_id, "error": "; ".join(result.problems), "text": job.text}

    meta = result.metadata
    fmt = ds.image_format
    if ds.save_clean and result.clean is not None:
        p = _shard_dir(out / "images_clean", job.index, ds.shard_size) / f"{sample_id}.{fmt}"
        _save_image(p, result.clean, fmt)
        meta.clean_image = str(p.relative_to(out)).replace("\\", "/")
    if ds.save_degraded and result.degraded is not None:
        p = _shard_dir(out / "images_degraded", job.index, ds.shard_size) / f"{sample_id}.{fmt}"
        _save_image(p, result.degraded, fmt)
        meta.degraded_image = str(p.relative_to(out)).replace("\\", "/")
    if ds.save_trajectory:
        p = _shard_dir(out / "trajectories", job.index, ds.shard_size) / f"{sample_id}.json"
        result.trajectory.save(p, normalized=True, compact=ds.save_compact)
        meta.trajectory_file = str(p.relative_to(out)).replace("\\", "/")
    return meta.to_dict()


def make_jobs(
    corpus: Sequence[str], cfg: PipelineConfig, start_index: int = 0
) -> list[Job]:
    ds = cfg.dataset
    jobs: list[Job] = []
    idx = start_index
    rng = np.random.default_rng(ds.seed)
    for text in corpus:
        for _ in range(max(1, ds.styles_per_text)):
            writer_id = int(rng.integers(0, max(1, ds.num_writers)))
            seed = int(stable_hash(ds.seed, idx, text, writer_id) % (2**31))
            jobs.append(Job(index=idx, text=text, writer_id=writer_id, seed=seed))
            idx += 1
            if ds.limit and len(jobs) >= ds.limit:
                return jobs
    return jobs


def build_dataset(
    corpus: Sequence[str],
    cfg: PipelineConfig,
    progress: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """Generate a full dataset on disk.  Returns a summary dict."""
    ds = cfg.dataset
    out = Path(ds.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg.to_json(out / "config.json")
    fonts_probe = SampleGenerator(cfg)
    save_writers(
        out / "writers.json",
        writer_table(
            ds.num_writers,
            cfg.writer,
            seed=ds.seed,
            font_keys=fonts_probe.fonts.keys,
            font_weights=cfg.fonts.weights,
            font_default_weight=getattr(cfg.fonts, "default_weight", 1.0),
        ),
    )
    del fonts_probe

    jobs = make_jobs(corpus, cfg)
    n_workers = workers if workers is not None else (ds.workers or os.cpu_count() or 1)
    n_workers = max(1, int(n_workers))
    if n_workers > 1:
        # spawned children inherit the environment, so the thread limits have to
        # be in place *before* the pool is created (they are read at import time)
        for var in _THREAD_ENV:
            os.environ[var] = "1"

    meta_path = out / "metadata.jsonl"
    err_path = out / "errors.jsonl"
    t0 = time.time()
    written = 0
    failed = 0

    iterator: Iterable[dict[str, Any] | None]
    if n_workers == 1:
        gen = SampleGenerator(cfg)
        iterator = (generate_and_write(gen, cfg, out, j) for j in jobs)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(cfg.to_dict(), str(out)),
        )
        iterator = pool.imap_unordered(_run_job, jobs, chunksize=max(1, ds.chunk_size))

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(jobs), unit="sample", dynamic_ncols=True)
        except ImportError:  # pragma: no cover
            bar = None

    try:
        with meta_path.open("w", encoding="utf-8") as mf, err_path.open("w", encoding="utf-8") as ef:
            for record in iterator:
                if bar is not None:
                    bar.update(1)
                if record is None:
                    failed += 1
                    continue
                if "error" in record:
                    failed += 1
                    ef.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if bar is not None:
            bar.close()

    elapsed = time.time() - t0
    summary = {
        "output_dir": str(out),
        "requested": len(jobs),
        "written": written,
        "failed": failed,
        "workers": n_workers,
        "elapsed_s": round(elapsed, 2),
        "samples_per_s": round(written / max(elapsed, 1e-9), 2),
        "corpus_lines": len(corpus),
        "num_writers": ds.num_writers,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    return summary
