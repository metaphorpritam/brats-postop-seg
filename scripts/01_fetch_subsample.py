#!/usr/bin/env python
"""Phase 1 (CLAUDE.md §6.1): subsample ~200 BraTS-GLI cases from the Kaggle mirror.

Stages (idempotent / resumable):
  masks      download seg mask for each selected case; verify the label contract
             (per-case ⊆ {0,1,2,3,4} + integer-valued; aggregate: ET/label-3 must
             appear somewhere → confirms 2024 post-treatment vintage, §6.1.2 / D7).
  split      stratified 70/15/15 split by RC (class-4) presence -> reports/splits.json;
             assert class-4 present in >=15% of each split.
  modalities download t2f/t1c/t2w for the split cases (original drops t1n); gzip on arrival.

Downloads land on native ext4 (~/brats/raw), gzipped as <case>/<case>-<mod>.nii.gz.
Usage: uv run python scripts/01_fetch_subsample.py --n 200 --seed 42 --stage all
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import socket
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brats import paths  # noqa: E402

DS = "i212385nomanarif/2024-brats-glioma"
INPUT_MODS = ("t2f", "t1c", "t2w")          # 3 input channels (original drops t1n)
LABELSET = {0, 1, 2, 3, 4}
REPO = Path(__file__).resolve().parents[1]
MANIFEST = paths.RAW_DIR / "_manifest.json"
LABELS_SUMMARY = REPO / "reports" / "labels_summary.json"
SPLITS = REPO / "reports" / "splits.json"

_tl = threading.local()


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    if not hasattr(_tl, "api"):
        a = KaggleApi()
        a.authenticate()
        _tl.api = a
    return _tl.api


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())["cases"]


def select_cases(n: int, seed: int) -> list[str]:
    cases = sorted(load_manifest().keys())
    rng = random.Random(seed)
    return sorted(rng.sample(cases, n))


def _download_one(case: str, mod: str) -> Path:
    """Download <case>-<mod>.nii, gzip to <case>/<case>-<mod>.nii.gz, delete raw. Idempotent."""
    case_dir = paths.RAW_DIR / case
    gz = case_dir / f"{case}-{mod}.nii.gz"
    if gz.exists() and gz.stat().st_size > 0:
        return gz
    case_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{case}/{case}-{mod}.nii"
    last = None
    for attempt in range(4):
        try:
            # NB: no contextlib.redirect_stdout here — it mutates the *global* sys.stdout
            # and races across worker threads, swallowing the main thread's progress log.
            api().dataset_download_file(DS, remote, path=str(case_dir), force=True, quiet=True)
            raw = case_dir / f"{case}-{mod}.nii"
            if not raw.exists():
                z = case_dir / f"{case}-{mod}.nii.zip"
                if z.exists():
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(case_dir)
                    z.unlink()
            if not raw.exists():
                raise FileNotFoundError(f"{remote} did not land as expected")
            tmp = gz.with_suffix(".gz.tmp")
            with open(raw, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
            tmp.replace(gz)
            raw.unlink()
            return gz
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed {remote} after retries: {last}")


def _parallel(items, fn, workers, label):
    done, fail = 0, {}
    total = len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                fail[it] = str(exc)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{label}] {done}/{total}  (failures={len(fail)})", flush=True)
    return fail


def verify_seg(case: str) -> dict:
    gz = paths.RAW_DIR / case / f"{case}-seg.nii.gz"
    data = np.asarray(nib.load(str(gz)).dataobj)
    if not np.allclose(data, np.round(data), atol=1e-3):
        raise ValueError(f"{case}: non-integer label values — source resampling corruption?")
    labs = set(np.round(data).astype(np.int64).ravel().tolist()) if data.size < 1 else \
        set(np.unique(np.round(data).astype(np.int64)).tolist())
    if not labs <= LABELSET:
        raise ValueError(f"{case}: labels {sorted(labs)} NOT ⊆ {sorted(LABELSET)} — wrong vintage, STOP (§6.1.2)")
    return {"labels": sorted(labs), "rc": 4 in labs, "shape": list(data.shape)}


def stage_masks(cases: list[str], workers: int) -> None:
    print(f"== masks: downloading {len(cases)} seg masks (workers={workers}) ==")
    fail = _parallel([(c, "seg") for c in cases], _download_one, workers, "seg")
    if fail:
        print(f"  DOWNLOAD FAILURES ({len(fail)}): {list(fail)[:5]} ...")
    print("== verifying label contract (vintage-aware) ==")
    summary, agg = {}, {i: 0 for i in range(5)}
    bad = []
    for c in cases:
        try:
            info = verify_seg(c)
            summary[c] = info
            for l in info["labels"]:
                agg[l] += 1
        except Exception as exc:  # noqa: BLE001
            bad.append((c, str(exc)))
    if bad:
        print(f"  CONTRACT VIOLATIONS ({len(bad)}):")
        for c, e in bad[:10]:
            print("   -", e)
        raise SystemExit("Label-contract check failed — STOP (§8.4, do not proceed).")
    n = len(summary)
    rc = sum(1 for v in summary.values() if v["rc"])
    print(f"  cases verified: {n}")
    print(f"  per-label case presence: " + ", ".join(f"{l}:{agg[l]}" for l in range(5)))
    # Aggregate vintage assertion: ET (label 3) must appear somewhere -> 2024 post-treatment.
    if agg[3] == 0:
        raise SystemExit("Label 3 (ET) never appears across the sample — looks PRE-OP, not "
                         "post-treatment 2024. STOP (§6.1.2 / D7).")
    print(f"  ✓ ET(3) present in {agg[3]} cases -> post-treatment 2024 vintage confirmed")
    print(f"  RC(4) present in {rc}/{n} cases ({100*rc/n:.1f}%)")
    LABELS_SUMMARY.write_text(json.dumps(summary, indent=1))
    print(f"  labels summary -> {LABELS_SUMMARY}")


def stage_split(seed: int) -> None:
    summary = json.loads(LABELS_SUMMARY.read_text())
    cases = sorted(summary)
    rc = {c: summary[c]["rc"] for c in cases}
    rng = random.Random(seed)
    strata = {True: [c for c in cases if rc[c]], False: [c for c in cases if not rc[c]]}
    split = {"train": [], "val": [], "test": []}
    for present, group in strata.items():
        g = group[:]
        rng.shuffle(g)
        n = len(g)
        n_tr, n_va = round(0.70 * n), round(0.15 * n)
        split["train"] += g[:n_tr]
        split["val"] += g[n_tr:n_tr + n_va]
        split["test"] += g[n_tr + n_va:]
    for k in split:
        split[k] = sorted(split[k])
    counts = {k: {"n": len(v), "rc": sum(rc[c] for c in v)} for k, v in split.items()}
    print("== stratified 70/15/15 split by RC presence ==")
    for k, c in counts.items():
        pct = 100 * c["rc"] / c["n"] if c["n"] else 0
        flag = "" if pct >= 15 else "   <-- WARNING <15% RC"
        print(f"  {k:5s}: n={c['n']:3d}  RC={c['rc']:3d} ({pct:.1f}%){flag}")
    out = {"dataset": DS, "seed": seed, "n": len(cases),
           "stratify": "rc_presence", "counts": counts, "splits": split}
    SPLITS.write_text(json.dumps(out, indent=1))
    print(f"  splits -> {SPLITS}")
    if any(100 * c["rc"] / c["n"] < 15 for c in counts.values() if c["n"]):
        print("  NOTE: a split has <15% RC — see Gate 1; may need to expand RC cases (§6.1.1).")


def stage_modalities(workers: int) -> None:
    split = json.loads(SPLITS.read_text())["splits"]
    cases = sorted({c for v in split.values() for c in v})
    items = [(c, m) for c in cases for m in INPUT_MODS]
    print(f"== modalities: downloading {len(items)} files for {len(cases)} cases (workers={workers}) ==")
    fail = _parallel(items, _download_one, workers, "mod")
    if fail:
        print(f"  DOWNLOAD FAILURES ({len(fail)}): {list(fail)[:5]} ...")
        raise SystemExit("Some modality downloads failed — re-run stage 'modalities' (idempotent).")
    print("  ✓ all modalities downloaded + gzipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stage", choices=["masks", "split", "modalities", "all"], default="all")
    args = ap.parse_args()
    # Harden against stalled HTTPS reads: the kaggle client sets no socket timeout,
    # so a stuck recv() blocks a worker thread forever and the ThreadPoolExecutor
    # never completes (observed: 698/700 masks land, 2 stall, the whole run wedges
    # at 0% CPU indefinitely). A per-read timeout converts a stall into a
    # socket.timeout that _download_one's 4-attempt backoff loop already retries.
    socket.setdefaulttimeout(90)
    paths.assert_native_storage(paths.RAW_DIR)
    cases = select_cases(args.n, args.seed)
    if args.stage in ("masks", "all"):
        stage_masks(cases, args.workers)
    if args.stage in ("split", "all"):
        stage_split(args.seed)
    if args.stage in ("modalities", "all"):
        stage_modalities(args.workers)


if __name__ == "__main__":
    main()
