"""Regenerate the paired outdoor baseline in time order, saving inputs for shallow readouts."""

import argparse
import gc
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

import paired_low_sites_replay_20261118 as paired
from ground_geometry_filter_20261118 import GroundEvidence, protection
from sanpo_ground_alert_replay_20261118 import evaluator

frozen = paired.frozen
CACHE = frozen.ROOT / "artifacts.local/work/shallow-alerts-20261118/cache"


def ensure_memory():
    if psutil.virtual_memory().available >= 2 * 1024**3:
        return
    # Release only this worker's idle pages on a memory-constrained Windows host.
    import ctypes
    import sys
    gc.collect()
    if sys.platform == "win32":
        lib = ctypes.WinDLL("psapi", use_last_error=True)
        if not lib.EmptyWorkingSet(ctypes.c_void_p(-1)):
            raise ctypes.WinError(ctypes.get_last_error())
    print("Releasing idle worker pages; free MiB",
          round(psutil.virtual_memory().available / 1024**2), flush=True)
    for _ in range(20):
        time.sleep(1)
        if psutil.virtual_memory().available >= 2 * 1024**3:
            return
    raise MemoryError("Below 2 GiB of free physical memory")


def check_old_support(z, frame, floor, sem, geom, old_packet, evidence):
    """Reference coordinates only enter this reproduction check, never the saved input."""
    with np.load(old_packet, allow_pickle=False) as packet:
        old = packet["pred_masks"]
        area_indices = packet["pred_indices"]
        old_area = packet["pred_area_m2"]
    region = evaluator(floor)
    for i, (name, remove) in enumerate(zip(frozen.ARMS, (None, sem, geom), strict=True)):
        masks, area = frozen.supported(z, frame, region, remove)
        bits = {band: bool(area[m].sum() >= frozen.core.MIN_AREA)
                for band, m in zip(frozen.BANDS, masks, strict=True)}
        if bits != evidence[name]:
            raise ValueError(f"Prior alarm mismatch at {frame.seq}/{frame.index}, {name}")
        if not np.array_equal(np.packbits(masks, axis=-1, bitorder="little"), old[i]):
            raise ValueError(f"Prior band pixels differ at {frame.seq}/{frame.index}, {name}")
        if i == 0 and not np.array_equal(area.ravel()[area_indices], old_area):
            raise ValueError(f"Prior support areas differ at {frame.seq}/{frame.index}")


def run(manifest, model, ground):
    digest = frozen.sha(manifest)
    old_run = frozen.OUT / digest[:16]
    reference = frozen.read(old_run / "reference.json")
    previous = frozen.read(old_run / "predictions.json")
    if reference["manifest_sha256"] != digest or previous["manifest_sha256"] != digest:
        raise ValueError("Source digest differs from previous replay")
    dataset, records = frozen.inputs(manifest)
    out = CACHE / digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    result = []
    verified = 0
    for row, ref, old in zip(records, reference["rows"], previous["rows"], strict=True):
        ensure_memory()
        key = tuple(row[k] for k in ("seq", "i", "t"))
        if not all(tuple(r[k] for k in ("seq", "i", "t")) == key for r in (ref, old)):
            raise ValueError("Input order differs from the previous replay")
        target = out / (frozen.unikey(*key[:2]) + ".npz")
        scale_key = ("hybrid_out", dataset, key[0])
        entry = dict(seq=key[0], i=key[1], t=key[2], file=target.name,
                     scale_source=old["scale_source"], old_support_verified=old["support_path"] is not None)
        if target.is_file():
            with np.load(target, allow_pickle=False) as cached:
                if "last_scale" in cached.files and str(cached["manifest_sha256"]) == digest:
                    scale = float(cached["last_scale"])
                    if np.isfinite(scale):
                        model.last_fit[scale_key] = scale
                    result.append(entry)
                    verified += int(entry["old_support_verified"])
                    continue
        frame = frozen.load_frame(dataset, row, old_run, True, model_grid=True)
        frame.eye_height = reference["eye_height"].get(frame.seq)
        z = model.hybrid_out(frame)
        source = model.source["hybrid_out"]
        if source != old["scale_source"] or z is None or z.dtype != np.float32:
            raise ValueError(f"Missing or changed baseline depth/scale at {key}")
        probability = ground.probability(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        sem = probability >= .80
        geom = sem & ~protection(z, frame.K, probability)
        if old["support_path"] is not None:
            check_old_support(z, frame, ref["ground"], sem, geom,
                              old_run / old["support_path"], old["evidence"])
            verified += 1
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, z=z, probability=probability, sem=sem, geom=geom, K=frame.K,
                     last_scale=model.last_fit.get(scale_key, np.nan), manifest_sha256=digest)
        tmp.replace(target)
        result.append(entry)
        if len(result) == 1 or len(result) % 20 == 0:
            print(dataset, len(result), "/", len(records), flush=True)
    frozen.save(out / "index.json", dict(dataset=dataset, manifest_sha256=digest,
                old_prediction_sha256=frozen.sha(old_run / "predictions.json"),
                old_pixel_masks_and_areas_verified=verified, old_alarm_bits_verified=verified * 9,
                rows=result))
    print(dataset, "verified", verified, "of", len(result), "peak VRAM MiB",
          round(torch.cuda.max_memory_reserved() / 1024**2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = frozen.Methods()
    # The outdoor hybrid uses only the UniDepth branch of Methods.
    del model.rel
    torch.cuda.empty_cache()
    model.set_unidepth_pixels(315560)
    ground = GroundEvidence(device="cuda")
    for manifest in args.manifests:
        run(manifest.resolve(), model, ground)
