"""Official-order fresh Validation visits, Range-only native ARKit references.

Reads no model outputs. All admission decisions use the prespecified reference
coverage gate and are retained, including skipped captures and failed requests.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile
import numpy as np
from PIL import Image
import requests
from rgb_body_query_3rscan import optical_z, sensor_labels
from rgb_body_query_fixed_grid import queries, independent_xyz_labels, PUBLIC
from rgb_body_query_input_diagnostic import sha, write

ASSETS = (("lowres_wide", ".png"), ("lowres_depth", ".png"),
          ("lowres_wide_intrinsics", ".pincam"))
MISSING = {"", "NA", "nan", "NaN", None}


class Allocation:
    def __init__(self, root, wall_s, initial_received, download_limit=3_000_000_000):
        self.root, self.start = root, time.perf_counter()
        self.deadline = self.start + wall_s
        self.received, self.reserved, self.limit = initial_received, 0, download_limit
        if self.received > self.limit:
            raise TimeoutError("Official metadata fetch already exceeds download allocation")
        self.records = []
        self.session = requests.Session()

    def remaining(self):
        left = self.deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError("Acquisition command-wall allocation exhausted")
        return left

    def request(self, url, bounds=None):
        self.remaining()
        expected = bounds[1] - bounds[0] + 1 if bounds else 0
        if self.received + self.reserved + expected > self.limit:
            raise TimeoutError("Cumulative download allocation exhausted before Range reservation")
        self.reserved += expected
        rec = dict(url=url, method="GET" if bounds else "HEAD", bounds=bounds,
                   reserved_bytes=expected, received_bytes=0, status="STARTING")
        start = time.perf_counter()
        data = bytearray()
        response = None
        try:
            headers = {"Accept-Encoding": "identity"}
            if bounds:
                headers["Range"] = f"bytes={bounds[0]}-{bounds[1]}"
            response = self.session.request(rec["method"], url, headers=headers,
                timeout=(min(8, self.remaining()), min(20, self.remaining())), stream=True)
            rec.update(http_status=response.status_code, headers=dict(response.headers))
            if not bounds:
                if response.status_code != 200:
                    raise ValueError("Official HEAD did not return200")
                rec["status"] = "COMPLETE"
                return int(response.headers["Content-Length"])
            a, b, size = bounds
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {a}-{b}/{size}":
                raise ValueError("Exact HTTP206/Content-Range required; refusing full archive")
            for chunk in response.iter_content(chunk_size=65536):
                self.remaining()
                self.received += len(chunk)
                rec["received_bytes"] += len(chunk)
                data.extend(chunk)
                if len(data) > expected or self.received > self.limit:
                    raise ValueError("Range response exceeded reservation")
            if len(data) != expected:
                raise ValueError("Range response length mismatch")
            rec["status"] = "COMPLETE"
            return bytes(data)
        except Exception as exc:
            rec.update(status="FAILED", error=repr(exc))
            if data:
                partial = self.root / f"request-{len(self.records):05d}.partial"
                partial.write_bytes(data)
                rec["partial_path"] = str(partial)
            raise
        finally:
            if response is not None:
                response.close()
            self.reserved -= expected
            rec.update(wall_s=time.perf_counter()-start,
                       received_sha256=hashlib.sha256(data).hexdigest())
            self.records.append(rec)
            write(self.root / "network_progress.json", dict(received_bytes=self.received,
                outstanding_reserved_bytes=self.reserved, requests=self.records))


class RangeZip(io.RawIOBase):
    def __init__(self, url, size, allocation):
        self.url, self.size, self.allocation, self.pos = url, size, allocation, 0
        self.cache_start, self.cache = -1, b""
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos+n if whence == 1 else self.size+n
        if self.pos < 0: raise ValueError("Negative ZIP seek")
        return self.pos
    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if n <= 0: return b""
        result = bytearray()
        while len(result) < n:
            self.allocation.remaining()
            offset = self.pos-self.cache_start
            if not 0 <= offset < len(self.cache):
                a = self.pos
                length = min(self.size-a, max(n-len(result), 262144))
                self.cache = self.allocation.request(self.url, (a, a+length-1, self.size))
                self.cache_start, offset = a, 0
            take = min(n-len(result), len(self.cache)-offset)
            result.extend(self.cache[offset:offset+take])
            self.pos += take
        return bytes(result)


def acquire(root, wall_s=370, target_count=3, download_limit=3_000_000_000,
            cohort_prefix="new_arkit_"):
    started = time.perf_counter()
    root = root.resolve()
    plan = root / "PLAN.json"
    if not plan.exists(): raise FileNotFoundError("Root must freeze PLAN.json first")
    if (root / "acquisition_terminal.json").exists():
        raise FileExistsError("Preserve prior terminal; no implicit restart")
    official = root / "official"
    metadata_path = official / "raw_raw_train_val_splits.csv"
    metadata = list(csv.DictReader(metadata_path.read_text(encoding="utf-8-sig").splitlines()))
    inventory = json.loads((root / "consumed_inventory.json").read_text(encoding="utf-8-sig"))
    excluded = set(inventory["capture_ids"])
    old_visits = set(inventory["visit_ids"])
    fixed = queries()
    initial_bytes = sum(p.stat().st_size for p in official.iterdir() if p.is_file())
    if target_count < 1 or download_limit < 1:
        raise ValueError("Positive target_count and download_limit required")
    allocation = Allocation(root, wall_s-3, initial_bytes, download_limit)
    receipt = dict(status="STARTING", pid=os.getpid(), gpu_s=0, model_outputs_read=False,
        plan_sha256=sha(plan), metadata_sha256=sha(metadata_path),
        inventory_sha256=sha(root / "consumed_inventory.json"), source_sha256=sha(__file__),
        official_split="Validation", metadata_order="Original CSV physical row order; not sorted by video_id",
        frames_per_capture=16, near_positive_query_gate=16, target_count=target_count,
        download_limit_bytes=download_limit, cohort_prefix=cohort_prefix, candidates=[], accepted=[])
    write(root / "acquisition_frozen_inputs.json", receipt)
    public_rows, evaluator_rows = [], []
    attempted_visits = set()
    try:
        for index, row in enumerate(metadata):
            allocation.remaining()
            cap, visit = row["video_id"], row["visit_id"]
            if row["fold"] != "Validation" or cap in excluded or visit in MISSING or visit in old_visits or visit in attempted_visits:
                continue
            candidate = dict(metadata_index=index, capture=cap, visit_id=visit,
                             official_metadata_row=row, status="STARTING")
            receipt["candidates"].append(candidate)
            write(root / "candidate_selection_progress.json", receipt)
            source = root / "source" / cap
            source.mkdir(parents=True, exist_ok=True)
            cohort = cohort_prefix+cap
            sensor = root / "new-eval-sensor" / cohort
            sensor.mkdir(parents=True, exist_ok=True)
            archives, streams, indices, details, files = {}, {}, {}, {}, {}
            try:
                for asset, suffix in ASSETS:
                    url = f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/{cap}/{asset}.zip"
                    size = allocation.request(url)
                    stream = RangeZip(url, size, allocation)
                    archive = zipfile.ZipFile(stream)
                    streams[asset], archives[asset] = stream, archive
                    directory = [dict(name=z.filename, crc32=z.CRC, size=z.file_size,
                        compressed_size=z.compress_size, header_offset=z.header_offset) for z in archive.infolist()]
                    write(source / (asset+"_directory.json"), directory)
                    indices[asset] = {Path(n).stem: n for n in archive.namelist() if n.endswith(suffix)}
                    if len(indices[asset]) != sum(n.endswith(suffix) for n in archive.namelist()):
                        raise ValueError("Duplicate timestamp stems")
                    details[asset] = dict(url=url, full_official_archive_size_bytes=size,
                        directory_sha256=sha(source / (asset+"_directory.json")), constructed_subset=True)
                common = sorted(set.intersection(*(set(v) for v in indices.values())),
                                key=lambda n: float(n.rsplit("_", 1)[1]))
                chosen = np.rint(np.linspace(0, len(common)-1, 16)).astype(int)
                if len(set(chosen)) != 16: raise ValueError("Insufficient16 aligned eligible frames")
                names = [common[i] for i in chosen]
                write(source / "selection.json", dict(metadata_index=index, capture=cap, visit_id=visit,
                    metadata_sha256=sha(metadata_path), common_pairs=len(common),
                    chosen_indices=chosen.tolist(), source_ids=names, queries=fixed,
                    saved_before_selected_RGB_depth_intrinsics_member_reads=True))
                # Depth/K coverage admission needs no model or RGB pixels.
                # Preserve RGB directory alignment, but avoid downloading RGB
                # payloads for visits that fail the prespecified near POS gate.
                for asset in ("lowres_depth", "lowres_wide_intrinsics", "lowres_wide"):
                    if asset == "lowres_wide":
                        early_near_pos = 0
                        for name in names:
                            allocation.remaining()
                            raw = np.asarray(Image.open(io.BytesIO(files["lowres_depth",name])))
                            w,h,fx,fy,cx,cy = np.fromstring(files["lowres_wide_intrinsics",name].decode(),sep=" ")
                            k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64)
                            if (w,h)!=(256,192) or raw.shape!=(192,256) or raw.dtype!=np.uint16:
                                raise ValueError("Native reference dimension/type mismatch before coverage gate")
                            if not np.isfinite(k).all() or min(fx,fy)<=0:
                                raise ValueError("Invalid public K before coverage gate")
                            depth = optical_z(raw,1000.)
                            observed = np.ones(depth.shape,bool)
                            for q in fixed:
                                if q["low"][2]==.3 and q["high"][2]==.8:
                                    _, state = sensor_labels(depth,k,q,observed)
                                    early_near_pos += state["state"]=="POSITIVE"
                        if early_near_pos < 16:
                            attempted_visits.add(visit)
                            candidate.update(status="SKIP_NEAR_REFERENCE_COVERAGE",near_POS=early_near_pos,
                                frames=16, reason="nearPOS<16 frozen gate; no RGB/model inference read",
                                synchronized_RGB_directory_checked=True,RGB_payload_downloaded=False)
                            break
                    archive = archives[asset]
                    members = []
                    with zipfile.ZipFile(source / (asset+".zip"), "w", compression=zipfile.ZIP_STORED) as subset:
                        for name in names:
                            allocation.remaining()
                            member = indices[asset][name]
                            payload = archive.read(member)
                            subset.writestr(member, payload)
                            files[asset, name] = payload
                            members.append(dict(name=member, bytes=len(payload), crc32=archive.getinfo(member).CRC,
                                sha256=hashlib.sha256(payload).hexdigest()))
                    details[asset].update(selected_members=members, subset_sha256=sha(source / (asset+".zip")))
                if candidate["status"] == "SKIP_NEAR_REFERENCE_COVERAGE":
                    write(source / "download_receipt.json",dict(status="COMPLETE_DEPTH_K_COVERAGE_SUBSET",
                        archives=details,not_whole_archives=True,exact_HTTP206_ContentRange_checked=True,
                        zip_member_CRC_checked=True,RGB_payload_downloaded=False))
                    continue
                write(source / "download_receipt.json", dict(status="COMPLETE_RANGE_SUBSET",
                    archives=details, not_whole_archives=True, exact_HTTP206_ContentRange_checked=True,
                    zip_member_CRC_checked=True))
                sensor_rows = []
                for frame, name in enumerate(names):
                    allocation.remaining()
                    rgb_bytes, depth_bytes, kb = (files[a, name] for a, _ in ASSETS)
                    rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                    raw = np.asarray(Image.open(io.BytesIO(depth_bytes)))
                    w, h, fx, fy, cx, cy = np.fromstring(kb.decode(), sep=" ")
                    k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
                    if (w,h)!=(256,192) or rgb.size!=(256,192) or raw.shape!=(192,256) or raw.dtype!=np.uint16:
                        raise ValueError("Native registered256x192 asset size/type mismatch")
                    if not np.isfinite(k).all() or min(fx,fy)<=0: raise ValueError("Invalid public K")
                    depth = optical_z(raw, 1000.)
                    my, mx = np.indices(depth.shape, dtype=np.float32)
                    observed = np.ones(depth.shape, bool)
                    labels, states = [], []
                    for q in fixed:
                        label, state = sensor_labels(depth,k,q,observed)
                        independent, _ = independent_xyz_labels(depth,k,q,observed)
                        if not np.array_equal(label,independent): raise ValueError("Independent XYZ mismatch")
                        labels.append(label); states.append(state)
                    rp, ref = sensor / (name+".png"), sensor / (name+".npz")
                    rp.write_bytes(rgb_bytes)
                    np.savez_compressed(ref, labels=np.stack(labels), depth=depth, depth_K=k,
                                        color_K=k, map_x=mx, map_y=my, observed=observed)
                    sensor_rows.append(dict(environment="arkitscenes_"+cap, scan="arkitscenes_"+cap,
                        split="validation", official_split="Validation", frame=frame, source_id=name,
                        timestamp_s=float(name.rsplit("_",1)[1]), rgb_path=str(rp),rgb_sha256=sha(rp),
                        reference_path=str(ref),reference_sha256=sha(ref),color_K=k.tolist(),depth_K=k.tolist(),
                        color_shape=[192,256],depth_shape=[192,256],depth_shift=1000.,queries=states,
                        source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                        calibration_sha256=hashlib.sha256(kb).hexdigest()))
                counts = dict(Counter(q["state"] for r in sensor_rows for q in r["queries"]))
                near_pos = sum(state["state"]=="POSITIVE" for r in sensor_rows
                    for q,state in zip(fixed,r["queries"]) if q["low"][2]==.3 and q["high"][2]==.8)
                candidate.update(near_POS=near_pos, frames=16, query_state_counts=counts)
                attempted_visits.add(visit)
                manifest = dict(status="REAL_SENSOR_FIXED_FRAGMENT_READY", rows=sensor_rows, queries=fixed,
                    frames=16, groups=[dict(environment="arkitscenes_"+cap,scan="arkitscenes_"+cap,split="validation")],
                    state_counts_by_split={"validation":counts}, selection_sha256=sha(source / "selection.json"),
                    sensor="Official iPad native registered256x192 RGB/LiDAR/public pincam; optical-Z mm/1000; invalid0 UNKNOWN",
                    observation_contract="RGB/public K/query only; references evaluator-only",
                    query_contract="Frozen27 sampled first-return boxes, min16; no whole-volume clearance claim",
                    license="Official Apple non-commercial research license; ignored local storage, no redistribution",
                    official_data_role="New official Validation visit, eval-only Development")
                write(sensor / "dataset_manifest.json", manifest)
                obs = [dict({key:r[key] for key in PUBLIC},cohort=cohort,role="eval") for r in sensor_rows]
                write(sensor / "observations.json",dict(rows=[{k:r[k] for k in PUBLIC} for r in sensor_rows],
                    queries=fixed,dataset_manifest_sha256=sha(sensor / "dataset_manifest.json")))
                if near_pos < 16:
                    candidate.update(status="SKIP_NEAR_REFERENCE_COVERAGE", reason="nearPOS<16 frozen gate; no model inference")
                else:
                    candidate.update(status="ACCEPTED", cohort=cohort,
                        manifest_path=str(sensor / "dataset_manifest.json"), manifest_sha256=sha(sensor / "dataset_manifest.json"))
                    receipt["accepted"].append(dict(candidate))
                    public_rows.extend(obs); evaluator_rows.extend(sensor_rows)
            except TimeoutError:
                candidate.update(status="BUDGET_STOP", error=traceback.format_exc())
                raise
            except Exception as exc:
                candidate.update(status="SKIP_MISSING_OR_INELIGIBLE_SOURCE", error=repr(exc), traceback=traceback.format_exc())
            finally:
                for archive in archives.values(): archive.close()
                for stream in streams.values(): stream.close()
                write(source / "candidate_terminal.json", candidate)
                write(root / "candidate_selection_progress.json", receipt)
                print(json.dumps(candidate), flush=True)
            if len(receipt["accepted"]) == target_count: break
        receipt["status"] = f"COMPLETE{target_count}_NEW_VISITS" if len(receipt["accepted"])==target_count else "PARTIAL_INSUFFICIENT_ELIGIBLE_VISITS"
    except Exception as exc:
        receipt.update(status="BUDGET_STOP" if isinstance(exc,TimeoutError) else "FAILED_PARTIAL", error=repr(exc))
    finally:
        allocation.session.close()
        write(root / "new_public_roster.json",dict(status="COMPLETE" if len(receipt["accepted"])==target_count else "PARTIAL",
            rows=public_rows, frames=len(public_rows), evaluator_read=False, candidate_selection_frozen=True,
            queries=fixed, selection_plan_sha256=sha(plan),
            accepted_cohorts_order=[r["cohort"] for r in receipt["accepted"]]))
        write(root / "new_evaluator_roster.json",dict(rows=evaluator_rows,queries=fixed,role="eval-only"))
        receipt.update(command_wall_s=time.perf_counter()-started, downloaded_bytes=allocation.received,
            outstanding_reserved_bytes=allocation.reserved, requests=allocation.records,
            accepted_captures=len(receipt["accepted"]), accepted_frames=len(public_rows), resources_released=True)
        write(root / "acquisition_terminal.json",receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!="requests"}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--wall-s",type=float,default=370)
    p.add_argument("--target-count",type=int,default=3)
    p.add_argument("--download-limit",type=int,default=3_000_000_000)
    p.add_argument("--cohort-prefix",default="new_arkit_")
    a=p.parse_args()
    acquire(a.root,a.wall_s,a.target_count,a.download_limit,a.cohort_prefix)
