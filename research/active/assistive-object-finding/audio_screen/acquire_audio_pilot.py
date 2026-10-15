"""Acquire the bounded ESC-50 open-vocabulary rejection Development screen.

No GPU or API credentials. Keep official folds and Freesound source groups.
Do not turn ESC-50's dominant single label into exhaustive background tags.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import wave

import requests

SEED = "20261015"
KNOWN = ["dog", "crying_baby", "footsteps", "door_wood_knock", "clock_alarm",
         "glass_breaking", "car_horn", "siren", "engine", "toilet_flush"]
CAL_UNKNOWN = ["cat", "hen", "rain", "wind", "coughing", "laughing", "door_wood_creaks",
               "keyboard_typing", "train", "vacuum_cleaner"]
EVAL_UNKNOWN = ["rooster", "insects", "sea_waves", "crackling_fire", "sneezing", "clapping",
                "mouse_click", "church_bells", "helicopter", "chainsaw"]
REPO = "https://raw.githubusercontent.com/karolpiczak/ESC-50/"
DEFAULT_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def select(rows):
    """Fixed class/fold quotas; prefer a new src_file in each role."""
    assert set(KNOWN + CAL_UNKNOWN + EVAL_UNKNOWN) <= {r["category"] for r in rows}
    selected, used = [], {"calibration": set(), "evaluation": set()}
    strata = [("calibration", KNOWN, True, [(1, 1), (2, 1), (3, 1)]),
              ("calibration", CAL_UNKNOWN, False, [(1, 1), (2, 1), (3, 1)]),
              ("evaluation", KNOWN, True, [(4, 2), (5, 1)]),
              ("evaluation", EVAL_UNKNOWN, False, [(4, 2), (5, 1)])]
    for role, labels, known, quotas in strata:
        for label in labels:
            for fold, count in quotas:
                candidates = sorted((r for r in rows if r["category"] == label and int(r["fold"]) == fold),
                                    key=lambda r: digest((r["filename"] + SEED).encode()))
                chosen = []
                for _ in range(count):
                    available = [r for r in candidates if r["filename"] not in chosen]
                    fresh = [r for r in available if r["src_file"] not in used[role]]
                    row = dict((fresh or available)[0])
                    chosen.append(row["filename"])
                    used[role].add(row["src_file"])
                    row.update(role=role, known=known)
                    selected.append(row)
    assert len(selected) == 120 and len({r["filename"] for r in selected}) == 120
    assert not used["calibration"] & used["evaluation"], "Source leakage across roles"
    assert len({r["category"] for r in selected if r["known"]}) == 10
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Exact 40-character ESC-50 commit SHA; defaults to the observed pilot revision")
    args = parser.parse_args()
    assert "artifacts.local" in args.root.as_posix(), "Use the canonical artifact entry"
    root, started = args.root.resolve(), time.monotonic()
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    network = {"observed_audio_bytes": 0, "limit_audio_bytes": 130_000_000}
    lock = threading.Lock()
    receipts = []
    revision = args.revision.lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("--revision must be an exact 40-character commit SHA")
    response = requests.get(f"https://api.github.com/repos/karolpiczak/ESC-50/commits/{revision}", timeout=(5, 15))
    response.raise_for_status()
    if response.json()["sha"].lower() != revision:
        raise ValueError("Commit metadata does not match the requested revision")
    (data_root / "ESC50_source_commit.json").write_bytes(response.content)
    for filename, relative in (("esc50.csv", "meta/esc50.csv"), ("ESC50_LICENSE.txt", "LICENSE")):
        destination = data_root / filename
        if not destination.exists():
            response = requests.get(f"{REPO}{revision}/{relative}", timeout=(5, 20))
            response.raise_for_status()
            if len(response.content) > 500_000:
                raise ValueError("Unexpected ESC metadata size")
            destination.write_bytes(response.content)
    metadata = (data_root / "esc50.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
    selection = select(rows)
    license_text = (data_root / "ESC50_LICENSE.txt").read_text(encoding="utf-8")
    licenses = {}
    for line in license_text.splitlines():
        match = re.match(r"- \[([^]]+\.ogg)\]:.*\[([^]]+)\]$", line)
        if match:
            licenses[match.group(1)] = {"license": match.group(2), "attribution_line": line}
    selection_receipt = {"seed": SEED, "hash_rule": "sha256(filename + '20261015')",
                         "official_repository": "https://github.com/karolpiczak/ESC-50",
                         "audio_revision": revision, "metadata_sha256": digest(metadata),
                         "source_preference": "New src_file within role; official fold boundary strictly preserved",
                         "known_classes": KNOWN, "calibration_unknown_classes": CAL_UNKNOWN,
                         "evaluation_unknown_classes": EVAL_UNKNOWN, "selected": selection}
    write_json(data_root / "ESC50_selection.json", selection_receipt)

    def acquire(row):
        filename = row["filename"]
        path = data_root / "audio" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{REPO}{revision}/audio/{filename}"
        if not path.exists():
            last_error = None
            for attempt in range(3):
                try:
                    if time.monotonic() - started > 300:
                        raise TimeoutError("Bounded acquisition wall limit reached")
                    with requests.get(url, stream=True, timeout=(5, 15)) as response:
                        response.raise_for_status()
                        payload = bytearray()
                        for block in response.iter_content(65536):
                            with lock:
                                network["observed_audio_bytes"] += len(block)
                                if network["observed_audio_bytes"] > network["limit_audio_bytes"]:
                                    raise RuntimeError("Audio download byte limit reached")
                            payload.extend(block)
                            if len(payload) > 500_000:
                                raise ValueError("Unexpected ESC-50 clip size")
                    path.write_bytes(payload)
                    break
                except requests.RequestException as error:
                    last_error = type(error).__name__
            else:
                raise RuntimeError(f"ESC download failed for {filename}: {last_error}")
        payload = path.read_bytes()
        with wave.open(io.BytesIO(payload), "rb") as audio:
            params = {"sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
                      "sample_width_bytes": audio.getsampwidth(), "frames": audio.getnframes()}
            assert params == {"sample_rate": 44100, "channels": 1,
                              "sample_width_bytes": 2, "frames": 220500}, (filename, params)
            assert len(audio.readframes(audio.getnframes())) == 441000
        license_key = "-".join(filename[:-4].split("-")[:-1]) + ".ogg"
        attribution = licenses.get(license_key, {"license": "See official ESC-50 LICENSE",
                                               "attribution_line": ""})
        receipt = {"filename": filename, "url": url, "path": str(path), "bytes": len(payload),
                   "sha256": digest(payload), "dataset_license": "CC BY-NC 3.0 (ESC-10 subset CC BY 3.0)",
                   **attribution, **params}
        with lock:
            receipts.append(receipt)
            print(json.dumps({"downloaded_verified": len(receipts), "total": 120}), flush=True)
        return row, receipt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        acquired = list(pool.map(acquire, selection))
    observations, evaluation = [], []
    for row, receipt in acquired:
        sample_id = "esc50_" + row["filename"][:-4]
        observation = {"sample_id": sample_id, "audio_path": receipt["path"],
                       "source_id": row["src_file"]}
        observations.append(observation)
        evaluation.append({**observation, "label": row["category"], "target": int(row["target"]),
                           "fold": int(row["fold"]), "role": row["role"],
                           "known_primary_label": row["known"],
                           "dataset": "ESC-50", "split": "development", "duration_s": 5.0,
                           "audio_sha256": receipt["sha256"], "license": receipt["license"]})
    for filename, content in (("observations.jsonl", observations), ("evaluation.jsonl", evaluation),
                              ("audio_download_receipts.jsonl", sorted(receipts, key=lambda r: r["filename"]))):
        (data_root / filename).write_text("".join(json.dumps(r) + "\n" for r in content), encoding="utf-8")
    summary = {"status": "complete", "dataset": "ESC-50", "evidence_role": "Development",
               "clips": len(observations), "duration_s": len(observations) * 5,
               "roles": dict(Counter(r["role"] for r in evaluation)),
               "role_known_primary_label": dict(Counter(f"{r['role']}_{r['known_primary_label']}" for r in evaluation)),
               "source_ids": len({r["source_id"] for r in evaluation}),
               "source_cross_role_overlap": 0, "audio_payload_bytes": sum(r["bytes"] for r in receipts),
               "observed_audio_network_bytes": network["observed_audio_bytes"],
               "metadata_extra_transfer_upper_bound_bytes": 20_000_000,
               "wall_s": time.monotonic() - started, "backend": "CPU", "gpu_used": False,
               "placement_reason": "TASK_NOT_GPU_SUITABLE", "audio_revision": revision,
               "observations_sha256": digest((data_root / "observations.jsonl").read_bytes()),
               "evaluation_sha256": digest((data_root / "evaluation.jsonl").read_bytes()),
               "task_proxy": "主类别标签集外拒识代理",
               "limits": "Dominant-event labels; known_primary_label does not prove any individual sound absent. Not exhaustive polyphonic tags or real-world false-alarm truth. Held-out vocabulary is not necessarily unseen during model pretraining. No user-hearing benefit measured."}
    write_json(data_root / "data_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
