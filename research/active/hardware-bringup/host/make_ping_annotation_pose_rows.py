"""Recover the original regular pose rows in the PING annotation time stream."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

PING_SHA256 = "45e36ddfb9a1c876b135233b9f3a215c61df133cba2e87116a9f3807eb26760e"
ENV_SHA256 = "9e316f252ce39ee8495cc7564160901a0e0f27d9e24ad79e449eccee4be6a103"
MEMBER = "datasheet/trajectoryData.csv"


def read_source(mat_path, zip_path):
    for path, expected in ((mat_path, PING_SHA256), (zip_path, ENV_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected source bytes: {path}")
    with ZipFile(zip_path) as archive:
        text = archive.read(MEMBER).decode("utf-8")
    rows = list(csv.DictReader(StringIO(text)))
    participants = loadmat(mat_path, simplify_cells=True, variable_names=["trajdata"])["trajdata"]
    return rows, participants


def checked_pose_rows(person, rows, participant):
    time = participant["gttime"]
    if len(rows) != len(time) or np.any(np.diff(time) < 0):
        raise ValueError(f"Annotation rows or timestamp order disagree for {person}")
    ground = np.array([(float(row["x'"]), float(row["y'"])) for row in rows])
    estimated = np.array([(float(row["x"] or "nan"), float(row["y"] or "nan")) for row in rows])
    observed = np.all(np.isfinite(estimated), axis=1)
    checks = [
        np.allclose(ground, participant["gtpos"], rtol=0, atol=1e-7),
        np.allclose(estimated[observed], participant["estpos"][observed], rtol=0, atol=1e-7),
        [int(row["id3"][1:]) for row in rows] == participant["gtroute"].tolist(),
        [int(row["floor"]) for row in rows] == participant["gtfloor"].tolist(),
    ]
    if not all(checks):
        raise ValueError(f"CSV and MAT ordered rows disagree for {person}")
    pose_indices = [idx for idx, row in enumerate(rows) if row["code"] == "Pose"]
    for idx in pose_indices:
        # Regular Pose time2 labels have precision 0.1s. HH:MM:SS and MM:SS.s
        # appear in the CSV; compare the final fields circularly within an hour.
        minutes, seconds = rows[idx]["time2"].split(":")[-2:]
        csv_ms = (int(minutes) * 60 + float(seconds)) * 1000
        diff_ms = (csv_ms - int(time[idx]) + 1_800_000) % 3_600_000 - 1_800_000
        if abs(diff_ms) > 50.01:
            raise ValueError(f"CSV and MAT clock disagree for {person}, row {idx}")
    pose_time = [int(time[idx]) for idx in pose_indices]
    counts = Counter(pose_time)
    # Preserve ambiguous source rows in the receipt. Neither duplicate is a
    # reproducible observation at the shared timestamp if their coordinates differ.
    unambiguous = [idx for idx in pose_indices if counts[int(time[idx])] == 1]
    return {
        "pose_row_indices": pose_indices,
        "pose_time_ms": pose_time,
        "unambiguous_pose_row_indices": unambiguous,
        "duplicate_pose_timestamps_ms": [t for t, count in counts.items() if count > 1],
        "valid_csv_estimated_position_rows": int(np.count_nonzero(observed)),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=here / "data.mat")
    parser.add_argument("--zip", type=Path, default=here.parent / "envfactors" / "CHI_code.zip")
    parser.add_argument("--out", type=Path, default=here / "annotation_pose_rows.json")
    args = parser.parse_args()
    rows, participants = read_source(args.mat, args.zip)
    by_person = {}
    for index, participant in enumerate(participants, 1):
        person = f"P{index}"
        by_person[person] = checked_pose_rows(person, [row for row in rows if row["id2"] == person], participant)
    if sum(len(u["gttime"]) for u in participants) != len(rows):
        raise ValueError("CSV contains unexpected annotation rows")
    receipt = {
        "ann_source": "https://envfactors.github.io/code/CHI_code.zip",
        "ann_member": MEMBER,
        "ann_zip_sha256": ENV_SHA256,
        "ping_mat_sha256": PING_SHA256,
        "annotation_rows": len(rows),
        "pose_rows": sum(len(u["pose_row_indices"]) for u in by_person.values()),
        "unambiguous_pose_rows": sum(len(u["unambiguous_pose_row_indices"]) for u in by_person.values()),
        "csv_estimated_position_rows_missing": len(rows) - sum(
            u["valid_csv_estimated_position_rows"] for u in by_person.values()
        ),
        "position_control": (
            "All ordered participant rows match annotation xy, route and floor; "
            "all non-missing estimated xy also match to 1e-7 m. "
            "Pose clocks agree to the rounded source CSV precision of 0.1 s."
        ),
        "index_origin": "0-based, original unsliced participant vector in PING data.mat",
        "pose_time_ms": {p: u["pose_time_ms"] for p, u in by_person.items()},
        "people": by_person,
    }
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"{receipt['annotation_rows']} joined rows, {receipt['pose_rows']} Pose, "
        f"{receipt['unambiguous_pose_rows']} with unique timestamps"
    )


if __name__ == "__main__":
    main()
