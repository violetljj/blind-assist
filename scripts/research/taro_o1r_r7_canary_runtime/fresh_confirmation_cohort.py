from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


SCHEMA = "blindassist.taro.o1r.r7_fresh_confirmation_cohort_candidate.v1"
PROTOCOL_SHA256 = "1419070D09951AE7251C9832EF006C329F82D1DA1C46DB8F759ABBF6ECA11A01"
EXCLUSION_COMMIT = "5902304924e773c9d113178056d45e14559fd0cf"
SELECTION_SALT = "TARO_O1R_R7_FRESH_POSITIVE_OCCUPANCY_CONFIRMATION_V1"
ROLE = "FRESH_CONFIRMATION"
PARENT_COUNT = 8
UPSAMPLING_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/depth_upsampling/upsampling_train_val_splits.csv"
RAW_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/raw/raw_train_val_splits.csv"
THREEDOD_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/threedod/3dod_train_val_splits.csv"
METADATA_BINDINGS = {
    UPSAMPLING_METADATA: (59280, "17935C5567F3004EA01BE394D6DC9EEC4ED96F0A7E097884C8B648578CCA2F6B"),
    RAW_METADATA: (132940, "F93CD6A1EC0AEA5E103313F3BB4660744B011A1EAA8AA44A992E2C7C2966B145"),
    THREEDOD_METADATA: (132310, "B753C50B830076A8396A352C60A49D060EC00D7A91290147BC9DC374697519CE"),
}
PATHSPECS = [
    ":(glob)docs/**/*.md",
    ":(glob)docs/**/*.json",
    ":(glob)scripts/**/*.md",
    ":(glob)scripts/**/*.json",
    "DATASET_MASTER_LEDGER.csv",
]
EXPECTED_EXCLUDED_COUNT = 202
EXPECTED_EXCLUDED_SHA256 = "2732A3442054F4D7F2E2897E21AAC95D6B4C659F516A80C1D0B60E5FBD116C3A"
EXPECTED_ELIGIBLE_COUNT = 1731
EXPECTED_ROSTER = [
    ("478025", "47895446", "0022857814CECEC19FB91297CDBA3F5B3FB6A2D009F821FB0AD72E2F9F1F0416"),
    ("455339", "44358418", "00409A59EC8C5207ABE7970FEDCD9971F48C8D80C9372D66F844BBAF26411B41"),
    ("437126", "43649393", "00414689E5F5DDF82DFEB7279EA1595E1D1A9546CE9BE7B935F94CCCB22EDBEC"),
    ("434689", "42898502", "0056040E68DF8994AC8E502DA03B5DA10A5FFB1DDAA9E587582612CAAAE1A697"),
    ("437294", "43649767", "009F17C4E55BD7C7A70E16A0D3799D72C97F8FF7BDD4CD783C74E4776C3F5D79"),
    ("469452", "47333159", "0103C099C551A9AB2C4E072D306D60D1EBACF60BE8CB3038767520CD1EBD219E"),
    ("421069", "42444477", "010B31EA27021743358D2BE1C12B67B0F1296024182C610DACFA165C70372345"),
    ("469635", "47115118", "0128F2C1D37B75BD81AE7A584AC81D1CB9A9CCE1FCB3324BF46E3A5E6653AFC9"),
]


class FreshCohortError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise FreshCohortError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows) and set(rows[0]) == {"video_id", "visit_id", "fold"}, "R7_FRESH_METADATA_SCHEMA")
    return rows


def _excluded(repo_root: Path, known_ids: set[str]) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "grep", "-h", "-o", "-P", r"(?<![0-9])[0-9]{6,8}(?![0-9])", EXCLUSION_COMMIT, "--", *PATHSPECS],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return sorted({value for value in result.stdout.splitlines() if value in known_ids})


def _rank(visit_id: str, video_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{ROLE}:{visit_id}:{video_id}".encode("ascii")).hexdigest().upper()


def build_plan(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    for relative, (size, digest) in METADATA_BINDINGS.items():
        path = root / relative
        _require(path.is_file() and path.stat().st_size == size and _sha256(path) == digest, f"R7_FRESH_METADATA_BINDING:{relative}")
    upsampling = _rows(root / UPSAMPLING_METADATA)
    raw = _rows(root / RAW_METADATA)
    threedod = _rows(root / THREEDOD_METADATA)
    known = {value for row in upsampling for value in (row["visit_id"], row["video_id"]) if value != "NA"}
    excluded = _excluded(root, known)
    exclusion_sha = hashlib.sha256(("\n".join(excluded) + "\n").encode("utf-8")).hexdigest().upper()
    _require(len(excluded) == EXPECTED_EXCLUDED_COUNT and exclusion_sha == EXPECTED_EXCLUDED_SHA256, "R7_FRESH_EXCLUSION_DRIFT")
    excluded_set = set(excluded)
    eligible = [
        {
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "official_fold": "Training",
            "selection_rank_sha256": _rank(row["visit_id"], row["video_id"]),
        }
        for row in upsampling
        if row["fold"] == "Training" and row["visit_id"] != "NA" and row["visit_id"] not in excluded_set and row["video_id"] not in excluded_set
    ]
    _require(len(eligible) == EXPECTED_ELIGIBLE_COUNT, "R7_FRESH_ELIGIBLE_COUNT_DRIFT")
    roster: list[dict[str, str]] = []
    visits: set[str] = set()
    for row in sorted(eligible, key=lambda value: value["selection_rank_sha256"]):
        if row["visit_id"] in visits:
            continue
        roster.append(row)
        visits.add(row["visit_id"])
        if len(roster) == PARENT_COUNT:
            break
    observed = [(row["visit_id"], row["video_id"], row["selection_rank_sha256"]) for row in roster]
    _require(observed == EXPECTED_ROSTER, "R7_FRESH_ROSTER_DRIFT")
    raw_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in raw}
    threedod_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in threedod}
    for row in roster:
        key = (row["visit_id"], row["video_id"], "Training")
        _require(key in raw_keys and key in threedod_keys, f"R7_FRESH_CROSS_SPLIT_MISSING:{row['video_id']}")
    prior_parents = {parent for parent, _ in adapter.ADAPTER_FIT_ROSTER + adapter.O0R_EVAL_CANDIDATE_ROSTER} | set(prospective_factor_runtime.FORBIDDEN_R6_UNTOUCHED_PARENTS)
    _require(not ({row["visit_id"] for row in roster} & prior_parents), "R7_FRESH_PRIOR_PARENT_OVERLAP")
    assets = [
        {"asset": "upsampling.zip", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/Training/{video_id}.zip"},
        {"asset": "lowres_wide_intrinsics.zip", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide_intrinsics.zip"},
        {"asset": "lowres_wide.traj", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide.traj"},
    ]
    requests = [
        {"visit_id": row["visit_id"], "video_id": row["video_id"], "asset": asset["asset"], "url": asset["url_template"].format(video_id=row["video_id"])}
        for row in roster
        for asset in assets
    ]
    plan = {
        "schema": SCHEMA,
        "status": "METADATA_ONLY_FRESH_COHORT_FROZEN",
        "protocol_sha256": PROTOCOL_SHA256,
        "source": {
            "repository_commit": "7283761bf26c27570ec59a5dc0f8686fbff07726",
            "official_fold": "Training",
            "metadata_bindings": [{"path": path, "bytes": size, "sha256": digest} for path, (size, digest) in METADATA_BINDINGS.items()],
        },
        "selection": {
            "exclusion_snapshot_commit": EXCLUSION_COMMIT,
            "matched_official_identity_count": len(excluded),
            "matched_official_identities_sha256": exclusion_sha,
            "selection_salt": SELECTION_SALT,
            "selection_rule": "Rank SHA256('{salt}:{role}:{visit_id}:{video_id}') over official Training rows absent from the frozen repository exclusion snapshot; retain the first video per unique visit.",
            "eligible_row_count": len(eligible),
            "role": ROLE,
            "roster": roster,
        },
        "request_plan": {
            "method": "HEAD",
            "response_body_bytes_allowed": 0,
            "request_count": len(requests),
            "expanded_requests_sha256": adapter.canonical_sha256(requests),
            "requests": requests,
        },
        "invariants": {
            "parent_count": len(roster),
            "unique_visit_count": len(visits),
            "present_in_all_three_official_split_tables": True,
            "prior_taro_parent_overlap": 0,
            "model_or_truth_inputs_read": False,
            "media_body_bytes_read": False,
        },
        "authority": {
            "metadata_selection": True,
            "head_requests": False,
            "source_download": False,
            "source_decode": False,
            "model_execution": False,
            "truth_scoring": False,
            "training": False,
        },
        "unique_successor": "TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK",
    }
    return validate_plan(plan, repo_root=root, recompute=False)


def validate_plan(value: Mapping[str, Any], *, repo_root: Path, recompute: bool = True) -> dict[str, Any]:
    plan = json.loads(json.dumps(dict(value)))
    _require(plan.get("schema") == SCHEMA and plan.get("protocol_sha256") == PROTOCOL_SHA256, "R7_FRESH_PLAN_IDENTITY_DRIFT")
    roster = plan.get("selection", {}).get("roster", [])
    observed = [(row.get("visit_id"), row.get("video_id"), row.get("selection_rank_sha256")) for row in roster]
    _require(observed == EXPECTED_ROSTER, "R7_FRESH_ROSTER_DRIFT")
    requests = plan.get("request_plan", {})
    _require(requests.get("method") == "HEAD" and requests.get("response_body_bytes_allowed") == 0 and requests.get("request_count") == 24 and adapter.canonical_sha256(requests.get("requests")) == requests.get("expanded_requests_sha256"), "R7_FRESH_REQUEST_PLAN_DRIFT")
    _require(plan.get("authority") == {"metadata_selection": True, "head_requests": False, "source_download": False, "source_decode": False, "model_execution": False, "truth_scoring": False, "training": False}, "R7_FRESH_PLAN_AUTHORITY_DRIFT")
    _require(plan.get("unique_successor") == "TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK", "R7_FRESH_PLAN_SUCCESSOR_DRIFT")
    if recompute:
        _require(plan == build_plan(repo_root), "R7_FRESH_PLAN_RECOMPUTE_DRIFT")
    return plan


def main() -> int:
    plan = build_plan(Path(__file__).resolve().parents[3])
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
