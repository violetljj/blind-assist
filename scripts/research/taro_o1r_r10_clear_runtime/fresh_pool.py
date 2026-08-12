#!/usr/bin/env python3
"""Freeze the metadata-only 32-parent R10 source-only candidate pool."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as metadata


SCHEMA = "blindassist.taro.o1r.r10_fresh_source_only_candidate_pool.v1"
SELECTION_SALT = "TARO_O1R_R10_FRESH_SOURCE_ONLY_CLEAR_ENRICHED_POOL_V1"
ROLE = "SOURCE_ONLY_POOL_CANDIDATE"
PARENT_COUNT = 32
EXCLUSION_COMMIT = "ca1ef83ff88788225d37b0a7ac673091d59d004f"
EXPECTED_EXCLUDED_COUNT = 320
EXPECTED_EXCLUDED_SHA256 = "73D46A21B1FF5BC7FD16FA8D6BC79F5A68319CB26D46A57E80E4C214B55C4A3A"
EXPECTED_ELIGIBLE_COUNT = 1544
EXPECTED_POOL = (
    ("469615", "47430839", "005B83B70B20657767EB74E218197A50748290E836FCD687EA21D4082D630EC6"),
    ("467152", "47333281", "006614491AD7FAF7643AEEB49E5CF1526E671310E275182BA9C48441A741F4C8"),
    ("421657", "42445633", "007822CF0A1A6E3DE65981FFD1C19E285510F47EBFD140E2C94A5E9AB428ACA8"),
    ("472622", "47430739", "00C1CF2DB9A05D5554B76AD5525E026F87C965704205BF4127F15E09BB37FF0A"),
    ("435664", "42899970", "00DB79D1B5B513CF0C6E374A64EA0F1D8F841BFEAB6BCD6FC03566EE4BEFFF7E"),
    ("470335", "47429663", "010E8DB6A7D8151659A02A129CE42A5278CF0CFBFE0F215EE1A11AA8BD4A8BDB"),
    ("464755", "44796189", "011801FB3F1CD8C7081520B2C6C093155C38B8333B4E08FC192F6D63481F1CB4"),
    ("472306", "47204818", "01230ADEC5EE3DD5EA9801DBD8635CD2F369569460464762BD6B616F114154AE"),
    ("471014", "47204325", "0131EFDD9C4CE0F1685EAC69BA6F7AA18F68607C999CC723AE30F2EE4DADD0A3"),
    ("426166", "42898458", "01651B712E47830A61A8FF94B2CC18EB1E3FC0340D0BEBFF70C791EF9D1D6976"),
    ("469749", "47430894", "01852E252C7EF20A82D68A96129D2D4FBBAF035A0EB1850A5959DD33E63FCBD6"),
    ("422163", "42445676", "01B22D14C49F27ACB80541A4E8FC189576AE52257204CF235B209517F16DFA48"),
    ("484715", "48458678", "01B60958EAE398E9F4AD2949CE3A954352B2DB34A90AE73261641BEE3E9FCDE3"),
    ("482231", "47670023", "01D249775D9781452FD2153F85453494C276E4C6A5559864C6B184381A876A89"),
    ("421254", "42444754", "01EA8FC0D9935941A6CD886A1EE127AB2BEFADAC536AA1125A34F30FCEA81E59"),
    ("426247", "42898191", "02015FFB7F369BB26030105388E3914EF67BF3BD3C6C8F729FC87184B35AAB4D"),
    ("434903", "42899259", "02121F43E1F10B0414B301230BA3F2B202E29B1A1F0E14547AFF74146A3CF35D"),
    ("472477", "47204828", "021B1BE83CF4E5E23FFBBB5D078B55F539A38968EAAB2E82F0AAB02BB022BB80"),
    ("423070", "42447202", "02487D80D959EF360B6C7FE4D8D84B717B2F451B85B33FDF76F86F992E414889"),
    ("483249", "48458282", "025F397504866B9788CD920636DE399635A4904633A5AD8353DB5C6C6CEEC376"),
    ("426262", "42898449", "0261609C59ED0D9A60E2031D9D0174076D78BF7CEB9D35092CD8DEBF3DCCAB21"),
    ("470872", "47430097", "0275D20E9B4C1FCA110498F38EBC572524693412C76F06B735D7528CB013D929"),
    ("466906", "47331621", "02B37365E308F7E18466B2CFE4DF9A056DC2351946711A1D61497BBAE526FB3F"),
    ("435363", "42899774", "02BC784D2E0DE11E502211A65853B7B53C9B412BFFD1F81C2B0264D5A8636CE7"),
    ("469278", "47332748", "02D703578AD41BFD90B828BACB88CBC02238F33E7DE58C38EE8DDCF988C45F19"),
    ("466896", "45261344", "030762E651AE25F57F751E0402D86F2D62BDFD271B1E8581F4A2FB0EDD4F83BF"),
    ("471674", "47332312", "0321EAFE614A8DB5ECC3B1728029D9B1C53FAECEED7652196A162A488FE3B63D"),
    ("468380", "45261722", "0349BC7C7B5CBAC5D5AEBBD6B21E35B0092FD04D06F1601686D83F24425E650C"),
    ("468146", "47334493", "03602D2D9D19DB6195F107AB12084A2C447797A36E6D8CB362BBA8CFF53A07B4"),
    ("471431", "47429841", "0378E6B8E8D0B4EA6EADB0D15477F9FAB4C6E8786A81350827CE18DA3B837102"),
    ("422148", "42445697", "039A92A63D50F2FDBF91EDBF62350DB093465498F6D0A04410298B0AD168D82F"),
    ("482315", "47670063", "039EC3C6059F71A41D9BDF84B72C71197E5428B83B96173E198BB4725287995B"),
)
ASSET_TEMPLATES = (
    ("upsampling.zip", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/Training/{video_id}.zip"),
    ("lowres_wide_intrinsics.zip", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide_intrinsics.zip"),
    ("lowres_wide.traj", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide.traj"),
)


class FreshPoolError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise FreshPoolError(code)


def _rank(visit_id: str, video_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{ROLE}:{visit_id}:{video_id}".encode("ascii")).hexdigest().upper()


def _excluded(root: Path, known: set[str]) -> list[str]:
    result = subprocess.run(["git", "-C", str(root), "grep", "-h", "-o", "-P", r"(?<![0-9])[0-9]{6,8}(?![0-9])", EXCLUSION_COMMIT, "--", *metadata.PATHSPECS], check=True, capture_output=True, text=True, encoding="utf-8")
    return sorted({value for value in result.stdout.splitlines() if value in known})


def build_pool(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    for relative, (size, digest) in metadata.METADATA_BINDINGS.items():
        path = root / relative
        require(path.is_file() and path.stat().st_size == size and metadata._sha256(path) == digest, f"R10_METADATA_BINDING:{relative}")
    upsampling = metadata._rows(root / metadata.UPSAMPLING_METADATA)
    raw = metadata._rows(root / metadata.RAW_METADATA)
    threedod = metadata._rows(root / metadata.THREEDOD_METADATA)
    known = {value for row in upsampling for value in (row["visit_id"], row["video_id"]) if value != "NA"}
    excluded = _excluded(root, known)
    exclusion_sha = hashlib.sha256(("\n".join(excluded) + "\n").encode("utf-8")).hexdigest().upper()
    require(len(excluded) == EXPECTED_EXCLUDED_COUNT and exclusion_sha == EXPECTED_EXCLUDED_SHA256, "R10_EXCLUSION_DRIFT")
    excluded_set = set(excluded)
    raw_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in raw}
    threedod_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in threedod}
    eligible = [
        {"visit_id": row["visit_id"], "video_id": row["video_id"], "official_fold": "Training", "pool_rank_sha256": _rank(row["visit_id"], row["video_id"])}
        for row in upsampling
        if row["fold"] == "Training" and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded_set and row["video_id"] not in excluded_set
        and (row["visit_id"], row["video_id"], "Training") in raw_keys
        and (row["visit_id"], row["video_id"], "Training") in threedod_keys
    ]
    require(len(eligible) == EXPECTED_ELIGIBLE_COUNT, "R10_ELIGIBLE_COUNT_DRIFT")
    pool = []
    visits = set()
    for row in sorted(eligible, key=lambda item: item["pool_rank_sha256"]):
        if row["visit_id"] in visits:
            continue
        pool.append(row)
        visits.add(row["visit_id"])
        if len(pool) == PARENT_COUNT:
            break
    require(tuple((row["visit_id"], row["video_id"], row["pool_rank_sha256"]) for row in pool) == EXPECTED_POOL, "R10_POOL_ROSTER_DRIFT")
    requests = [{"visit_id": row["visit_id"], "video_id": row["video_id"], "asset": asset, "url": template.format(video_id=row["video_id"])} for row in pool for asset, template in ASSET_TEMPLATES]
    return validate_pool({
        "schema": SCHEMA,
        "status": "METADATA_ONLY_FRESH_POOL_FROZEN_NO_NETWORK_AUTHORITY",
        "selection_salt": SELECTION_SALT,
        "role": ROLE,
        "exclusion_snapshot_commit": EXCLUSION_COMMIT,
        "excluded_identity_count": len(excluded),
        "excluded_identities_sha256": exclusion_sha,
        "eligible_row_count": len(eligible),
        "pool_parent_count": len(pool),
        "pool": pool,
        "request_plan": {"method": "HEAD", "response_body_bytes_allowed": 0, "request_count": len(requests), "expanded_requests_sha256": adapter.canonical_sha256(requests), "requests": requests},
        "source_payload_read": False,
        "faro_payload_read": False,
        "model_output_read": False,
        "network_authorized": False,
        "unknown_is_negative": False,
    }, repo_root=root, recompute=False)


def validate_pool(value: Mapping[str, Any], *, repo_root: Path, recompute: bool = True) -> dict[str, Any]:
    plan = json.loads(json.dumps(dict(value)))
    require(plan.get("schema") == SCHEMA and plan.get("status") == "METADATA_ONLY_FRESH_POOL_FROZEN_NO_NETWORK_AUTHORITY", "R10_POOL_IDENTITY")
    require(tuple((row.get("visit_id"), row.get("video_id"), row.get("pool_rank_sha256")) for row in plan.get("pool", [])) == EXPECTED_POOL, "R10_POOL_ROSTER_DRIFT")
    request = plan.get("request_plan", {})
    require(request.get("method") == "HEAD" and request.get("response_body_bytes_allowed") == 0 and request.get("request_count") == PARENT_COUNT * 3 and adapter.canonical_sha256(request.get("requests")) == request.get("expanded_requests_sha256"), "R10_REQUEST_PLAN_DRIFT")
    require(plan.get("source_payload_read") is plan.get("faro_payload_read") is plan.get("model_output_read") is plan.get("network_authorized") is False and plan.get("unknown_is_negative") is False, "R10_POOL_AUTHORITY")
    if recompute:
        require(plan == build_pool(repo_root), "R10_POOL_RECOMPUTE_DRIFT")
    return plan


if __name__ == "__main__":
    print(json.dumps(build_pool(Path(__file__).resolve().parents[3]), sort_keys=True, separators=(",", ":")))
