#!/usr/bin/env python3
"""Freeze the metadata-only 24-parent R8 source-only selection pool."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as r7_cohort


POOL_SALT = "TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_POOL_V1"
POOL_ROLE = "SOURCE_ONLY_POOL"
POOL_PARENT_COUNT = 24
EXPECTED_ELIGIBLE_COUNT = 1689
EXPECTED_POOL = [
    ("468296", "47431071", "000F861F94B365C25A444318D86FE83B8285C48DF53B9F7AEEDB353599FF1127"),
    ("482090", "47669984", "0038F6C3A6798C8640C98DCADD34639FB7C5E36E34976078FDF6F56DE6A99A3B"),
    ("482095", "47895719", "0040BDE47DB8D7F1BDDA804D6EC11F983D38BE11F349E696D11FBB85E94A7A05"),
    ("470945", "47115624", "006177DF3C152DBB7EE648744C5589C000393E0EFBBB68A50A916EF0D136DE62"),
    ("426150", "42898750", "0077BC28395B955F503812E0857CCED4CD5B31E3860C4E63D6FD7CDC794F3775"),
    ("437450", "43828149", "00BFBF6BAA4F37A45C01D44A32557B48233618D66EBAF40788411C6D3FC9F369"),
    ("482115", "47895520", "00CF9CFC2E514A4C47055E3A96FC40D80A3939BA78473208B01783BD3F3B8B6B"),
    ("435730", "42899817", "00E90736BAFDABF78304F662D9804146F8430B8D2CB1215A688F4D2CF92FE252"),
    ("466911", "47331686", "00EF15A10D248952CAA544E9A7B9DE32FC536C68419F306C4F723035B3F6E6E8"),
    ("435353", "42899542", "0101C363EE7FBE1E86E3E4F0A4E9B32C0451751D7AC3398549B79FDBC520F7DB"),
    ("466111", "44796477", "0115CC62847FC92D9569A914B33CC6216649E6B7EB587F106247B5C7F168E19D"),
    ("469837", "47334046", "0126C1ABDBA10DC3151F2FAFFC9D25EA4C52EDCD7567D126714C6F97ED0C2AB8"),
    ("469618", "47430832", "01280CDD7334332914D5EFF98D0D3EE196F70E0E850267C4BF5CE140BDC0C20D"),
    ("467324", "45261562", "01AA0A17AA082C8E54D8D2BEA0F5908CD2BB4C6DDCCBEC5C2E9B47B7115AC923"),
    ("423770", "42898065", "01C9B033328792AC52765E7752166A703C50C4B53E2F733F4A2DF367EFF414FD"),
    ("467344", "47333964", "01E03E358FF88918B35CC1747CF71F0905C6D2E22E29A9BE08D5F4D38A7234A3"),
    ("435729", "42899650", "028CF5ABAE14561FDBBD3B36F82493809598B8F95B7A1DBA45C0469F0BD5BF2B"),
    ("422155", "42445684", "02C0ACD3008CDF322343B2D6A1522E743D708FFDB32EE71EEECA803750BEC426"),
    ("467304", "47333988", "033E4A3E12C935EF11463155B508BED1DED045601B6701C6E034005A3BD8E824"),
    ("482761", "47670197", "0370E2AB54E5E6A3BC018ADB725966AA91119A319F3748B8F486AB4F247F8336"),
    ("421264", "42444885", "0371DECE18AC97082D148D8764E060C96B726B63EA21C43C5C5C047BC2978515"),
    ("471106", "47332083", "0375335FA08AE6DE8D12A425D3723D054DA8D959A47D30CF72AF994BD59BD439"),
    ("423611", "42898089", "03A5F393D018F767E9A277E3A43BF148815699157A410C173DA4ECEAEBA183EB"),
    ("467345", "47333963", "03AF584BD29C4203B57FB092464EFDBB094DB05C07A464341C044086155EC57A"),
]


class R8PoolError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise R8PoolError(code)


def _rank(visit_id: str, video_id: str) -> str:
    return hashlib.sha256(f"{POOL_SALT}:{POOL_ROLE}:{visit_id}:{video_id}".encode("ascii")).hexdigest().upper()


def build_pool(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    for relative, (size, digest) in r7_cohort.METADATA_BINDINGS.items():
        path = root / relative
        require(path.is_file() and path.stat().st_size == size and r7_cohort._sha256(path) == digest, f"R8_POOL_METADATA_BINDING:{relative}")
    upsampling = r7_cohort._rows(root / r7_cohort.UPSAMPLING_METADATA)
    raw = r7_cohort._rows(root / r7_cohort.RAW_METADATA)
    threedod = r7_cohort._rows(root / r7_cohort.THREEDOD_METADATA)
    known = {value for row in upsampling for value in (row["visit_id"], row["video_id"]) if value != "NA"}
    excluded = set(r7_cohort._excluded(root, known))
    excluded.update(value for row in r7_cohort.EXPECTED_ROSTER for value in row[:2])
    raw_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in raw}
    threedod_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in threedod}
    eligible = [
        {"visit_id": row["visit_id"], "video_id": row["video_id"], "official_fold": "Training", "pool_rank_sha256": _rank(row["visit_id"], row["video_id"])}
        for row in upsampling
        if row["fold"] == "Training" and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded and row["video_id"] not in excluded
        and (row["visit_id"], row["video_id"], "Training") in raw_keys
        and (row["visit_id"], row["video_id"], "Training") in threedod_keys
    ]
    require(len(eligible) == EXPECTED_ELIGIBLE_COUNT, "R8_POOL_ELIGIBLE_COUNT_DRIFT")
    pool = []
    visits = set()
    for row in sorted(eligible, key=lambda item: item["pool_rank_sha256"]):
        if row["visit_id"] in visits:
            continue
        pool.append(row)
        visits.add(row["visit_id"])
        if len(pool) == POOL_PARENT_COUNT:
            break
    observed = [(row["visit_id"], row["video_id"], row["pool_rank_sha256"]) for row in pool]
    require(observed == EXPECTED_POOL, "R8_POOL_ROSTER_DRIFT")
    assets = (
        ("upsampling.zip", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/Training/{video_id}.zip"),
        ("lowres_wide_intrinsics.zip", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide_intrinsics.zip"),
        ("lowres_wide.traj", "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide.traj"),
    )
    requests = [{"visit_id": row["visit_id"], "video_id": row["video_id"], "asset": asset, "url": template.format(video_id=row["video_id"])} for row in pool for asset, template in assets]
    return {
        "schema": "blindassist.taro.o1r.r8_source_only_clear_pool.v1",
        "status": "METADATA_ONLY_POOL_FROZEN_NO_NETWORK_AUTHORITY",
        "selection_salt": POOL_SALT,
        "eligible_row_count": len(eligible),
        "pool_parent_count": len(pool),
        "pool": pool,
        "request_plan": {"method": "HEAD", "response_body_bytes_allowed": 0, "request_count": len(requests), "expanded_requests_sha256": r7_cohort.adapter.canonical_sha256(requests), "requests": requests},
        "r7_fresh_parent_overlap_count": 0,
        "source_payload_read": False,
        "faro_payload_read": False,
        "network_authorized": False,
    }


__all__ = ["EXPECTED_POOL", "POOL_PARENT_COUNT", "R8PoolError", "build_pool"]
