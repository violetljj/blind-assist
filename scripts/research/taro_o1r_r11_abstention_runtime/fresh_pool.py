#!/usr/bin/env python3
"""Freeze the metadata-only 48-parent R11 fresh confirmation pool."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as metadata
from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool as r10_pool


SCHEMA = "blindassist.taro.o1r.r11_fresh_source_only_candidate_pool.v1"
SELECTION_SALT = "TARO_O1R_R11_FRESH_DUAL_CLASS_CONFIRMATION_POOL_V1"
ROLE = "SOURCE_ONLY_POOL_CANDIDATE"
PARENT_COUNT = 48
SELECTED_PARENT_COUNT = 24
EXCLUSION_COMMIT = "0fe2f78b9000b59a111f3a3a84e7789a5dcac415"
EXPECTED_SNAPSHOT_EXCLUDED_COUNT = 446
EXPECTED_UNION_EXCLUDED_COUNT = 494
EXPECTED_UNION_EXCLUDED_SHA256 = "183AA9A2CDFE6B0DD8E8E6B3FC181D006CB9CDEBF1B0EA59A8C6F856CC8E929B"
EXPECTED_ELIGIBLE_COUNT = 1297
EXPECTED_POOL = (
    ("466160", "44796584", "000716D67871E16131E22B7576CD38947F17290423102175C1E597ECC3F86DEC"),
    ("437986", "43895956", "003965F207C60BA89C529CB57866D4511BC304A283AD461EB6C1E6E613CF6416"),
    ("422200", "42445103", "00EC1BE8A72B6FF0A3A2E6BFE0B7C765A74D89C32852FE8AA97A9AC809C014B6"),
    ("469411", "47333205", "0111B4AE2545D3B5646C0A4F26A164683CBD51F90059A9DE0E1494998BDAC61F"),
    ("471148", "47332195", "015534960561E5580C8047DEA191AD24169054365D4B756D03BE8873A651828F"),
    ("466879", "45261357", "017F691B3BA1065777F9B32E068E6B8D1969B850CD6D0A2ACC7BFF496921E981"),
    ("464233", "44358660", "01803353EB956CAB5BBE7F5C79FF65BAD01C17332C4E49A89605A9FEDDE7344D"),
    ("468393", "45261653", "01C06A31A30703A4C2225C2A17EBA960CA5EFB18CD3AAEBF692BD68D38ED5F3D"),
    ("482763", "47670233", "01D2F7B210A36384204BA39F80C110E7D015145CEC11D2F77B97032B3CF668C1"),
    ("421667", "42445612", "02082C43E470E855331011CB10CDBBA71CF3F5A98BA02260EBF90BF4DEDF8E19"),
    ("442392", "44358328", "020FF620257BBA45A4BF5134A894F9FA60E8A73E532A14794241444ABBFAE8AE"),
    ("470551", "47331518", "021F9FF1E783AAB92991C019AAF6A11407A6C89DF9EC232ED313C7476E7B6B64"),
    ("435324", "42899216", "023134E940B8A10B1FF5C052F7A7C003055ADAD93AC37F4E7B685BF7F3277D5A"),
    ("467181", "47333239", "024137D3DA3FC89D0E6124B0E26A9B8910F7E7C8CDDB0A0BE173F14CAE329AA7"),
    ("472050", "47895250", "025FD5A8D2E2ACA575AE3E43B523DA1552E2C212437D317028EDC45D2E120CE8"),
    ("468275", "47331298", "0264946A83C303B5E51F368996DE330072D36D8104A229C31C17D32A54A86A71"),
    ("422010", "42445999", "02752FF4C96BFD60D68F43CA87F52B86FAD961EB756FCEB7DFB89A264D8A301F"),
    ("423966", "42898332", "02DF53A7B3719493FD863C4AD9483F7FC2789A7BE953A0205ABFEA284796BB86"),
    ("471429", "47429750", "03DD78A95D9A7D8A5B165D9B994B98F0165333B5950CFDFAF89DFAD7DC6FAA83"),
    ("435359", "42899489", "0424D6D5ED20556B46EFDC41DF598F62686F9DCA862F0BD546FBA48E30D98006"),
    ("471103", "47204305", "049F77424152DEE40027B90B9C34D8E9012E81BE3431461DCC52D0EDD7C640C3"),
    ("421392", "42444933", "04ABFF6C9462D7B919058288235F00BD27475965CB1AA95A45F9D33866374D18"),
    ("471672", "47895199", "04AF7F65E332BB66FF33FF60C2210A5FCEFE5ADBB597FCBDA81F83732B792570"),
    ("422384", "42446607", "04BF8BAFA7A4039AF1746E7EF471329564E5C099D84EAEE1992E9CAC8083E61F"),
    ("470997", "47430189", "04EA6C0F4B4F480642484E1BA18F53A0590FF5E417DAF88D4113A720C5E5E879"),
    ("470334", "47115321", "04EE012D1D5D6A093111D3251C2E9F7F086DBFDC6141BBBB3DDFF491F4B4FE01"),
    ("438968", "44358340", "050F3B5EDAD2A9D3B47469C8D6DE7959BBF872CE84FD84E8984EEE5A5D1D8FF8"),
    ("483987", "48018776", "052728FDE0451684137B1B609EFF3B32B2E963DC3E4FF84A73497AE575A2020C"),
    ("483244", "48458252", "05296249B6D56EB1D3F0B7236BCEB14C718CA728C93C879445AE8E20CDB21B85"),
    ("482484", "47670100", "053115C437D2DE1FF0C9CF24CF050EE81663DE4DC91105FF181A41E94666962D"),
    ("467254", "47331793", "056E8543DB857ABD381FE793310E7236C809588DE272C15F6D4A6347D8C4F6C7"),
    ("469267", "47332705", "05E556D01A881C2C8DC6C1E1B0A4C2AF6244A2E13789419C21C65B5A3DA377CC"),
    ("469638", "47430805", "06348D9A2C0A967C2B29F6EAF17EA70DD1A2210C619DBA63487AD39965FAAEE4"),
    ("482980", "47670270", "0635E861D8F6853C5C0A568D51981E961A440206BCDE2D5A9E57268D5FB38AC3"),
    ("464762", "44358710", "064B0C5900882A35980F42F3F75E57091BC59FECBF7AB9603DB91EC6107F2786"),
    ("464803", "44796311", "0650EB6C5139172869C37EDA0D9A07EED3AA3B057AD359EBF84D504902FB3570"),
    ("469256", "47332778", "0695F48C11148D85222E550E0DF8C5FCD1BB1A1846962A501DA546E923D6451F"),
    ("434700", "42898560", "070C9DD839435C4A5CE3B5A488BB504A56C8AC5C822F782EB459550E245B6D6E"),
    ("481508", "47669880", "07A9C1CE02512606E78A279C5BEB9EDBF7550CDC00B916449AE5A99B15206AB0"),
    ("467179", "47333264", "07C23B197DD9FCF34D9254F5FE06F1483423288DE1814EA73E173E82EECE15F9"),
    ("423452", "42897434", "07F15258504FA6BA2098C605AF27B41F23D8129A60CBAAB2492DCE52F718EA7B"),
    ("471098", "47204316", "07F61102B9FF6F9608FE620CD653F00019304F6321BB590D8F90B65401A22825"),
    ("482135", "47895659", "08040D1498453C56B3CF4D48F96303FD2804CA4638A5800FD2A40118F0C58ED1"),
    ("421063", "42444513", "081E0D42EF9DAE6A62CC8039C93D8A5243AE6A6DD9A22D2EB80652A83B7C6A59"),
    ("470546", "47331355", "085FB8F8A3736C0E67212BE2B913B5895D0EF38DBC5D5CEE6F1243B8E2265F68"),
    ("469419", "45663317", "0884C0236A111762F8DE2B8DAC96C051DBE39EEFA27511A5649B156CC1DF1257"),
    ("470318", "47331040", "08D482ED99C7711FD8FB7D575D3302681B72F7CF69DE4D52444D5E59831CEBCD"),
    ("469822", "47115198", "092E7AA4E14707909B2D6E1B00E266D1660A6A640D6DD9BDA0580D71312DC364"),
)
ASSET_TEMPLATES = r10_pool.ASSET_TEMPLATES


class FreshPoolError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise FreshPoolError(code)


def _rank(visit_id: str, video_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{ROLE}:{visit_id}:{video_id}".encode("ascii")).hexdigest().upper()


def _snapshot_excluded(root: Path, known: set[str]) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "grep", "-h", "-o", "-P", r"(?<![0-9])[0-9]{6,8}(?![0-9])", EXCLUSION_COMMIT, "--", *metadata.PATHSPECS],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {value for value in result.stdout.splitlines() if value in known}


def _r10_consumed_identities() -> set[str]:
    return {value for visit_id, video_id, _rank_sha in r10_pool.EXPECTED_POOL for value in (visit_id, video_id)}


def build_pool(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    for relative, (size, digest) in metadata.METADATA_BINDINGS.items():
        path = root / relative
        require(path.is_file() and path.stat().st_size == size and metadata._sha256(path) == digest, f"R11_METADATA_BINDING:{relative}")
    upsampling = metadata._rows(root / metadata.UPSAMPLING_METADATA)
    raw = metadata._rows(root / metadata.RAW_METADATA)
    threedod = metadata._rows(root / metadata.THREEDOD_METADATA)
    known = {value for row in upsampling for value in (row["visit_id"], row["video_id"]) if value != "NA"}
    snapshot = _snapshot_excluded(root, known)
    require(len(snapshot) == EXPECTED_SNAPSHOT_EXCLUDED_COUNT, "R11_EXCLUSION_SNAPSHOT_DRIFT")
    r10_identities = _r10_consumed_identities()
    excluded = sorted(snapshot | r10_identities)
    exclusion_sha = hashlib.sha256(("\n".join(excluded) + "\n").encode("utf-8")).hexdigest().upper()
    require(
        len(excluded) == EXPECTED_UNION_EXCLUDED_COUNT and exclusion_sha == EXPECTED_UNION_EXCLUDED_SHA256,
        "R11_EXCLUSION_UNION_DRIFT",
    )
    excluded_set = set(excluded)
    raw_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in raw}
    threedod_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in threedod}
    eligible = [
        {
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "official_fold": "Training",
            "pool_rank_sha256": _rank(row["visit_id"], row["video_id"]),
        }
        for row in upsampling
        if row["fold"] == "Training"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded_set
        and row["video_id"] not in excluded_set
        and (row["visit_id"], row["video_id"], "Training") in raw_keys
        and (row["visit_id"], row["video_id"], "Training") in threedod_keys
    ]
    require(len(eligible) == EXPECTED_ELIGIBLE_COUNT, "R11_ELIGIBLE_COUNT_DRIFT")
    pool = []
    visits: set[str] = set()
    for row in sorted(eligible, key=lambda item: item["pool_rank_sha256"]):
        if row["visit_id"] in visits:
            continue
        pool.append(row)
        visits.add(row["visit_id"])
        if len(pool) == PARENT_COUNT:
            break
    observed = tuple((row["visit_id"], row["video_id"], row["pool_rank_sha256"]) for row in pool)
    require(observed == EXPECTED_POOL, "R11_POOL_ROSTER_DRIFT")
    require(not ({value for row in pool for value in (row["visit_id"], row["video_id"])} & r10_identities), "R11_R10_POOL_OVERLAP")
    requests = [
        {"visit_id": row["visit_id"], "video_id": row["video_id"], "asset": asset, "url": template.format(video_id=row["video_id"])}
        for row in pool
        for asset, template in ASSET_TEMPLATES
    ]
    return validate_pool(
        {
            "schema": SCHEMA,
            "status": "METADATA_ONLY_FRESH_POOL_FROZEN_NO_EXECUTION_AUTHORITY",
            "selection_salt": SELECTION_SALT,
            "role": ROLE,
            "exclusion_snapshot_commit": EXCLUSION_COMMIT,
            "snapshot_excluded_identity_count": len(snapshot),
            "r10_consumed_pool_parent_count": len(r10_pool.EXPECTED_POOL),
            "union_excluded_identity_count": len(excluded),
            "union_excluded_identities_sha256": exclusion_sha,
            "eligible_row_count": len(eligible),
            "pool_parent_count": len(pool),
            "selected_parent_count_after_source_only_ranking": SELECTED_PARENT_COUNT,
            "pool_content_sha256": adapter.canonical_sha256(pool),
            "pool": pool,
            "request_plan": {
                "method": "HEAD",
                "response_body_bytes_allowed": 0,
                "request_count": len(requests),
                "expanded_requests_sha256": adapter.canonical_sha256(requests),
                "requests": requests,
            },
            "source_payload_read": False,
            "faro_payload_read": False,
            "model_output_read": False,
            "network_authorized": False,
            "data_use_authorized": False,
            "unknown_is_negative": False,
        },
        repo_root=root,
        recompute=False,
    )


def validate_pool(value: Mapping[str, Any], *, repo_root: Path, recompute: bool = True) -> dict[str, Any]:
    plan = json.loads(json.dumps(dict(value)))
    require(
        plan.get("schema") == SCHEMA
        and plan.get("status") == "METADATA_ONLY_FRESH_POOL_FROZEN_NO_EXECUTION_AUTHORITY"
        and plan.get("selection_salt") == SELECTION_SALT
        and plan.get("role") == ROLE,
        "R11_POOL_IDENTITY",
    )
    require(
        tuple((row.get("visit_id"), row.get("video_id"), row.get("pool_rank_sha256")) for row in plan.get("pool", [])) == EXPECTED_POOL
        and plan.get("pool_parent_count") == PARENT_COUNT
        and plan.get("selected_parent_count_after_source_only_ranking") == SELECTED_PARENT_COUNT
        and plan.get("pool_content_sha256") == adapter.canonical_sha256(plan.get("pool")),
        "R11_POOL_ROSTER_DRIFT",
    )
    request = plan.get("request_plan", {})
    require(
        request.get("method") == "HEAD"
        and request.get("response_body_bytes_allowed") == 0
        and request.get("request_count") == PARENT_COUNT * len(ASSET_TEMPLATES)
        and adapter.canonical_sha256(request.get("requests")) == request.get("expanded_requests_sha256"),
        "R11_REQUEST_PLAN_DRIFT",
    )
    require(
        plan.get("exclusion_snapshot_commit") == EXCLUSION_COMMIT
        and plan.get("snapshot_excluded_identity_count") == EXPECTED_SNAPSHOT_EXCLUDED_COUNT
        and plan.get("r10_consumed_pool_parent_count") == len(r10_pool.EXPECTED_POOL)
        and plan.get("union_excluded_identity_count") == EXPECTED_UNION_EXCLUDED_COUNT
        and plan.get("union_excluded_identities_sha256") == EXPECTED_UNION_EXCLUDED_SHA256
        and plan.get("eligible_row_count") == EXPECTED_ELIGIBLE_COUNT,
        "R11_POOL_EXCLUSION_DRIFT",
    )
    require(
        plan.get("source_payload_read") is False
        and plan.get("faro_payload_read") is False
        and plan.get("model_output_read") is False
        and plan.get("network_authorized") is False
        and plan.get("data_use_authorized") is False
        and plan.get("unknown_is_negative") is False,
        "R11_POOL_AUTHORITY",
    )
    if recompute:
        require(plan == build_pool(repo_root), "R11_POOL_RECOMPUTE_DRIFT")
    return plan


if __name__ == "__main__":
    print(json.dumps(build_pool(Path(__file__).resolve().parents[3]), sort_keys=True, separators=(",", ":")))
