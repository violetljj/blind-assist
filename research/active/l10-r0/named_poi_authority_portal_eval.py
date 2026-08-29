"""Evaluate a proof-gated, authority-carrying functional portal set.

PB5 deliberately reuses the sealed PB4 identity result and the unchanged
dual-family portal proposer.  It tests the composition contract only: a portal
proposal may carry a requested-entity authority token when the same image has a
correct SCIL proof for that entity.  It never emits COMMIT or navigation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2

from named_poi_portal_lattice import PortalLatticeConfig, propose_portal_sets


REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source(spec: dict[str, Any]) -> Path:
    path = _repo_path(spec["path"])
    actual = _sha256(path)
    expected = spec["sha256"]
    if actual != expected:
        raise ValueError(f"SOURCE_HASH_MISMATCH:{path}:{actual}:{expected}")
    return path


def _portal_set_member(box: Sequence[float], region: Sequence[float]) -> bool:
    center_x = 0.5 * (box[0] + box[2])
    center_y = 0.5 * (box[1] + box[3])
    center_inside = (
        region[0] <= center_x <= region[2]
        and region[1] <= center_y <= region[3]
    )
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    proposal_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    coverage = intersection / proposal_area if proposal_area > 0.0 else 0.0
    return center_inside and coverage >= 0.5


def _structure_authorizable(
    proposal: dict[str, Any], image_height: int, rule: dict[str, Any]
) -> bool:
    """Apply the frozen, truth-blind paired-handle opportunity predicate."""
    bottom_fraction = float(proposal["box_xyxy"][3]) / float(image_height)
    return (
        proposal["family"] == rule["required_family"]
        and float(proposal["proposal_score"]) >= rule["minimum_proposal_score"]
        and float(proposal["normalized_height"])
        >= rule["minimum_normalized_height"]
        and bottom_fraction >= rule["minimum_box_bottom_fraction"]
    )


def _authority_token(
    *, key: str, target: str, image_sha256: str, pb4_result_sha256: str
) -> tuple[str, dict[str, str]]:
    payload = {
        "authority": "PB4_C2_SCIL_SAME_FRAME_CORRECT_PROOF",
        "frame_key": key,
        "requested_entity": target,
        "image_sha256": image_sha256,
        "pb4_result_sha256": pb4_result_sha256,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


def _query_rows(manifest: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for entity in manifest["entities"]:
        if entity["split"] != split:
            continue
        for query in entity["queries"]:
            key = f"query:{query['key']}"
            rows[key] = {
                **query,
                "target": entity["id"],
                "split": split,
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    source_paths = {
        name: _verify_source(spec) for name, spec in protocol["sources"].items()
    }
    component_paths = {
        name: _verify_source(spec)
        for name, spec in protocol["frozen_components"].items()
        if "path" in spec and "sha256" in spec
    }
    expected_evaluator = component_paths["evaluator"].resolve()
    if expected_evaluator != Path(__file__).resolve():
        raise ValueError("EVALUATOR_PATH_MISMATCH")

    manifest = _load_json(source_paths["dataset_manifest"])
    pb4 = _load_json(source_paths["pb4_formal_result"])
    audit = _load_json(source_paths["portal_opportunity_audit"])
    if pb4["decision"] != protocol["cohort_contract"]["required_pb4_decision"]:
        raise ValueError("PB4_DECISION_MISMATCH")

    scil_rows = pb4["raw_records"]["test"]["C2_SCIL"]
    proof_positive = {
        row["key"]: row
        for row in scil_rows
        if row["proof"] is not None and row["correct"] is True
    }
    identity_unknown = {
        row["key"]: row for row in scil_rows if row["proof"] is None
    }
    audit_rows = {row["key"]: row for row in audit["frames"]}
    if set(audit_rows) != set(proof_positive):
        raise ValueError("AUDIT_FRAME_SET_MUST_EQUAL_PB4_PROOF_POSITIVE_SET")
    if len(proof_positive) != protocol["cohort_contract"]["proof_positive_frames"]:
        raise ValueError("PROOF_POSITIVE_COUNT_MISMATCH")

    manifest_queries = _query_rows(manifest, "test")
    test_entities = sorted(
        entity["id"] for entity in manifest["entities"] if entity["split"] == "test"
    )
    rows = []
    for key in sorted(proof_positive):
        proof = proof_positive[key]
        audit_row = audit_rows[key]
        query = manifest_queries[key]
        if proof["target"] != query["target"] or proof["proof"] != query["target"]:
            raise ValueError(f"IDENTITY_TARGET_MISMATCH:{key}")
        if query["sha256"] != audit_row["image_sha256"]:
            raise ValueError(f"AUDIT_IMAGE_HASH_MISMATCH:{key}")
        image_path = Path(query["local_path"])
        if not image_path.exists():
            entity = query["target"]
            index = key.rsplit(":", 1)[-1]
            image_path = (
                REPO_ROOT
                / "artifacts.local"
                / "knowledge"
                / "named-poi-biscript-v1"
                / "images"
                / entity
                / f"query-{index}.jpg"
            )
        if _sha256(image_path) != audit_row["image_sha256"]:
            raise ValueError(f"LOCAL_IMAGE_HASH_MISMATCH:{key}")

        base = {
            "key": key,
            "target": proof["target"],
            "facet": proof["facet"],
            "image_sha256": audit_row["image_sha256"],
            "opportunity_status": audit_row["status"],
            "scil_proof": {
                "proof": proof["proof"],
                "top_score": proof["top_score"],
                "margin": proof["margin"],
                "top_carrier": proof["top_carrier"],
                "strong_script_count": proof["strong_script_count"],
            },
        }
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"IMAGE_DECODE_FAILED:{key}")
        actual_size = [int(image.shape[1]), int(image.shape[0])]
        if actual_size != audit_row["local_image_size"]:
            raise ValueError(f"IMAGE_SIZE_MISMATCH:{key}:{actual_size}")
        proposals, diagnostics = propose_portal_sets(image)
        truth = audit_row.get("portal_set_box_xyxy")
        opportunity_rule = protocol["structural_opportunity_gate"]
        annotated = [
            {
                **asdict(proposal),
                "truth_member": _portal_set_member(proposal.box_xyxy, truth)
                if truth is not None
                else None,
                "structure_authorizable": _structure_authorizable(
                    asdict(proposal), actual_size[1], opportunity_rule
                ),
            }
            for proposal in proposals[:10]
        ]
        retained_top1 = bool(truth) and bool(annotated) and bool(
            annotated[0]["truth_member"]
        )
        retained_top3 = bool(truth) and any(
            bool(row["truth_member"]) for row in annotated[:3]
        )
        structurally_authorizable = [
            row for row in annotated[:3] if row["structure_authorizable"]
        ]
        authorized_truth_retained = bool(truth) and any(
            bool(row["truth_member"]) for row in structurally_authorizable
        )
        token, token_payload = _authority_token(
            key=key,
            target=proof["target"],
            image_sha256=audit_row["image_sha256"],
            pb4_result_sha256=protocol["sources"]["pb4_formal_result"]["sha256"],
        )
        authorized = [
            {
                "proposal_id": proposal["proposal_id"],
                "requested_entity": proof["target"],
                "authority_token": token,
            }
            for proposal in structurally_authorizable
        ]
        wrong_requests = [entity for entity in test_entities if entity != proof["target"]]
        status = audit_row["status"]
        if status == "IN_SCOPE_PORTAL_PRESENT":
            if authorized_truth_retained:
                state = "ENTITY_BOUND_PORTAL_SET"
                deficit_action = None
            elif authorized:
                state = "PORTAL_AUTHORIZED_TRUTH_MISS"
                deficit_action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
            else:
                state = "PORTAL_OPPORTUNITY_REJECTED"
                deficit_action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "NEGATIVE_NO_PORTAL":
            state = (
                "FALSE_PORTAL_AUTHORIZATION"
                if authorized
                else "IDENTITY_PROVED_PORTAL_NOT_OBSERVABLE"
            )
            deficit_action = audit_row["deficit_action"]
        elif status == "OUT_OF_SCOPE_ARCHITECTURAL_ENTRANCE":
            state = (
                "OOD_PORTAL_AUTHORIZATION_LEAKAGE"
                if authorized
                else "IDENTITY_PROVED_PORTAL_REPRESENTATION_OOD"
            )
            deficit_action = audit_row["deficit_action"]
        else:
            raise ValueError(f"UNKNOWN_AUDIT_STATUS:{key}:{status}")
        rows.append(
            {
                **base,
                "state": state,
                "deficit_action": deficit_action,
                "proposer_executed": True,
                "truth_box_xyxy": truth,
                "truth_kind": audit_row.get("kind"),
                "truth_retained_top1": retained_top1,
                "truth_retained_top3": retained_top3,
                "authorized_truth_retained_top3": authorized_truth_retained,
                "diagnostics": diagnostics,
                "proposals": annotated,
                "authority_token_payload": token_payload if authorized else None,
                "authorized_proposals": authorized,
                "source_label_wrong_requests": wrong_requests,
                "unbound_wrong_request_authorizations": len(wrong_requests)
                if authorized
                else 0,
                "joined_wrong_request_authorizations": 0,
            }
        )

    unknown_rows = [
        {
            "key": key,
            "target": row["target"],
            "facet": row["facet"],
            "state": "IDENTITY_UNKNOWN",
            "deficit_action": "APPROACH_TEXT_OR_SWEEP_SIGN",
            "proposer_executed": False,
            "authorized_proposals": [],
        }
        for key, row in sorted(identity_unknown.items())
    ]
    primary_positive = [
        row for row in rows if row["opportunity_status"] == "IN_SCOPE_PORTAL_PRESENT"
    ]
    negative_controls = [
        row for row in rows if row["opportunity_status"] == "NEGATIVE_NO_PORTAL"
    ]
    ood_challenges = [
        row
        for row in rows
        if row["opportunity_status"] == "OUT_OF_SCOPE_ARCHITECTURAL_ENTRANCE"
    ]
    retained_top1 = sum(row["truth_retained_top1"] for row in primary_positive)
    retained_top3 = sum(row["truth_retained_top3"] for row in primary_positive)
    authorized_truth = sum(
        row["authorized_truth_retained_top3"] for row in primary_positive
    )
    unbound_wrong = sum(
        row["unbound_wrong_request_authorizations"] for row in rows
    )
    joined_wrong = sum(row["joined_wrong_request_authorizations"] for row in rows)
    wrong_reduction = (
        (unbound_wrong - joined_wrong) / unbound_wrong if unbound_wrong else 0.0
    )
    forbidden_states = {
        "COMMIT",
        "HANDOFF",
        "ARRIVED",
        "GUIDE_LEFT",
        "GUIDE_RIGHT",
        "GUIDE_FORWARD",
    }
    emitted_states = {row["state"] for row in rows + unknown_rows}
    thresholds = protocol["promotion_gate"]
    gates = {
        "frozen_primary_positive_count": len(primary_positive)
        == thresholds["primary_positive_count"],
        "authorized_top3_truth_retention": authorized_truth
        >= thresholds["minimum_authorized_top3_truth_retained"],
        "zero_negative_control_authorization": all(
            not row["authorized_proposals"] for row in negative_controls
        ),
        "zero_out_of_scope_authorization": all(
            not row["authorized_proposals"] for row in ood_challenges
        ),
        "correct_identity_authority": all(
            all(
                proposal["requested_entity"] == row["target"]
                for proposal in row["authorized_proposals"]
            )
            for row in rows
        ),
        "zero_wrong_request_authorization": joined_wrong == 0,
        "zero_identity_unknown_leakage": all(
            not row["authorized_proposals"] for row in unknown_rows
        ),
        "no_commit_or_navigation_state": forbidden_states.isdisjoint(emitted_states),
    }
    gate_met = all(gates.values())
    decision = (
        "L10_PB5_SCRIPT_PROVED_AUTHORITY_CARRYING_PORTAL_LATTICE_GATE_MET"
        if gate_met
        else "L10_PB5_SCRIPT_PROVED_AUTHORITY_CARRYING_PORTAL_LATTICE_GATE_NOT_MET"
    )
    metrics = {
        "pb4_proof_positive_frames": len(proof_positive),
        "in_scope_portal_positive_frames": len(primary_positive),
        "negative_no_portal_frames": len(negative_controls),
        "out_of_scope_architectural_entrance_frames": len(ood_challenges),
        "raw_truth_retained_top1": retained_top1,
        "raw_truth_retained_top3": retained_top3,
        "raw_truth_retained_top1_rate": retained_top1 / len(primary_positive),
        "raw_truth_retained_top3_rate": retained_top3 / len(primary_positive),
        "identity_bound_truth_retained_top3": authorized_truth,
        "identity_bound_truth_retained_top3_rate": authorized_truth
        / len(primary_positive),
        "negative_control_false_authorization_frames": sum(
            bool(row["authorized_proposals"]) for row in negative_controls
        ),
        "out_of_scope_authorization_leakage_frames": sum(
            bool(row["authorized_proposals"]) for row in ood_challenges
        ),
        "source_label_wrong_request_pairs": len(rows) * (len(test_entities) - 1),
        "unbound_wrong_request_authorizations": unbound_wrong,
        "joined_wrong_request_authorizations": joined_wrong,
        "wrong_request_authorization_reduction": wrong_reduction,
        "identity_unknown_frames": len(unknown_rows),
        "identity_unknown_portal_ownership_leakage": sum(
            bool(row["authorized_proposals"]) for row in unknown_rows
        ),
        "new_ocr_calls": 0,
        "new_embedding_calls": 0,
    }
    result = {
        "schema": "l10-named-poi-authority-carrying-portal-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "selection_authority": "SEALED_PB4_TEST_C2_SCIL_CORRECT_PROOFS_ONLY",
        "portal_configuration": asdict(PortalLatticeConfig()),
        "metrics": metrics,
        "gate": gates,
        "decision": decision,
        "rows": rows,
        "identity_unknown_rows": unknown_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "decision": decision, "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
