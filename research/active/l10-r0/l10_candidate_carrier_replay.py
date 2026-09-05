"""V2 typed adapter replay of exported, consumed SEVN observations.

No pixels, OCR, models, action choices or candidate scores are recomputed. The
legacy trace exports only its selected portal box. Keep existing local topology
and crop witnesses at their original surrogate scope; do not impose a new
containment veto or confuse private-crop coordinates with full-frame pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from l10_candidate_carrier_association import AssociationState, associate_frame


SCHEMA = "blindassist-l10-sevn-progressive-episode-result-v1"


def adapt_observation(observation: dict[str, Any], source_id: str) -> dict[str, Any]:
    """Read runtime output and public render metadata only; never evaluator truth."""
    output = observation["runtime_output"]
    receipt = observation["render_receipt"]
    height, width = receipt["viewport_shape_hwc"][:2]
    binding = output.get("selected_binding")

    def normalized(box: list[float]) -> list[float]:
        return [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height]

    context = {
        "source_id": source_id,
        "frame_id": receipt["viewport_pixel_sha256"],
        "roi_id": "full-viewport",
        "candidate_roster_complete": output["portal_candidate_count"] == 1 and binding is not None,
        "candidate_evidence": [], "text_evidence": [], "carrier_relations": [],
    }
    if binding:
        context["candidate_evidence"].append({
            "candidate_id": binding["portal_id"], "bbox_xyxy_norm": normalized(binding["portal_box_xyxy"]),
            "score": binding["portal_confidence"],
        })
    edge = binding.get("authority_edge") if binding else None
    private_medium = edge is not None and edge.get("type") == "PORTAL_TO_PRIVATE_MEDIUM_OBSERVATION_TO_EXACT_TOKEN"
    text_roi = f"private-medium-crop:{binding['portal_id']}" if private_medium else context["roi_id"]
    for index, text in enumerate(output.get("exact_target_text_candidates", ())):
        row = {"text_id": f"text-{index}", "text": text["canonical"], "role": "UNKNOWN", "roi_id": text_roi}
        # portal_private_rows leaves box_xyxy in the resized crop. The concise
        # exporter dropped its transform, so do not invent full-frame geometry.
        if private_medium:
            row["bbox_xyxy_px"] = list(text["box_xyxy"])
        else:
            row["bbox_xyxy_norm"] = normalized(text["box_xyxy"])
        context["text_evidence"].append(row)
    producer = None
    if edge and edge.get("source_portal_id") == binding["portal_id"] and edge.get("unique_portal_witness_count") == 1:
        if edge.get("type") in (
            "PORTAL_TO_LOCAL_OBSERVATION_TO_EXACT_TOKEN",
            "PORTAL_TO_PRIVATE_MEDIUM_OBSERVATION_TO_EXACT_TOKEN",
        ):
            producer = ("LOCAL_OBSERVATION_OF", f"recorded-authority-edge:{edge['type']}")
    elif binding and output.get("state") == "PIXEL_BOUND_MASK_PORTAL_PROPOSAL" and output.get("observation_branch") == "V2_RESULT_RETAINED":
        # infer_binding emits this state only after pair_topology is admissible.
        # That topology includes a credential halo; text need not lie inside the
        # door extent. The concise trace retained its result, not the full audit.
        producer = ("LOCAL_TOPOLOGY_SUPPORTS", "recorded-selected-mask-topology")
    if producer:
        # selected_binding records the chosen text but not its unique OCR row ID.
        # Retain that producer witness without guessing which equal-text box won.
        context["text_evidence"].append({"text_id": "selected-producer-text", "text": binding["text"], "role": "UNKNOWN", "roi_id": text_roi})
        context["carrier_relations"].append({
            **{key: context[key] for key in ("source_id", "frame_id")},
            "text_roi_id": text_roi, "candidate_roi_id": context["roi_id"],
            "text_id": "selected-producer-text", "candidate_id": binding["portal_id"],
            "predicate": producer[0], "status": "SUPPORTED", "evidence_id": producer[1],
        })
    context["preserved_local_producer"] = producer[1] if producer else None
    return context


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = Counter(row[key] for row in rows)
    return {"correct": values["CORRECT_BINDING"], "wrong": values["WRONG_BINDING"], "unknown": values["UNKNOWN"]}


def replay(input_path: Path, output_path: Path) -> dict[str, Any]:
    raw = input_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or payload.get("claim_scope") != "CONSUMED_SEVN_ADDRESS_DOOR_PIXEL_EPISODE_DEVELOPMENT_ONLY":
        raise ValueError("Only the exported consumed SEVN Development result is admitted")
    digest = hashlib.sha256(raw).hexdigest()
    rows = []
    for episode in payload["episode_results"]:
        for arm_name, arm in episode["arms"].items():
            row = {"episode_id": episode["episode_id"], "arm": arm_name, "before": arm["outcome"],
                   "local_surrogate_after": "UNKNOWN", "named_identity_after": "UNKNOWN", "association": "NO_SELECTED_CANDIDATE"}
            if arm["selected_observation_id"] is not None:
                available = [o for o in episode["observation_trace"] if
                             o["observation_id"] == arm["selected_observation_id"] and
                             o["runtime_output"]["binding_confidence"] is not None]
                selected = sorted(available, key=lambda o: (-o["runtime_output"]["binding_confidence"], o["frame_id"], o["heading_degrees"]))[0]
                if selected["runtime_output"]["binding_confidence"] != arm["binding_confidence"]:
                    raise ValueError("Exported selected view does not reproduce the frozen arm selection")
                public = adapt_observation(selected, f"SEVN-consumed-result:{digest}")
                association = associate_frame(public).candidates[0]
                row.update(association=association.state.value, reasons=association.reasons,
                           local_text_ids=association.local_text_ids, identity_text_ids=association.identity_text_ids,
                           complete_exported_roster=public["candidate_roster_complete"],
                           preserved_local_producer=public["preserved_local_producer"])
                if association.state in (AssociationState.LOCAL_SUPPORT, AssociationState.IDENTITY_SUPPORT):
                    row["local_surrogate_after"] = arm["outcome"]
                if association.state == AssociationState.IDENTITY_SUPPORT:
                    row["named_identity_after"] = arm["outcome"]
            rows.append(row)
    metrics = {}
    for arm in payload["metrics"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        before = counts(arm_rows, "before")
        after = counts(arm_rows, "local_surrogate_after")
        metrics[arm] = {"before_local_surrogate": before, "after_local_support_filter": after,
                        "correct_retention": after["correct"] / before["correct"] if before["correct"] else None,
                        "named_identity": counts(arm_rows, "named_identity_after"),
                        "extra_observations_unchanged": payload["metrics"][arm]["extra_observation_count"]}
    result = {
        "schema": "blindassist-l10-candidate-carrier-consumed-replay-v2", "adapter_version": 2,
        "input_path": str(input_path.resolve()),
        "input_sha256": digest, "authority": "EXPORTED_CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "intervention": "Preserve existing local producer witnesses with explicit coordinate domains; no candidate veto, reselection or new observation",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics, "rows": rows, "model_calls": 0, "pixel_reads": 0,
        "limitations": [
            "Only selected portal boxes were exported; omitted competitors cannot establish complete-roster association.",
            "Private-medium OCR boxes are crop-local and their transform is missing; V2 retains only the recorded local producer relation.",
            "Original mask topology allows text in an expanded credential halo outside the door extent; containment is not its contract.",
            "Private crop edges support local address-door surrogates, not independently established named identity or ownership.",
            "No current identity-sign role, identity relation, or track continuity receipt exists in this legacy export.",
            "Geometry or missing relation alone cannot establish exact identity or target absence.",
            "Local-surrogate parity is adapter compatibility, not an empirical accuracy improvement.",
        ],
    }
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != digest:
        raise ValueError("Consumed input changed during replay")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input, args.output)
    print(json.dumps({"authority": result["authority"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
