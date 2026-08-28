"""Run the frozen non-OCR support-ray stack on an ordered real-view prefix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_facade_fingerprint as facade  # noqa: E402
import named_poi_multifacet_entrance as entrance  # noqa: E402
import named_poi_native_support_ray_eval as native_eval  # noqa: E402
import named_poi_target_support_eval as common  # noqa: E402
from named_poi_active_entrance_belief import (  # noqa: E402
    ActiveEntranceBelief,
    FrameCandidate,
)
from named_poi_native_support_ray import (  # noqa: E402
    RayBindingState,
    SupportRayConfig,
    bind_support_rays_to_entrances,
)
from named_poi_target_support_field import (  # noqa: E402
    SupportFieldConfig,
    SupportProposal,
    build_target_support_field,
)

ENTRANCE_ONLY_PROMPT = (
    "pedestrian entrance door. building entrance door. glass door. "
    "automatic sliding door. doorway. exit door."
)


def _inside(box: list[float] | tuple[float, ...], region: list[float]) -> bool:
    center_x = 0.5 * (float(box[0]) + float(box[2]))
    center_y = 0.5 * (float(box[1]) + float(box[3]))
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def _active_rows(
    manifest: dict[str, Any], audit: dict[str, Any]
) -> tuple[list[facade.ImageRow], dict[int, dict[str, Any]]]:
    frames_by_index = {int(row["index"]): row for row in manifest["frames"]}
    truth = {int(key): value for key, value in audit["portal_set_truth"]["frames"].items()}
    rows = []
    for index in audit["admitted_ordered_frames"]:
        source = frames_by_index[int(index)]
        path = Path(source["local_path"])
        if common._sha256(path) != source["sha256"]:
            raise ValueError(f"ACTIVE_IMAGE_HASH_MISMATCH:{index}")
        image_size = list(Image.open(path).size)
        if image_size != truth[int(index)]["local_image_size"]:
            raise ValueError(f"ACTIVE_IMAGE_SIZE_MISMATCH:{index}")
        rows.append(
            facade.ImageRow(
                f"active:international-finance-centre:{int(index):02d}",
                "international-finance-centre",
                "evaluation",
                path,
                source["sha256"],
                source["commons_file"],
            )
        )
    return rows, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=HERE / "named_poi_native_support_ray_protocol_v1.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-active-view-v1/source_manifest.json",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=HERE / "named_poi_active_view_source_audit_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-active-view-perception-v2",
    )
    args = parser.parse_args()

    protocol_path = args.base_protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    audit_path = args.source_audit.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["decision"] != "ADMIT_REVERSE_SIDE_SAME_PORTAL_ACTIVE_VIEW_PROXY":
        raise ValueError("ACTIVE_SOURCE_NOT_ADMITTED")

    if "reference_multifacet_library" in protocol["sources"]:
        reference_rows, _, _ = native_eval._load_multifacet_roles(protocol)
    else:
        reference_rows, _, _, _ = entrance._load_rows(protocol)
    query_rows, truth_by_index = _active_rows(manifest, audit)
    target_ids = list(protocol["targets"])
    references_by_target = {
        target_id: [row for row in reference_rows if row.target_id == target_id]
        for target_id in target_ids
    }

    clip_path = ROOT / protocol["models"]["clip"]["path"]
    dino_path = ROOT / protocol["models"]["dinov2"]["path"]
    grounder_path = ROOT / protocol["models"]["grounding_dino"]["path"]
    if common._sha256(clip_path / "pytorch_model.bin") != protocol["models"]["clip"]["weights_sha256"]:
        raise ValueError("CLIP_HASH_MISMATCH")
    if common._sha256(dino_path / "model.safetensors") != protocol["models"]["dinov2"]["weights_sha256"]:
        raise ValueError("DINO_HASH_MISMATCH")
    if common._sha256(grounder_path / "model.safetensors") != protocol["models"]["grounding_dino"]["weights_sha256"]:
        raise ValueError("GROUNDER_HASH_MISMATCH")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = facade.AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(query_rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    encoder_backend, models = facade._select_backend(
        clip_path,
        dino_path,
        representative,
        output_root / "encoder_backend_receipt.json",
    )
    encoded = native_eval._encode_native_images(
        reference_rows + query_rows,
        models,
        clip_processor,
        dino_processor,
        str(encoder_backend["selected_device_type"]),
        int(protocol["models"]["dinov2"]["native_patch_grid"]),
        4,
    )
    reference_patches = {
        target_id: [encoded[row.key] for row in references_by_target[target_id]]
        for target_id in target_ids
    }

    grounder_processor = AutoProcessor.from_pretrained(grounder_path, local_files_only=True)
    grounder_backend, grounder = entrance._select_grounder(
        grounder_path,
        grounder_processor,
        representative_image,
        ENTRANCE_ONLY_PROMPT,
        output_root / "grounder_backend_receipt.json",
    )
    belief = ActiveEntranceBelief()
    rows = []
    for row in query_rows:
        image = Image.open(row.path).convert("RGB")
        source_index = int(row.key.rsplit(":", 1)[1])
        _, proposal_rows = entrance._proposal_evidence(
            grounder,
            grounder_processor,
            image,
            ENTRANCE_ONLY_PROMPT,
            str(grounder_backend["selected_device_type"]),
            float(protocol["models"]["grounding_dino"]["box_threshold"]),
            float(protocol["models"]["grounding_dino"]["text_threshold"]),
        )
        proposal_rows = proposal_rows[: int(protocol["models"]["grounding_dino"]["maximum_proposals"])]
        proposals = []
        for index, proposal in enumerate(proposal_rows):
            proposal_id = f"{row.key}:proposal:{index + 1:02d}"
            proposal["proposal_id"] = proposal_id
            proposals.append(
                SupportProposal(
                    proposal_id,
                    float(proposal["bound_score"]),
                    tuple(float(value) for value in proposal["box_xyxy"]),
                )
            )
        field, field_diagnostics = build_target_support_field(
            row.target_id,
            encoded[row.key],
            reference_patches,
            int(protocol["models"]["dinov2"]["native_patch_grid"]),
            SupportFieldConfig(),
        )
        ray = bind_support_rays_to_entrances(
            field,
            proposals,
            image.width,
            image.height,
            SupportRayConfig(),
        )
        edge_by_id = {edge.proposal_id: edge for edge in ray.edges}
        active_ids = (
            list(ray.candidate_set)
            if ray.state == RayBindingState.SET_VALUED
            else ([ray.selected_proposal_id] if ray.selected_proposal_id else [])
        )
        decision = belief.update(
            [
                FrameCandidate(
                    proposal_id,
                    edge_by_id[proposal_id].box_xyxy,
                    edge_by_id[proposal_id].edge_score,
                )
                for proposal_id in active_ids
            ],
            image.width,
            image.height,
        )
        region = truth_by_index[source_index]["portal_set_box_xyxy"]
        proposal_truth = {
            proposal.proposal_id: _inside(proposal.box_xyxy, region) for proposal in proposals
        }
        rows.append(
            {
                "frame_index": source_index,
                "query": row.key,
                "image_sha256": row.sha256,
                "image_size": [image.width, image.height],
                "portal_set_truth_box_xyxy": region,
                "ray_state": ray.state.value,
                "ray_candidate_set": list(ray.candidate_set),
                "ray_selected_proposal_id": ray.selected_proposal_id,
                "ray_truth_retained": any(proposal_truth.get(value, False) for value in active_ids),
                "belief": asdict(decision),
                "proposals": proposal_rows,
                "proposal_portal_set_truth": proposal_truth,
                "ray_edges": [asdict(edge) for edge in ray.edges],
                "field_diagnostics": field_diagnostics,
            }
        )

    result = {
        "schema": "l10-named-poi-active-view-perception-result-v2",
        "source_manifest_sha256": common._sha256(manifest_path),
        "source_audit_sha256": common._sha256(audit_path),
        "base_protocol_sha256": common._sha256(protocol_path),
        "execution_backends": {"encoder": encoder_backend, "grounder": grounder_backend},
        "ocr_calls": 0,
        "proposal_prompt": ENTRANCE_ONLY_PROMPT,
        "proposal_prompt_authority": "GOAL_CONDITIONED_FIND_ENTRANCE_EXCLUDES_STAIRS_ESCALATORS_AND_GENERIC_PASSAGEWAYS",
        "development_status": "CONSUMED_MECHANISM_DIAGNOSTIC_NOT_FRESH_EVIDENCE",
        "transition_authority": "ORDERED_REVERSE_SIDE_PROXY_NOT_COMMANDED_ACTION",
        "admitted_frames": len(rows),
        "truth_retained_frames": sum(bool(row["ray_truth_retained"]) for row in rows),
        "belief_states": [row["belief"]["state"] for row in rows],
        "rows": rows,
        "claim_scope": "Two-frame reverse-side real-perception prefix only; no active-view causality, unique leaf identity, outside entrance navigation, arrival, user benefit, or safety evidence.",
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output_path),
                "truth_retained_frames": result["truth_retained_frames"],
                "belief_states": result["belief_states"],
                "execution_backends": result["execution_backends"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
