"""Frozen NearID-style hard-negative unary experiment over source-disjoint CORe50 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as dino,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    visible_identity_probe as base,
)


PROTOCOL_ID = "NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0"
SCHEMA_VERSION = "blindassist_near_identity_hard_negative_unary_v0"
SEED = 20260824
ARCHIVE_URL = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
ARCHIVE_EXPECTED_BYTES = 5_892_103_007
MEMBER_RE = re.compile(r"(?:^|/)s(?P<session>\d+)/o(?P<object>\d+)/C_\d+_\d+_(?P<frame>\d+)\.png$", re.I)
SPLITS = {
    "train": {"categories": [1, 2, 3, 4], "objects": list(range(1, 21)), "sessions": [1, 2, 3, 4]},
    "calibration": {"categories": [5, 6, 7], "objects": list(range(21, 36)), "sessions": [5, 6, 7]},
    "test": {"categories": [8, 9, 10], "objects": list(range(36, 51)), "sessions": [8, 9, 10, 11]},
}
QUANTILES = {
    "train": [0.25, 0.50, 0.75],
    "calibration_candidate": [0.33, 0.67],
    "test_candidate": [0.25, 0.50, 0.75],
}
EXPECTED_COUNTS = {"train_tuples": 180, "calibration_pairs": 60, "test_pairs": 135}
HEAD_SPEC = {"input_dim": 384, "hidden_dim": 256, "output_dim": 128, "steps": 1200, "batch_size": 32}
OPTIMIZER = {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4}
LOSS = {"temperature": 0.07, "rank_weight": 0.5}
FALSE_ACCEPT_LIMIT = 0.05
CONTROL_RETENTION_GATE = 0.80
COVERAGE_GATE = 0.50
SOURCE_SESSION_RECALL_GATE = 0.40
SOURCE_SESSION_GAP_GATE = 0.35
CLAIM_CEILING = (
    "DEVELOPMENT_NEAR_IDENTITY_OBJECT_SIGNAL_ONLY_NO_PHYSICAL_INSTANCE_AUTHORITY_LOCALIZATION_"
    "PROPOSAL_NONE_HEAD_BELIEF_TRACKER_P1_CONTROL_SAFETY_OR_PRODUCT_CLAIM"
)


class NearIdentityExperimentError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NearIdentityExperimentError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_hash(value: Mapping[str, Any]) -> str:
    return base._body_hash(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_body(value: Mapping[str, Any], label: str) -> None:
    _require(value.get("body_sha256") == _body_hash(value), f"{label} body SHA mismatch")


def protocol_payload(protocol_doc: Path) -> dict[str, Any]:
    _require(protocol_doc.is_file(), f"missing protocol document: {protocol_doc}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": _utc_now(),
        "protocol_document": {"path": str(protocol_doc.resolve()), "sha256": _sha256_file(protocol_doc)},
        "dataset": {
            "name": "CORe50_128x128",
            "archive_url": ARCHIVE_URL,
            "expected_bytes": ARCHIVE_EXPECTED_BYTES,
            "license": "CC-BY-4.0",
            "splits": SPLITS,
            "quantiles": QUANTILES,
            "frame_rule": "sort_member_name_then_floor(q*(n-1))",
            "matched_context": "target_hard_and_ordinary_share_candidate_session_and_quantile",
        },
        "arms": {
            "baseline": {
                "name": "FROZEN_DINOV2_S_SYMMETRIC_LOCAL_PATCH",
                "repository": dino.MODEL_REPOSITORY,
                "revision": dino.MODEL_REVISION,
                "model_files": dino.MODEL_FILES,
            },
            "challenger": {
                "name": "DINO_MEAN_POOL_NEAR_IDENTITY_PROJECTION_HEAD",
                "head": HEAD_SPEC,
                "optimizer": OPTIMIZER,
                "loss": LOSS,
                "seed": SEED,
                "checkpoint_selection": "FINAL_STEP_ONLY",
            },
        },
        "calibration": {
            "false_accept_limit": FALSE_ACCEPT_LIMIT,
            "threshold_rule": "nextafter_descending_absence_max_at_index_floor(limit*N)_toward_positive_infinity",
            "quality_rule": "calibration_target_laplacian_variance_quantiles_0p33_0p67",
        },
        "gates": {
            "rescue_gt_collateral": True,
            "control_retention_min": CONTROL_RETENTION_GATE,
            "candidate_permutation_invariance": 1.0,
            "target_absent_false_accept_max": FALSE_ACCEPT_LIMIT,
            "ordinary_negative_false_accept_max": FALSE_ACCEPT_LIMIT,
            "accepted_coverage_min": COVERAGE_GATE,
            "matched_context_required": True,
            "source_session_min_same_instance_recall": SOURCE_SESSION_RECALL_GATE,
            "source_session_max_recall_gap": SOURCE_SESSION_GAP_GATE,
        },
        "expected_counts": EXPECTED_COUNTS,
        "forbidden": [
            "PDM", "FUSION", "LAYOUT_VERIFIER", "REBASE", "DEEP_SETS", "MULTIPLE_REFERENCES",
            "THRESHOLD_SWEEP", "ACTIVE_SEARCH", "BELIEF", "TRACKER", "P1", "DEFAULT_APP",
        ],
        "recovery": {
            "before_pretest_lock": "restart_unlocked_run_from_step_zero_only",
            "after_pretest_lock": "resume_deterministic_test_under_identical_hashes_only",
            "after_final_report": "refuse_overwrite_or_rerun",
            "external_model_calls": 0,
        },
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "NEAR_IDENTITY_PROTOCOL_FROZEN_NO_OUTCOME",
    }
    payload["body_sha256"] = _body_hash(payload)
    return payload


def freeze_protocol(protocol_doc: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite protocol freeze: {output}")
    payload = protocol_payload(protocol_doc)
    _atomic_json(output, payload)
    return payload


def _category_for_object(object_id: int) -> int:
    _require(1 <= object_id <= 50, f"invalid CORe50 object id: {object_id}")
    return (object_id - 1) // 5 + 1


def _next_same_category(object_id: int) -> int:
    start = (_category_for_object(object_id) - 1) * 5 + 1
    return start + ((object_id - start + 1) % 5)


def _ordinary_object(object_id: int, split_name: str) -> int:
    categories = SPLITS[split_name]["categories"]
    category = _category_for_object(object_id)
    position = (object_id - 1) % 5
    next_category = categories[(categories.index(category) + 1) % len(categories)]
    return (next_category - 1) * 5 + position + 1


def _sample_id(split_name: str, session: int, object_id: int, quantile: float) -> str:
    return f"{split_name}-s{session:02d}-o{object_id:02d}-q{int(round(quantile * 100)):03d}"


def _slot_for_pair(pair_id: str) -> str:
    return "A" if hashlib.sha256(pair_id.encode("utf-8")).digest()[0] % 2 == 0 else "B"


def _index_members(archive: Path) -> dict[tuple[int, int], list[str]]:
    _require(archive.is_file(), f"missing archive: {archive}")
    _require(archive.stat().st_size == ARCHIVE_EXPECTED_BYTES, "CORe50 archive byte count drifted")
    indexed: dict[tuple[int, int], list[str]] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            match = MEMBER_RE.search(info.filename.replace("\\", "/"))
            if match is None:
                continue
            key = (int(match.group("session")), int(match.group("object")))
            indexed.setdefault(key, []).append(info.filename)
    for members in indexed.values():
        members.sort()
        _require(len(members) == len(set(members)), "duplicate CORe50 member name")
    return indexed


def _choose_member(indexed: Mapping[tuple[int, int], Sequence[str]], session: int, object_id: int, q: float) -> str:
    members = indexed.get((session, object_id), ())
    _require(members, f"missing frames for s{session}/o{object_id}")
    index = math.floor(q * (len(members) - 1))
    return str(members[index])


def _build_samples(indexed: Mapping[tuple[int, int], Sequence[str]]) -> list[dict[str, Any]]:
    required: set[tuple[str, int, int, float]] = set()
    for object_id in SPLITS["train"]["objects"]:
        for q in QUANTILES["train"]:
            for session in SPLITS["train"]["sessions"]:
                required.add(("train", session, object_id, q))
    for object_id in SPLITS["calibration"]["objects"]:
        required.add(("calibration", 5, object_id, 0.50))
        for session in (6, 7):
            for q in QUANTILES["calibration_candidate"]:
                required.add(("calibration", session, object_id, q))
    for object_id in SPLITS["test"]["objects"]:
        required.add(("test", 8, object_id, 0.50))
        for session in (9, 10, 11):
            for q in QUANTILES["test_candidate"]:
                required.add(("test", session, object_id, q))
    samples = []
    for split_name, session, object_id, q in sorted(required):
        samples.append(
            {
                "sample_id": _sample_id(split_name, session, object_id, q),
                "split": split_name,
                "category_id": _category_for_object(object_id),
                "physical_object_id": object_id,
                "session_id": session,
                "quantile": q,
                "archive_member": _choose_member(indexed, session, object_id, q),
            }
        )
    return samples


def _sample_lookup(samples: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int, float], str]:
    return {
        (str(row["split"]), int(row["session_id"]), int(row["physical_object_id"]), float(row["quantile"])):
        str(row["sample_id"])
        for row in samples
    }


def _build_train_tuples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _sample_lookup(samples)
    rows = []
    for object_id in SPLITS["train"]["objects"]:
        for positive_session in (2, 3, 4):
            for q in QUANTILES["train"]:
                rows.append(
                    {
                        "tuple_id": f"train-o{object_id:02d}-s{positive_session:02d}-q{int(q*100):03d}",
                        "anchor": lookup[("train", 1, object_id, q)],
                        "positive": lookup[("train", positive_session, object_id, q)],
                        "hard": lookup[("train", positive_session, _next_same_category(object_id), q)],
                        "ordinary": lookup[("train", positive_session, _ordinary_object(object_id, "train"), q)],
                    }
                )
    return rows


def _build_pairs(samples: Sequence[Mapping[str, Any]], split_name: str) -> list[dict[str, Any]]:
    _require(split_name in {"calibration", "test"}, "pairs exist only for calibration or test")
    lookup = _sample_lookup(samples)
    ref_session = 5 if split_name == "calibration" else 8
    candidate_sessions = (6, 7) if split_name == "calibration" else (9, 10, 11)
    quantiles = QUANTILES[f"{split_name}_candidate"]
    rows = []
    for object_id in SPLITS[split_name]["objects"]:
        for session in candidate_sessions:
            for q in quantiles:
                pair_id = f"{split_name}-o{object_id:02d}-s{session:02d}-q{int(q*100):03d}"
                target_slot = _slot_for_pair(pair_id)
                hard_slot = "B" if target_slot == "A" else "A"
                absence_hard_slot = _slot_for_pair(pair_id + "-absence")
                ordinary_slot = "B" if absence_hard_slot == "A" else "A"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "split": split_name,
                        "category_id": _category_for_object(object_id),
                        "reference_object_id": object_id,
                        "candidate_session_id": session,
                        "candidate_quantile": q,
                        "reference": lookup[(split_name, ref_session, object_id, 0.50)],
                        "present_slots": {
                            target_slot: lookup[(split_name, session, object_id, q)],
                            hard_slot: lookup[(split_name, session, _next_same_category(object_id), q)],
                        },
                        "target_slot": target_slot,
                        "hard_slot": hard_slot,
                        "absence_slots": {
                            absence_hard_slot: lookup[(split_name, session, _next_same_category(object_id), q)],
                            ordinary_slot: lookup[(split_name, session, _ordinary_object(object_id, split_name), q)],
                        },
                        "absence_hard_slot": absence_hard_slot,
                        "ordinary_slot": ordinary_slot,
                    }
                )
    return rows


def validate_roster(roster: Mapping[str, Any]) -> None:
    _verify_body(roster, "roster")
    samples = roster["samples"]
    sample_ids = [row["sample_id"] for row in samples]
    _require(len(sample_ids) == len(set(sample_ids)), "duplicate sample id")
    for split_name, spec in SPLITS.items():
        rows = [row for row in samples if row["split"] == split_name]
        _require({row["physical_object_id"] for row in rows} == set(spec["objects"]), f"{split_name} object drift")
        _require({row["category_id"] for row in rows} == set(spec["categories"]), f"{split_name} category drift")
        _require({row["session_id"] for row in rows} == set(spec["sessions"]), f"{split_name} session drift")
    for field in ("physical_object_id", "category_id", "session_id"):
        sets = [{row[field] for row in samples if row["split"] == name} for name in SPLITS]
        _require(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3)), f"{field} leakage")
    _require(len(roster["train_tuples"]) == EXPECTED_COUNTS["train_tuples"], "train tuple count drift")
    _require(len(roster["calibration_pairs"]) == EXPECTED_COUNTS["calibration_pairs"], "calibration count drift")
    _require(len(roster["test_pairs"]) == EXPECTED_COUNTS["test_pairs"], "test count drift")
    sample_by_id = {row["sample_id"]: row for row in samples}
    for pair in [*roster["calibration_pairs"], *roster["test_pairs"]]:
        candidates = [sample_by_id[value] for value in [*pair["present_slots"].values(), *pair["absence_slots"].values()]]
        _require(all(row["session_id"] == pair["candidate_session_id"] for row in candidates), "unmatched session")
        _require(all(row["quantile"] == pair["candidate_quantile"] for row in candidates), "unmatched quantile")


def freeze_roster(protocol_path: Path, archive: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite roster: {output}")
    protocol = _load_json(protocol_path)
    _verify_body(protocol, "protocol")
    _require(protocol["protocol_id"] == PROTOCOL_ID, "protocol id drifted")
    indexed = _index_members(archive)
    samples = _build_samples(indexed)
    roster = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": _utc_now(),
        "protocol_body_sha256": protocol["body_sha256"],
        "archive": {"path": str(archive.resolve()), "bytes": archive.stat().st_size, "sha256": _sha256_file(archive)},
        "selection_read": "ZIP_CENTRAL_DIRECTORY_ONLY_NO_PIXEL_DECODE",
        "samples": samples,
        "train_tuples": _build_train_tuples(samples),
        "calibration_pairs": _build_pairs(samples, "calibration"),
        "test_pairs": _build_pairs(samples, "test"),
        "terminal": "CORe50_ROSTER_FROZEN_NO_MODEL_OUTCOME",
    }
    roster["body_sha256"] = _body_hash(roster)
    validate_roster(roster)
    _atomic_json(output, roster)
    return roster


def select_absence_threshold(scores: Sequence[float], limit: float = FALSE_ACCEPT_LIMIT) -> float:
    _require(scores, "cannot calibrate threshold on zero absence scores")
    _require(0.0 <= limit < 1.0, "false accept limit out of range")
    descending = sorted((float(value) for value in scores), reverse=True)
    allowed = math.floor(limit * len(descending))
    return float(np.nextafter(descending[allowed], math.inf))


def _materialize(archive: Path, roster: Mapping[str, Any], run_dir: Path, splits: set[str]) -> dict[str, Path]:
    selected = [row for row in roster["samples"] if row["split"] in splits]
    paths: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as bundle:
        for row in selected:
            destination = run_dir / "materialized" / row["split"] / f"{row['sample_id']}.png"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".png.tmp")
                with bundle.open(row["archive_member"]) as source, temporary.open("wb") as sink:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        sink.write(block)
                os.replace(temporary, destination)
            paths[row["sample_id"]] = destination
    _require(len(paths) == len(selected), "materialization count drifted")
    return paths


def _encode_samples(paths: Mapping[str, Path], model_dir: Path, device: str, output: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    model_lock = dino._validate_model(model_dir, device)
    encoder = dino.DenseEncoder(model_dir, device)
    sample_ids = sorted(paths)
    tensors = [dino._crop_tensor(paths[sample_id], [0.0, 0.0, 1.0, 1.0]) for sample_id in sample_ids]
    encoded = encoder.encode(tensors)
    patches = {sample_id: value.astype("float32") for sample_id, value in zip(sample_ids, encoded)}
    pooled = {}
    for sample_id, value in patches.items():
        vector = value.mean(axis=0)
        pooled[sample_id] = (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype("float32")
    _atomic_npz(
        output,
        sample_ids=np.asarray(sample_ids),
        patches=np.stack([patches[key] for key in sample_ids]),
        pooled=np.stack([pooled[key] for key in sample_ids]),
    )
    return patches, pooled, {**model_lock, "encoded_samples": len(sample_ids), "forward_batches": encoder.forward_batches}


def _load_features(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        sample_ids = [str(value) for value in data["sample_ids"]]
        patches_array = data["patches"]
        pooled_array = data["pooled"]
    return (
        {key: patches_array[index] for index, key in enumerate(sample_ids)},
        {key: pooled_array[index] for index, key in enumerate(sample_ids)},
    )


def near_identity_loss(anchor: Any, positive: Any, hard: Any, ordinary: Any) -> Any:
    import torch

    temperature = LOSS["temperature"]
    similarities = torch.stack(
        [(anchor * positive).sum(dim=-1), (anchor * hard).sum(dim=-1), (anchor * ordinary).sum(dim=-1)], dim=1
    )
    disc = -torch.nn.functional.log_softmax(similarities / temperature, dim=1)[:, 0]
    rank = torch.nn.functional.softplus((similarities[:, 2] - similarities[:, 1]) / temperature)
    return (disc + LOSS["rank_weight"] * rank).mean()


def _make_head() -> Any:
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(HEAD_SPEC["input_dim"], HEAD_SPEC["hidden_dim"]),
        torch.nn.GELU(),
        torch.nn.Linear(HEAD_SPEC["hidden_dim"], HEAD_SPEC["output_dim"]),
    )


def _head_forward(head: Any, values: Any) -> Any:
    import torch

    return torch.nn.functional.normalize(head(values), dim=-1)


def _save_head(head: Any, path: Path) -> None:
    state = head.state_dict()
    _atomic_npz(path, **{key: value.detach().cpu().numpy() for key, value in state.items()})


def _load_head(path: Path, device: str) -> Any:
    import torch

    head = _make_head()
    with np.load(path, allow_pickle=False) as data:
        state = {key: torch.from_numpy(data[key]) for key in data.files}
    head.load_state_dict(state)
    return head.to(torch.device(device)).eval()


def _train_head(pooled: Mapping[str, np.ndarray], tuples: Sequence[Mapping[str, Any]], device: str, output: Path) -> dict[str, Any]:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    target_device = torch.device(device)
    head = _make_head().to(target_device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=OPTIMIZER["learning_rate"], weight_decay=OPTIMIZER["weight_decay"])
    rng = np.random.default_rng(SEED)
    losses = []
    for step in range(HEAD_SPEC["steps"]):
        indices = rng.integers(0, len(tuples), size=HEAD_SPEC["batch_size"])
        batch = [tuples[int(index)] for index in indices]
        tensors = []
        for role in ("anchor", "positive", "hard", "ordinary"):
            tensors.append(torch.from_numpy(np.stack([pooled[row[role]] for row in batch])).to(target_device))
        embeddings = [_head_forward(head, value) for value in tensors]
        loss = near_identity_loss(*embeddings)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, 99, 299, 599, 899, 1199}:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    head.eval()
    _save_head(head, output)
    return {"final_step": HEAD_SPEC["steps"], "loss_receipts": losses, "head_sha256": _sha256_file(output)}


def _challenger_embeddings(head: Any, pooled: Mapping[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    import torch

    keys = sorted(pooled)
    result = {}
    with torch.inference_mode():
        for start in range(0, len(keys), 128):
            batch_keys = keys[start : start + 128]
            values = torch.from_numpy(np.stack([pooled[key] for key in batch_keys])).to(torch.device(device))
            encoded = _head_forward(head, values).cpu().numpy()
            result.update({key: value for key, value in zip(batch_keys, encoded)})
    return result


def _score_pairs(
    pairs: Sequence[Mapping[str, Any]],
    patches: Mapping[str, np.ndarray],
    challenger: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mask = np.ones(dino.PATCH_COUNT, dtype=bool)
    result: dict[str, Any] = {"baseline": {}, "challenger": {}}
    for pair in pairs:
        pair_id = pair["pair_id"]
        reference_id = pair["reference"]
        for arm in result:
            result[arm][pair_id] = {"present": {}, "absence": {}}
        for task_name, slots in (("present", pair["present_slots"]), ("absence", pair["absence_slots"])):
            for slot in ("A", "B"):
                candidate_id = slots[slot]
                result["baseline"][pair_id][task_name][slot] = float(
                    dino.symmetric_local_score(patches[reference_id], patches[candidate_id], mask, mask)["symmetric_score"]
                )
                result["challenger"][pair_id][task_name][slot] = float(
                    challenger[reference_id] @ challenger[candidate_id]
                )
    return result


def _laplacian_variance(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _require(image is not None, f"cannot read quality image: {path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _quality_group(value: float, cutoffs: Sequence[float]) -> str:
    if value <= cutoffs[0]:
        return "LOW"
    if value <= cutoffs[1]:
        return "MID"
    return "HIGH"


def _decision(scores: Mapping[str, float], threshold: float) -> str:
    if max(scores.values()) < threshold or scores["A"] == scores["B"]:
        return "NONE"
    return "A" if scores["A"] > scores["B"] else "B"


def _group_metrics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    result = {}
    for value in sorted({str(row[field]) for row in rows}):
        group = [row for row in rows if str(row[field]) == value]
        result[value] = {
            "count": len(group),
            "same_instance_recall": sum(bool(row["correct"]) for row in group) / len(group),
            "coverage": sum(row["decision"] != "NONE" for row in group) / len(group),
        }
    return result


def evaluate_arm(
    pair_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    pairs: Sequence[Mapping[str, Any]],
    threshold: float,
    quality: Mapping[str, str],
) -> dict[str, Any]:
    present_rows = []
    absence_rows = []
    permutation_checks = 0
    for pair in pairs:
        pair_id = pair["pair_id"]
        scores = pair_scores[pair_id]
        present_decision = _decision(scores["present"], threshold)
        absence_decision = _decision(scores["absence"], threshold)
        permutation_checks += int(
            _decision({"A": scores["present"]["B"], "B": scores["present"]["A"]}, threshold)
            == ({"A": "B", "B": "A", "NONE": "NONE"}[present_decision])
        )
        present_rows.append(
            {
                "pair_id": pair_id,
                "category_id": pair["category_id"],
                "candidate_session_id": pair["candidate_session_id"],
                "quality_group": quality[pair_id],
                "scores": scores["present"],
                "target_slot": pair["target_slot"],
                "hard_slot": pair["hard_slot"],
                "decision": present_decision,
                "correct": present_decision == pair["target_slot"],
                "same_class_false_commit": present_decision == pair["hard_slot"],
                "target_margin": scores["present"][pair["target_slot"]] - scores["present"][pair["hard_slot"]],
            }
        )
        absence_rows.append(
            {
                "pair_id": pair_id,
                "scores": scores["absence"],
                "decision": absence_decision,
                "false_accept": absence_decision != "NONE",
                "ordinary_false_accept": absence_decision == pair["ordinary_slot"],
            }
        )
    count = len(present_rows)
    committed = [row for row in present_rows if row["decision"] != "NONE"]
    groups = {
        "category": _group_metrics(present_rows, "category_id"),
        "session": _group_metrics(present_rows, "candidate_session_id"),
        "quality": _group_metrics(present_rows, "quality_group"),
    }
    session_recalls = [row["same_instance_recall"] for row in groups["session"].values()]
    maxima = sorted({max(row["scores"].values()) for row in present_rows}, reverse=True)
    risk_coverage = []
    for score_threshold in maxima:
        selected = [row for row in present_rows if max(row["scores"].values()) >= score_threshold]
        risk_coverage.append(
            {
                "threshold": score_threshold,
                "coverage": len(selected) / count,
                "risk": sum(not row["correct"] for row in selected) / len(selected),
            }
        )
    return {
        "metrics": {
            "pair_count": count,
            "same_instance_recall": sum(row["correct"] for row in present_rows) / count,
            "same_class_false_commit": sum(row["same_class_false_commit"] for row in present_rows) / count,
            "coverage": len(committed) / count,
            "selective_risk": (sum(not row["correct"] for row in committed) / len(committed)) if committed else None,
            "target_absent_false_accept": sum(row["false_accept"] for row in absence_rows) / len(absence_rows),
            "ordinary_negative_false_accept": sum(row["ordinary_false_accept"] for row in absence_rows) / len(absence_rows),
            "candidate_permutation_invariance": permutation_checks / count,
            "source_session_min_same_instance_recall": min(session_recalls),
            "source_session_recall_gap": max(session_recalls) - min(session_recalls),
        },
        "groups": groups,
        "risk_coverage": risk_coverage,
        "present_rows": present_rows,
        "absence_rows": absence_rows,
    }


def _calibrate(
    scores: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    thresholds = {}
    for arm in ("baseline", "challenger"):
        absence_maxima = [max(scores[arm][pair["pair_id"]]["absence"].values()) for pair in pairs]
        threshold = select_absence_threshold(absence_maxima)
        thresholds[arm] = {
            "threshold": threshold,
            "calibration_absence_count": len(absence_maxima),
            "calibration_false_accept_count": sum(value >= threshold for value in absence_maxima),
        }
    target_quality = []
    for pair in pairs:
        target_id = pair["present_slots"][pair["target_slot"]]
        target_quality.append(_laplacian_variance(paths[target_id]))
    cutoffs = [float(value) for value in np.quantile(np.asarray(target_quality), [0.33, 0.67])]
    return {"thresholds": thresholds, "quality_laplacian_variance_cutoffs": cutoffs}


def _quality_map(pairs: Sequence[Mapping[str, Any]], paths: Mapping[str, Path], cutoffs: Sequence[float]) -> dict[str, str]:
    return {
        pair["pair_id"]: _quality_group(_laplacian_variance(paths[pair["present_slots"][pair["target_slot"]]]), cutoffs)
        for pair in pairs
    }


def _bind_inputs(protocol_path: Path, roster_path: Path, archive: Path, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    roster = _load_json(roster_path)
    _verify_body(protocol, "protocol")
    validate_roster(roster)
    _require(roster["protocol_body_sha256"] == protocol["body_sha256"], "roster/protocol binding drifted")
    _require(archive.stat().st_size == roster["archive"]["bytes"], "archive byte drift")
    _require(_sha256_file(archive) == roster["archive"]["sha256"], "archive SHA drift")
    for name, expected in dino.MODEL_FILES.items():
        _require(_sha256_file(model_dir / name).upper() == expected, f"model hash drift: {name}")
    return protocol, roster


def run_experiment(protocol_path: Path, roster_path: Path, archive: Path, model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    protocol, roster = _bind_inputs(protocol_path, roster_path, archive, model_dir)
    final_path = run_dir / "final-report.json"
    _require(not final_path.exists(), f"sealed final report exists; refusing rerun: {final_path}")
    pretest_path = run_dir / "pretest-lock.json"
    if run_dir.exists() and not pretest_path.exists():
        raise NearIdentityExperimentError("unlocked partial run exists; remove only this task-owned run and restart from step zero")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cal_features = run_dir / "train-cal-features.npz"
    head_path = run_dir / "near-identity-head.npz"
    calibration_path = run_dir / "calibration-report.json"
    if not pretest_path.exists():
        train_cal_paths = _materialize(archive, roster, run_dir, {"train", "calibration"})
        patches, pooled, model_lock = _encode_samples(train_cal_paths, model_dir, device, train_cal_features)
        training = _train_head(pooled, roster["train_tuples"], device, head_path)
        head = _load_head(head_path, device)
        challenger = _challenger_embeddings(head, pooled, device)
        calibration_scores = _score_pairs(roster["calibration_pairs"], patches, challenger)
        calibration = _calibrate(calibration_scores, roster["calibration_pairs"], train_cal_paths)
        calibration_report = {
            "schema_version": SCHEMA_VERSION,
            "protocol_body_sha256": protocol["body_sha256"],
            "roster_body_sha256": roster["body_sha256"],
            "model_lock": model_lock,
            "training": training,
            "calibration": calibration,
            "terminal": "CALIBRATION_COMPLETE_TEST_PIXELS_UNREAD",
        }
        calibration_report["body_sha256"] = _body_hash(calibration_report)
        _atomic_json(calibration_path, calibration_report)
        pretest = {
            "schema_version": SCHEMA_VERSION,
            "locked_at_utc": _utc_now(),
            "protocol_file_sha256": _sha256_file(protocol_path),
            "protocol_body_sha256": protocol["body_sha256"],
            "roster_file_sha256": _sha256_file(roster_path),
            "roster_body_sha256": roster["body_sha256"],
            "archive_sha256": roster["archive"]["sha256"],
            "model_files": dino.MODEL_FILES,
            "head_sha256": _sha256_file(head_path),
            "calibration_file_sha256": _sha256_file(calibration_path),
            "calibration_body_sha256": calibration_report["body_sha256"],
            "thresholds": calibration["thresholds"],
            "quality_laplacian_variance_cutoffs": calibration["quality_laplacian_variance_cutoffs"],
            "test_pixel_state_before_lock": "UNMATERIALIZED_UNREAD",
            "terminal": "SEALED_TEST_EXECUTION_LOCKED",
        }
        pretest["body_sha256"] = _body_hash(pretest)
        _atomic_json(pretest_path, pretest)
    pretest = _load_json(pretest_path)
    _verify_body(pretest, "pretest lock")
    _require(pretest["protocol_body_sha256"] == protocol["body_sha256"], "pretest protocol drift")
    _require(pretest["roster_body_sha256"] == roster["body_sha256"], "pretest roster drift")
    _require(pretest["head_sha256"] == _sha256_file(head_path), "head changed after pretest lock")
    test_paths = _materialize(archive, roster, run_dir, {"test"})
    test_features = run_dir / "test-features.npz"
    if test_features.exists():
        patches, pooled = _load_features(test_features)
        model_lock = dino._validate_model(model_dir, device)
    else:
        patches, pooled, model_lock = _encode_samples(test_paths, model_dir, device, test_features)
    head = _load_head(head_path, device)
    challenger = _challenger_embeddings(head, pooled, device)
    scores = _score_pairs(roster["test_pairs"], patches, challenger)
    raw = {
        "schema_version": SCHEMA_VERSION,
        "pretest_lock_body_sha256": pretest["body_sha256"],
        "model_runtime": model_lock,
        "scores": scores,
        "terminal": "RAW_TEST_UNARY_SCORES_SEALED",
    }
    raw["body_sha256"] = _body_hash(raw)
    raw_path = run_dir / "raw-test-scores.json"
    if raw_path.exists():
        existing = _load_json(raw_path)
        _verify_body(existing, "existing raw test scores")
        _require(existing["body_sha256"] == raw["body_sha256"], "deterministic raw test score drift")
        raw = existing
    else:
        _atomic_json(raw_path, raw)
    quality = _quality_map(roster["test_pairs"], test_paths, pretest["quality_laplacian_variance_cutoffs"])
    arms = {
        arm: evaluate_arm(
            raw["scores"][arm], roster["test_pairs"], pretest["thresholds"][arm]["threshold"], quality
        )
        for arm in ("baseline", "challenger")
    }
    baseline_rows = {row["pair_id"]: row for row in arms["baseline"]["present_rows"]}
    challenger_rows = {row["pair_id"]: row for row in arms["challenger"]["present_rows"]}
    rescues = sum(not baseline_rows[key]["correct"] and challenger_rows[key]["correct"] for key in baseline_rows)
    collateral = sum(baseline_rows[key]["correct"] and not challenger_rows[key]["correct"] for key in baseline_rows)
    controls = [key for key, row in baseline_rows.items() if row["correct"]]
    retained = sum(challenger_rows[key]["correct"] for key in controls)
    retention = retained / len(controls) if controls else 0.0
    metrics = arms["challenger"]["metrics"]
    gates = {
        "rescue_gt_collateral": rescues > collateral,
        "control_retention_ge_0p80": retention >= CONTROL_RETENTION_GATE,
        "candidate_permutation_invariance_eq_1": metrics["candidate_permutation_invariance"] == 1.0,
        "target_absent_false_accept_le_0p05": metrics["target_absent_false_accept"] <= FALSE_ACCEPT_LIMIT,
        "ordinary_negative_false_accept_le_0p05": metrics["ordinary_negative_false_accept"] <= FALSE_ACCEPT_LIMIT,
        "accepted_coverage_ge_0p50": metrics["coverage"] >= COVERAGE_GATE,
        "matched_context_construction": True,
        "source_session_min_recall_ge_0p40": metrics["source_session_min_same_instance_recall"] >= SOURCE_SESSION_RECALL_GATE,
        "source_session_recall_gap_le_0p35": metrics["source_session_recall_gap"] <= SOURCE_SESSION_GAP_GATE,
    }
    if all(gates.values()):
        outcome = "NEAR_IDENTITY_HARD_NEGATIVE_UNARY_SIGNAL_SUPPORTED_DEVELOPMENT"
    elif rescues > 0:
        outcome = "NEAR_IDENTITY_HARD_NEGATIVE_UNARY_MIXED_WITH_COLLATERAL_DEVELOPMENT"
    else:
        outcome = "NEAR_IDENTITY_HARD_NEGATIVE_UNARY_NOT_SUPPORTED_DEVELOPMENT"
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluated_at_utc": _utc_now(),
        "protocol_body_sha256": protocol["body_sha256"],
        "roster_body_sha256": roster["body_sha256"],
        "pretest_lock_body_sha256": pretest["body_sha256"],
        "raw_test_scores_body_sha256": raw["body_sha256"],
        "paired": {
            "rescue_count": rescues,
            "collateral_count": collateral,
            "net_change": rescues - collateral,
            "baseline_correct_control_count": len(controls),
            "challenger_control_retained_count": retained,
            "control_retention": retention,
        },
        "arms": arms,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "scientific_outcome": outcome,
        "occlusion": "NOT_EVALUABLE_SOURCE_LABEL",
        "reliable_verifier_status": "RELIABLE_VERIFIER_NOT_ESTABLISHED",
        "product_status": "NO_P1 / DEFAULT_APP_UNCHANGED",
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0_COMPLETE",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(final_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-protocol")
    freeze.add_argument("--protocol-doc", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    roster = subparsers.add_parser("freeze-roster")
    roster.add_argument("--protocol", type=Path, required=True)
    roster.add_argument("--archive", type=Path, required=True)
    roster.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--roster", type=Path, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-protocol":
        result = freeze_protocol(args.protocol_doc, args.output)
    elif args.command == "freeze-roster":
        result = freeze_roster(args.protocol, args.archive, args.output)
    else:
        result = run_experiment(args.protocol, args.roster, args.archive, args.model_dir, args.run_dir, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
