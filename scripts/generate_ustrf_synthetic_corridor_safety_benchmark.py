#!/usr/bin/env python3
"""Generate and CUDA-audit labelled body-capsule corridor safety scenarios.

Every scene is an analytic local body grid with explicit traversable ground, hazards, and fault
truth.  It validates planning/supervisor contracts only; it contains neither camera pixels nor a
claim about perception or real-world assistive performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_ustrf_synthetic_corridor_safety_v1"
SEED = 20260720
LATERALS = tuple(range(-3, 4))
FORWARDS = tuple(range(1, 5))
OFFSETS = (-2, -1, 0, 1, 2)


def _pack(items: list[tuple[str, int, int]]) -> str:
    return ";".join(f"{kind}:{lateral}:{forward}" for kind, lateral, forward in items)


def _single(rng: random.Random, kind: str) -> list[tuple[str, int, int]]:
    return [(kind, rng.choice(LATERALS), rng.choice(FORWARDS))]


def _scenes() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    templates: list[tuple[str, str, list[tuple[str, int, int]], str]] = []
    templates += [("clear", "none", [], "none") for _ in range(32)]
    for family, kind, count in (("occupancy", "O", 44), ("drop", "D", 40), ("head", "H", 40), ("dynamic", "M", 40)):
        templates += [(family, "none", _single(rng, kind), "none") for _ in range(count)]
    for index in range(16):
        kind = ("O", "D", "H", "M")[index % 4]
        templates.append(("full_width_block", "none", [(kind, lateral, 2) for lateral in LATERALS], "none"))
    templates += [("central_unknown", "none", [("U", 0, rng.choice(FORWARDS))], "none") for _ in range(16)]
    templates += [("fault_pose_lost", "pose_lost", [], "pose_lost") for _ in range(8)]
    templates += [("fault_stale_geometry", "stale_geometry", [], "stale_geometry") for _ in range(8)]
    for _ in range(12):
        count = rng.randint(2, 5)
        kinds = [rng.choice(("O", "D", "H", "M")) for _ in range(count)]
        templates.append(("mixed", "none", [(kind, rng.choice(LATERALS), rng.choice(FORWARDS)) for kind in kinds], "none"))
    assert len(templates) == 256
    return [
        {"scenario_id": f"corridor-{index:03d}-{family}", "family": family, "fault": fault, "hazards": _pack(hazards), "expected_action": "", "expected_selected_offset": "", "expected_has_safe_corridor": "", "expected_critical": ""}
        for index, (family, fault, hazards, _) in enumerate(templates)
    ]


def _truth(rows: list[dict[str, str]]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for synthetic corridor benchmark")
    hazards = torch.zeros((len(rows), len(FORWARDS), len(LATERALS)), dtype=torch.bool, device="cuda")
    central_unknown = torch.zeros(len(rows), dtype=torch.bool, device="cuda")
    unknown_cells = torch.zeros_like(hazards)
    for index, row in enumerate(rows):
        for token in filter(None, row["hazards"].split(";")):
            kind, lateral, forward = token.split(":")
            if kind == "U":
                central_unknown[index] = True
                unknown_cells[index, int(forward) - 1, int(lateral) + 3] = True
            else:
                hazards[index, int(forward) - 1, int(lateral) + 3] = True
    safe = []
    for offset in OFFSETS:
        # Python slice end is exclusive: include all three capsule cells
        # [offset-1, offset, offset+1], not just its left two cells.
        capsule = (hazards | unknown_cells)[:, :, offset - 1 + 3:offset + 1 + 3 + 1]
        safe.append(~capsule.any(dim=(1, 2)))
    safe_tensor = torch.stack(safe, dim=1)
    for index, row in enumerate(rows):
        has_safe = bool(safe_tensor[index].any().item())
        selected = next((offset for offset, allowed in zip(OFFSETS, safe_tensor[index].tolist()) if allowed), None)
        # Stable planner tie break: closest to desired 0, then left.
        if has_safe:
            selected = min((offset for offset, allowed in zip(OFFSETS, safe_tensor[index].tolist()) if allowed), key=lambda offset: (abs(offset), offset))
        critical = bool(hazards[index].any().item()) or bool(central_unknown[index].item()) or row["fault"] != "none"
        action = "STOP_AND_REASSESS" if row["fault"] != "none" or bool(central_unknown[index].item()) or not has_safe else "SLOW_DOWN"
        row["expected_has_safe_corridor"] = str(has_safe).lower()
        # Fault-injected scenes deliberately have no authorized corridor selection even when
        # their latent ground map happens to be clear.
        row["expected_selected_offset"] = "" if selected is None or row["fault"] != "none" else str(selected)
        row["expected_action"] = action
        row["expected_critical"] = str(critical).lower()


def generate(root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark root: {root}")
    root.mkdir(parents=True)
    rows = _scenes()
    _truth(rows)
    columns = list(rows[0])
    with (root / "kotlin_corridor_safety_replay.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    spec = {
        "format": SCHEMA, "seed": SEED, "coordinate_frame": "synthetic-body-local-v1",
        "grid": {"cell_size_meters": .5, "laterals": list(LATERALS), "forwards": list(FORWARDS), "capsule_half_width_cells": 1, "candidate_offsets": list(OFFSETS)},
        "classes": {"O": "occupancy", "D": "drop", "H": "head_obstacle", "M": "dynamic_ttc", "U": "unknown"},
        "scene_count": len(rows), "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "annotation_target": "dependency-free Kotlin replay TSV + GPU truth audit", "production_authority": False,
    }
    (root / "dataset_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return audit(root)


def audit(root: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for synthetic corridor benchmark audit")
    with (root / "kotlin_corridor_safety_replay.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    expected_critical = torch.tensor([row["expected_critical"] == "true" for row in rows], device="cuda")
    stops = torch.tensor([row["expected_action"] == "STOP_AND_REASSESS" for row in rows], device="cuda")
    clear = torch.tensor([row["family"] == "clear" for row in rows], device="cuda")
    report = {
        "format": "blindassist_ustrf_synthetic_corridor_safety_audit_v1",
        "dataset_format": SCHEMA, "scene_count": len(rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "critical_scene_count": int(expected_critical.sum().item()),
        "expected_stop_count": int(stops.sum().item()),
        "clear_scene_count": int(clear.sum().item()),
        "expected_clear_stop_count": int((stops & clear).sum().item()),
        "body_frame_ground_truth": True,
        "local_ground_truth": True,
        "dynamic_event_truth": True,
        "production_authority": False,
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    preview = "\n".join(f"<tr><td>{row['scenario_id']}</td><td>{row['family']}</td><td>{row['hazards'] or 'clear'}</td><td>{row['expected_action']}</td><td>{row['expected_selected_offset'] or '-'}</td></tr>" for row in rows)
    (qa / "preview.html").write_text(f"<!doctype html><meta charset='utf-8'><title>USTRF synthetic corridor QA</title><table><tr><th>id</th><th>family</th><th>truth</th><th>action</th><th>offset</th></tr>{preview}</table>", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = audit(args.output) if args.audit_only else generate(args.output)
    print(json.dumps({"scenes": report["scene_count"], "critical": report["critical_scene_count"], "backend": report["compute_backend"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
