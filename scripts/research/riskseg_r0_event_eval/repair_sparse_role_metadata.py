from __future__ import annotations

import argparse
import json
from pathlib import Path


def repair(root: Path) -> tuple[int, int]:
    inspected = repaired = 0
    for spec_path in root.rglob("candidate_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("status") != "output_blind_event_eval_rgb_screening_only":
            continue
        inspected += 1
        manifest = spec_path.with_name("manifest.rgb_timeline.jsonl")
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changed = False
        for row in rows:
            if row.pop("model_assisted_candidate_screening_only", None) is not None:
                changed = True
            expected = {
                "role": "output_blind_rgb_cost_control_screen_only",
                "model_output_accessed": False,
                "training_authorized": False,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    row[key] = value
                    changed = True
        if changed:
            temporary = manifest.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            temporary.replace(manifest)
            repaired += 1
    return inspected, repaired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    args = parser.parse_args()
    total_inspected = total_repaired = 0
    for root in args.root:
        inspected, repaired = repair(root)
        total_inspected += inspected
        total_repaired += repaired
    print(json.dumps({
        "ok": True,
        "inspected_manifests": total_inspected,
        "repaired_manifests": total_repaired,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
