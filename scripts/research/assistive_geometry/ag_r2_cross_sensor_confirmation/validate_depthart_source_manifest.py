"""Independent validator for the frozen DepthART runtime-source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

EXPECTED_SOURCE_ROOT = Path("F:/ba-data/blindassist-artifacts-20260805/models/depthart/source")
EXPECTED_PATHS = (
    "deploy/shared/selective_scan/depthart_selective_scan/__init__.py",
    "deploy/shared/selective_scan/depthart_selective_scan/cross_scan.py",
    "deploy/shared/selective_scan/depthart_selective_scan/integration.py",
    "deploy/shared/selective_scan/depthart_selective_scan/onnx.py",
    "deploy/shared/selective_scan/depthart_selective_scan/ops.py",
    "deploy/shared/selective_scan/depthart_selective_scan/optimize.py",
    "deploy/shared/selective_scan/depthart_selective_scan/precision.py",
    "deploy/shared/selective_scan/depthart_selective_scan/reference.py",
    "metric/common.py",
    "metric/dataset/transform.py",
    "metric/infer_dataset.py",
    "metric/infer_image.py",
    "metric/model.py",
    "metric/network/activation.py",
    "metric/network/attention.py",
    "metric/network/camera_embed.py",
    "metric/network/daa.py",
    "metric/network/dpt.py",
    "metric/network/layer_scale.py",
    "metric/network/mlp.py",
    "metric/network/sfh.py",
    "metric/network/tinyvim.py",
    "metric/network/tvimblock.py",
    "metric/network/util/blocks.py",
    "metric/network/util/geometric.py",
    "metric/network/util/positional_embedding.py",
    "metric/network/util/sht.py",
    "metric/network/util/transform.py",
    "metric/util/metric.py",
)


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(path: Path, *, expected_source_root: Path = EXPECTED_SOURCE_ROOT) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError("F2_DEPTHART_MANIFEST_READ") from error
    _require(isinstance(value, dict) and set(value) == {"schema", "source_root", "files"}, "F2_DEPTHART_MANIFEST_SCHEMA")
    _require(value["schema"] == "blindassist.depthart.source_manifest.v1", "F2_DEPTHART_MANIFEST_SCHEMA")
    root = Path(value["source_root"])
    _require(root.is_absolute() and root.resolve() == expected_source_root.resolve(), "F2_DEPTHART_MANIFEST_ROOT")
    rows = value["files"]
    _require(isinstance(rows, list) and len(rows) == len(EXPECTED_PATHS), "F2_DEPTHART_MANIFEST_COUNT")
    observed: list[str] = []
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "F2_DEPTHART_MANIFEST_ROW")
        relative = row["path"]
        _require(
            isinstance(relative, str)
            and PurePosixPath(relative).as_posix() == relative
            and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts,
            "F2_DEPTHART_MANIFEST_PATH",
        )
        observed.append(relative)
        member = root.joinpath(*PurePosixPath(relative).parts)
        _require(member.is_file(), "F2_DEPTHART_MANIFEST_MEMBER_MISSING")
        _require(type(row["bytes"]) is int and member.stat().st_size == row["bytes"], "F2_DEPTHART_MANIFEST_BYTES")
        _require(isinstance(row["sha256"], str) and _sha256(member) == row["sha256"].upper(), "F2_DEPTHART_MANIFEST_SHA")
    _require(tuple(observed) == EXPECTED_PATHS, "F2_DEPTHART_MANIFEST_EXACT_PATH_SET")
    return {
        "valid": True,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "source_root": str(root.resolve()),
        "model_or_checkpoint_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate(args.manifest)
    except ValidationError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
