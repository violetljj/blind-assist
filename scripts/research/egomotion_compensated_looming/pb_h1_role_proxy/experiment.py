from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


K_BONN = np.asarray(
    (
        (542.822841, 0.0, 315.593520),
        (0.0, 542.576870, 237.756098),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
IMAGE_SIZE_WH = (640, 480)
DEPTH_UNITS_PER_METER = 5000.0
BONN_SEQUENCE = "rgbd_bonn_crowd2"
BONN_WINDOW_RANK = 0
BONN_WINDOW = ("1548339892.26121", "1548339902.26121")


def _fixture_pixels() -> np.ndarray:
    x = np.arange(48.0, 593.0, 16.0)
    y = np.arange(40.0, 441.0, 16.0)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


def run_controlled_fixture() -> dict[str, Any]:
    pixels = _fixture_pixels()
    depth = np.full(pixels.shape[0], 3.0, dtype=np.float64)
    dt_s = 0.1
    speed_m_s = 0.3
    angle = math.radians(8.0) * dt_s
    rotation_yaw = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    motions = {
        "pure_rotation": (
            rotation_yaw,
            np.zeros(3, dtype=np.float64),
        ),
        "lateral_translation": (
            np.eye(3, dtype=np.float64),
            np.asarray((-speed_m_s * dt_s, 0.0, 0.0)),
        ),
        "forward_approach": (
            np.eye(3, dtype=np.float64),
            np.asarray((0.0, 0.0, -speed_m_s * dt_s)),
        ),
    }
    summaries: dict[str, Any] = {}
    for name, (rotation, translation) in motions.items():
        summaries[name] = summarize_translation_induced_geometry(
            translation_induced_geometry(
                pixels,
                depth,
                K_BONN,
                rotation,
                translation,
                dt_s,
                image_size_wh=IMAGE_SIZE_WH,
                zbuffer=False,
            )
        )
    expected_forward = math.log(3.0 / (3.0 - speed_m_s * dt_s)) / dt_s
    pure = summaries["pure_rotation"]
    lateral = summaries["lateral_translation"]
    forward = summaries["forward_approach"]
    checks = {
        "pure_rotation_zero_translation_geometry": (
            pure["median_absolute_radial_expansion_per_s"] <= 1e-12
            and pure["q90_time_normalized_parallax_rad_per_s"] <= 1e-7
        ),
        "equal_speed_old_proxy_collision": abs(
            lateral["raw_translation_speed_m_s"]
            - forward["raw_translation_speed_m_s"]
        )
        <= 1e-12,
        "forward_analytic_radial_calibration": abs(
            forward["median_signed_radial_expansion_per_s"]
            - expected_forward
        )
        <= 1e-12,
        "forward_coherent_positive_expansion": (
            forward["radial_expansion_positive_fraction"] == 1.0
            and forward["median_signed_radial_expansion_per_s"] > 0.05
        ),
        "lateral_signed_radial_cancellation": (
            abs(lateral["median_signed_radial_expansion_per_s"]) < 0.01
            and 0.40
            <= lateral["radial_expansion_positive_fraction"]
            <= 0.60
        ),
        "lateral_parallax_exceeds_forward": (
            lateral["q90_time_normalized_parallax_rad_per_s"]
            > forward["q90_time_normalized_parallax_rad_per_s"]
        ),
    }
    return {
        "fixture_id": "PB-H1-FRONTOPARALLEL-PLANE-R0",
        "depth_m": 3.0,
        "dt_s": dt_s,
        "matched_translation_speed_m_s": speed_m_s,
        "point_count": int(pixels.shape[0]),
        "expected_forward_radial_expansion_per_s": expected_forward,
        "motions": summaries,
        "checks": checks,
        "physical_calibration_pass": bool(all(checks.values())),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_depth_index(raw: bytes) -> list[str]:
    paths: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise ValueError("PB_H1_BONN_DEPTH_INDEX")
        paths.append(tokens[1])
    return paths


def _decode_depth(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        array = np.asarray(image)
    if array.shape != (480, 640) or array.dtype != np.uint16:
        raise ValueError("PB_H1_BONN_DEPTH_PNG")
    return array


def _sample_depth(depth_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, 480, 8, dtype=np.int64)
    xs = np.arange(0, 640, 8, dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys)
    raw = depth_raw[yy, xx].ravel()
    valid = raw > 0
    pixels = np.column_stack((xx.ravel()[valid], yy.ravel()[valid])).astype(
        np.float64
    )
    return pixels, raw[valid].astype(np.float64) / DEPTH_UNITS_PER_METER


def run_burned_bonn_window(
    ledger_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    pairs = [
        row
        for row in ledger["pairs"]
        if row.get("sequence_id") == BONN_SEQUENCE
        and row.get("window_rank") == BONN_WINDOW_RANK
        and row.get("candidate") is True
        and row.get("truth_covered") is True
    ]
    if not pairs:
        raise ValueError("PB_H1_BONN_NO_COVERED_PAIR")
    pair_rows: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        depth_paths = _parse_depth_index(
            archive.read(f"{BONN_SEQUENCE}/depth.txt")
        )
        for row in pairs:
            rank = int(row["previous_depth_source_row_rank"])
            depth = _decode_depth(
                archive.read(f"{BONN_SEQUENCE}/{depth_paths[rank]}")
            )
            pixels, depth_m = _sample_depth(depth)
            rotation = np.asarray(
                [
                    [float.fromhex(value) for value in matrix_row]
                    for matrix_row in row["R_current_from_previous_hex"]
                ],
                dtype=np.float64,
            )
            translation = np.asarray(
                [
                    float.fromhex(value)
                    for value in row["t_current_from_previous_hex"]
                ],
                dtype=np.float64,
            )
            summary = summarize_translation_induced_geometry(
                translation_induced_geometry(
                    pixels,
                    depth_m,
                    K_BONN,
                    rotation,
                    translation,
                    float(row["dt"]),
                    image_size_wh=IMAGE_SIZE_WH,
                    zbuffer=True,
                )
            )
            pair_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "previous_depth_source_row_rank": rank,
                    **summary,
                }
            )
    evaluable = [row for row in pair_rows if row["evaluable"]]
    keys = (
        "raw_translation_speed_m_s",
        "median_signed_radial_expansion_per_s",
        "median_absolute_radial_expansion_per_s",
        "radial_expansion_positive_fraction",
        "q90_time_normalized_parallax_rad_per_s",
    )
    summary = {
        key: float(np.median([row[key] for row in evaluable]))
        for key in keys
    }
    summary.update(
        {
            "evaluable": bool(evaluable),
            "candidate_pair_count": len(pairs),
            "evaluable_pair_count": len(evaluable),
            "pair_coverage": len(evaluable) / len(pairs),
            "median_valid_fraction": float(
                np.median([row["valid_fraction"] for row in evaluable])
            ),
        }
    )
    return {
        "role": "BURNED_DIAGNOSTIC_ONLY_NOT_CONFIRMATION",
        "selection_rule": (
            "Deterministic first window in frozen B1A WINDOWS denominator; "
            "not selected using PB-H1 output."
        ),
        "sequence_id": BONN_SEQUENCE,
        "window_rank": BONN_WINDOW_RANK,
        "start": BONN_WINDOW[0],
        "end": BONN_WINDOW[1],
        "ledger_sha256": _sha256_file(ledger_path),
        "archive_sha256": _sha256_file(archive_path),
        "summary": summary,
        "pairs": pair_rows,
        "limitations": [
            "The source ledger belongs to the INVALID B1A execution and is "
            "used only as a burned geometry diagnostic.",
            "The measure predicts ego-translation-induced image geometry "
            "from previous depth; it does not estimate independent object "
            "motion or RCLE algorithm performance.",
            "Visibility uses in-bounds projection plus destination z-buffer; "
            "no current-depth static-scene consistency filter is applied.",
        ],
    }


def build_result(repo_root: Path) -> dict[str, Any]:
    ledger = (
        repo_root
        / "artifacts.local/evidence/rcle_phase_b_bonn_b1/"
        "b1a_geometry_admission/ledger.json"
    )
    archive = (
        repo_root
        / "artifacts.local/datasets/rcle_phase_b_bonn_b0_r1/archives/"
        "rgbd_bonn_crowd2.zip"
    )
    fixture = run_controlled_fixture()
    bonn = run_burned_bonn_window(ledger, archive)
    formula_correct = fixture["physical_calibration_pass"]
    old_proxy_misaligned = bool(
        fixture["checks"]["equal_speed_old_proxy_collision"]
        and fixture["checks"]["forward_coherent_positive_expansion"]
        and fixture["checks"]["lateral_signed_radial_cancellation"]
        and fixture["checks"]["lateral_parallax_exceeds_forward"]
    )
    bonn_evaluable = bool(
        bonn["summary"]["pair_coverage"] >= 0.80
        and bonn["summary"]["median_valid_fraction"] >= 0.50
    )
    audit_tum = bool(formula_correct and old_proxy_misaligned and bonn_evaluable)
    return {
        "schema_version": "rcle.pb_h1_role_proxy.result.v1",
        "protocol_id": "RCLE-PHASE-B-PROGRESSIVE-DISCOVERY-R0",
        "hypothesis_id": "PB-H1-ROLE-PROXY",
        "stage": "DISCOVERY",
        "claims_allowed": ["DATA_CHARACTERIZED"],
        "formula": {
            "decomposition": "X_r=R*X; X_f=R*X+t",
            "radial_expansion_per_s": "log(rho(X_f)/rho(X_r))/dt",
            "time_normalized_parallax_rad_per_s": (
                "acos(unit(X_r) dot unit(X_f))/dt"
            ),
            "raw_translation_speed_m_s": "norm(t)/dt",
            "radial_center": "calibrated principal point",
            "minimum_radius_px": 8.0,
            "depth_sampling": "8-pixel raster for Bonn; full fixed fixture grid",
            "visibility": (
                "positive depth, both projections in bounds, destination "
                "nearest-z winner"
            ),
            "aggregation": (
                "within-pair median signed/absolute radial expansion and Q90 "
                "parallax; Bonn window is the median of pair summaries"
            ),
        },
        "controlled_fixture": fixture,
        "burned_bonn_window": bonn,
        "conclusions": {
            "old_raw_speed_proxy_misaligned": old_proxy_misaligned,
            "formula_physically_correct_on_fixture": formula_correct,
            "bonn_diagnostic_evaluable": bonn_evaluable,
            "tum_fr2_rpy_audit_worthwhile": audit_tum,
            "important_qualification": (
                "Median absolute radial expansion alone is direction-agnostic; "
                "lateral versus approach separation uses signed radial "
                "coherence together with time-normalized parallax."
            ),
        },
        "terminal": (
            "PB_H1_SUPPORTS_TUM_FR2_RPY_METADATA_GEOMETRY_AUDIT"
            if audit_tum
            else "PB_H1_STOP_NO_TUM_AUDIT"
        ),
        "authority_boundary": (
            "Synthetic physical calibration plus one burned Bonn diagnostic; "
            "not confirmation, algorithm evidence, safety, or production "
            "authority."
        ),
    }
