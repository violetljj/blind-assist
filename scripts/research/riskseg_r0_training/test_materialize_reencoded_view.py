from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .materialize_reencoded_view import (
    MaterializationError,
    _canonicalize_source,
    materialize,
    sha256_file,
)
from scripts.research.dual_loop_segmentation_r2_p0.canonicalizer import load_contract


REPO_ROOT = Path(__file__).resolve().parents[3]
TMP_ROOT = REPO_ROOT / "artifacts.local" / "tmp" / "riskseg-r0-training-tests"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


class FrozenFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions = {
            "train": {
                "train-a": 50,
                "train-b": 50,
                "train-c": 50,
                "train-d": 50,
                "train-e": 60,
                "train-f": 60,
            },
            "dev": {
                "dev-a": 50,
                "dev-b": 50,
                "dev-c": 50,
                "dev-d": 50,
            },
        }
        self.training = root / "dataset" / "training_manifest.jsonl"
        self.blind = root / "dataset" / "blind_holdout" / "manifest.jsonl"
        self.fresh = root / "fresh_holdout" / "manifest.jsonl"
        self.ledger = root / "ledger.json"
        self._build()

    def _asset(
        self,
        base: Path,
        session: str,
        *,
        mask_values: np.ndarray | None = None,
    ) -> tuple[Path, Path]:
        image = base / "images" / f"{session}.png"
        mask = base / "masks" / f"{session}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(17, 23, 31)).save(image)
        values = (
            mask_values
            if mask_values is not None
            else np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
        )
        Image.fromarray(values.astype(np.uint8), mode="L").save(mask)
        return image, mask

    def _rows(
        self,
        *,
        manifest: Path,
        asset_base: Path,
        sessions: dict[str, int],
        prefix: str,
        source_native: bool = False,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for session, count in sessions.items():
            mask_values = (
                np.asarray([[1, 2], [4, 0]], dtype=np.uint8)
                if source_native
                else None
            )
            image, mask = self._asset(
                asset_base,
                session,
                mask_values=mask_values,
            )
            image_rel = image.relative_to(asset_base).as_posix()
            mask_rel = mask.relative_to(asset_base).as_posix()
            for index in range(count):
                rows.append(
                    {
                        "id": f"{prefix}-{session}-{index:03d}",
                        "session_id": session,
                        "sequence_id": f"{session}:fixture",
                        "frame_index": index,
                        "image_path": image_rel,
                        "image_sha256": sha256_file(image),
                        "semantic_mask_path": mask_rel,
                        "semantic_mask_sha256": sha256_file(mask),
                    }
                )
        return rows

    def _build(self) -> None:
        training_rows = self._rows(
            manifest=self.training,
            asset_base=self.training.parent,
            sessions=self.sessions["dev"],
            prefix="dev",
        )
        # The frozen source manifest has 600 rows but only the four declared dev
        # sessions are selected.
        training_rows.extend(
            self._rows(
                manifest=self.training,
                asset_base=self.training.parent,
                sessions={f"unused-{index}": 50 for index in range(8)},
                prefix="unused",
            )
        )
        _write_jsonl(self.training, training_rows)

        # This reproduces the frozen blind manifest's one-level asset-root
        # indirection: manifest under blind_holdout, assets under dataset.
        blind_rows = self._rows(
            manifest=self.blind,
            asset_base=self.blind.parent.parent,
            sessions={"train-e": 60, "train-f": 60},
            prefix="blind",
        )
        _write_jsonl(self.blind, blind_rows)

        fresh_rows = self._rows(
            manifest=self.fresh,
            asset_base=self.fresh.parent,
            sessions={
                "train-a": 50,
                "train-b": 50,
                "train-c": 50,
                "train-d": 50,
            },
            prefix="fresh",
            source_native=True,
        )
        _write_jsonl(self.fresh, fresh_rows)
        self.refresh_ledger()

    def refresh_ledger(self) -> None:
        ledger = {
            "schema_version": "blindassist.riskseg_r0.data_role_ledger.v1",
            "protocol_id": "RISKSEG_R0",
            "class_contract": {
                "legacy_to_riskseg_r0_id": {"0": 0, "1": 2, "2": 1, "3": 3}
            },
            "roles": {
                role: {
                    "role": (
                        "TRAIN_ONLY_CONSUMED_DEVELOPMENT"
                        if role == "train"
                        else "DEV_SELECTION_ONLY_CONSUMED_DEVELOPMENT"
                    ),
                    "frame_count": sum(sessions.values()),
                    "session_count": len(sessions),
                    "sessions": sessions,
                }
                for role, sessions in self.sessions.items()
            },
            "source_manifests": [
                {
                    "role_slice": "dev_200",
                    "path": self.training.relative_to(REPO_ROOT).as_posix(),
                    "whole_manifest_row_count": 600,
                    "selected_row_count": 200,
                    "sha256": sha256_file(self.training),
                },
                {
                    "role_slice": "train_consumed_old_blind_120",
                    "path": self.blind.relative_to(REPO_ROOT).as_posix(),
                    "selected_row_count": 120,
                    "sha256": sha256_file(self.blind),
                },
                {
                    "role_slice": "train_r1_consumed_fresh_200",
                    "path": self.fresh.relative_to(REPO_ROOT).as_posix(),
                    "selected_row_count": 200,
                    "sha256": sha256_file(self.fresh),
                },
            ],
            "riskseg_r0_forward_only_overlay": {
                "minimum_nonzero_class_coverage_sessions": {
                    "train_blocking_obstacle": 2,
                    "train_boundary_level_change": 2,
                    "dev_blocking_obstacle": 2,
                    "dev_boundary_level_change": 2,
                }
            },
        }
        _write_json(self.ledger, ledger)

    def rows(self, manifest: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ]

    def replace_rows(self, manifest: Path, rows: list[dict[str, object]]) -> None:
        _write_jsonl(manifest, rows)
        self.refresh_ledger()


class MaterializeReencodedViewTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        self.root = Path(self.temporary.name)
        self.fixture = FrozenFixture(self.root)
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materializes_exact_frozen_view_and_receipt(self) -> None:
        receipt = materialize(
            repo_root=REPO_ROOT,
            ledger_path=self.fixture.ledger,
            output_root=self.output,
        )
        self.assertEqual(receipt["row_count"], 520)
        self.assertEqual(receipt["role_counts"], {"dev": 200, "train": 320})
        self.assertEqual(receipt["session_counts"], self.fixture.sessions)
        self.assertTrue(receipt["atomic_publish"])
        self.assertEqual(
            receipt["class_pixel_counts"]["total"],
            {
                "0": 8_519_680,
                "1": 8_519_680,
                "2": 8_519_680,
                "3": 8_519_680,
            },
        )
        self.assertTrue(
            all(gate["passed"] for gate in receipt["coverage_gates"].values())
        )
        manifest_rows = [
            json.loads(line)
            for line in (self.output / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(manifest_rows), 520)
        first_mask = self.output / manifest_rows[0]["output_mask_path"]
        with Image.open(first_mask) as image:
            values = np.asarray(image)
            self.assertEqual(values.shape, (256, 256))
            self.assertEqual(
                [
                    int(values[0, 0]),
                    int(values[0, -1]),
                    int(values[-1, 0]),
                    int(values[-1, -1]),
                ],
                [0, 2, 1, 3],
            )
        self.assertEqual(
            sha256_file(first_mask), manifest_rows[0]["output_mask_sha256"]
        )
        self.assertIn("source_mask_sha256", manifest_rows[0])
        self.assertIn(manifest_rows[0]["role"], {"train", "dev"})
        fresh_row = next(
            row
            for row in manifest_rows
            if row["source_manifest_role_slice"]
            == "train_r1_consumed_fresh_200"
        )
        self.assertEqual(fresh_row["source_mask_decoder"], "source_native")
        self.assertRegex(fresh_row["old_canonical_png_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            fresh_row["source_to_old_canonical_mapping_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            fresh_row["old_canonical_to_riskseg_r0_mapping_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_rejects_unknown_native_mask_id(self) -> None:
        source = self.root / "unknown.png"
        Image.fromarray(np.asarray([[0, 31]], dtype=np.uint8), mode="L").save(source)
        contract = load_contract(
            REPO_ROOT
            / "configs"
            / "dual_loop_segmentation_r2_p0"
            / "canonicalization_contract.json"
        )
        with self.assertRaisesRegex(MaterializationError, "unmapped IDs"):
            _canonicalize_source(
                source_path=source,
                decoder="source_native",
                contract=contract,
            )

    def test_rejects_duplicate_selected_id_without_publication(self) -> None:
        rows = self.fixture.rows(self.fixture.blind)
        rows[1]["id"] = rows[0]["id"]
        self.fixture.replace_rows(self.fixture.blind, rows)
        with self.assertRaisesRegex(MaterializationError, "duplicate selected row ID"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".output.tmp-*")), [])

    def test_rejects_session_count_drift(self) -> None:
        rows = self.fixture.rows(self.fixture.blind)
        rows[0]["session_id"] = "train-f"
        self.fixture.replace_rows(self.fixture.blind, rows)
        with self.assertRaisesRegex(MaterializationError, "session/frame count drift"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_path_escape_before_asset_access(self) -> None:
        rows = self.fixture.rows(self.fixture.blind)
        rows[0]["image_path"] = "../escape.png"
        self.fixture.replace_rows(self.fixture.blind, rows)
        with self.assertRaisesRegex(MaterializationError, "path escape rejected"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_source_hash_drift(self) -> None:
        image = self.fixture.blind.parent.parent / "images" / "train-e.png"
        image.write_bytes(b"corrupt")
        with self.assertRaisesRegex(MaterializationError, "image SHA-256 mismatch"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_source_manifest_hash_drift(self) -> None:
        with self.fixture.blind.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(
            MaterializationError, "source manifest SHA-256 mismatch"
        ):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_insufficient_role_session_class_coverage(self) -> None:
        rows = self.fixture.rows(self.fixture.training)
        for session in ("dev-b", "dev-c", "dev-d"):
            mask = self.fixture.training.parent / "masks" / f"{session}.png"
            Image.fromarray(
                np.asarray([[0, 0], [0, 3]], dtype=np.uint8), mode="L"
            ).save(mask)
            new_sha = sha256_file(mask)
            for row in rows:
                if row["session_id"] == session:
                    row["semantic_mask_sha256"] = new_sha
        self.fixture.replace_rows(self.fixture.training, rows)
        with self.assertRaisesRegex(MaterializationError, "DATA_SPLIT_NOT_READY"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_output_outside_artifacts_local(self) -> None:
        with self.assertRaisesRegex(MaterializationError, "artifacts.local"):
            materialize(
                repo_root=REPO_ROOT,
                ledger_path=self.fixture.ledger,
                output_root=REPO_ROOT / "unsafe-output",
            )


if __name__ == "__main__":
    unittest.main()
