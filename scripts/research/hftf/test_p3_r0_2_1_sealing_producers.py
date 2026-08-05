from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).parent


def load(name: str):
    path = BASE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


derive = load("derive_p3_r0_2_1_bonn_private_targets.py")
seal = load("seal_p3_r0_2_1_target_bundle.py")
coverage = load("produce_p3_r0_2_1_aggregate_coverage_receipt.py")
keygen = load("generate_p3_r0_2_1_sealing_key.py")
common = load("p3_r0_2_1_sealing_common.py")


class SealingProducerTest(unittest.TestCase):
    def test_clearance_state_threshold_is_frozen(self) -> None:
        self.assertEqual(derive.state_for(None), "UNKNOWN_GROUND")
        self.assertEqual(derive.state_for(1.5), "OCCUPIED")
        self.assertEqual(derive.state_for(1.500001), "CLEAR")

    def test_authenticated_envelope_is_deterministic_and_tamper_evident(self) -> None:
        key, nonce, plain, associated = bytes(range(32)), bytes(range(12)), b"private targets", b"bindings"
        first = seal.seal(key, nonce, plain, associated)
        second = seal.seal(key, nonce, plain, associated)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(seal.MAGIC))
        self.assertNotEqual(first[-32:], seal.seal(key, nonce, plain + b"x", associated)[-32:])

    def test_coverage_counts_all_nine_classes_and_key_transitions(self) -> None:
        clips = []
        sequences = [
            ["CLEAR", "OCCUPIED", "UNKNOWN_GROUND", "CLEAR"],
            ["OCCUPIED", "CLEAR", "UNKNOWN_GROUND", "OCCUPIED"],
        ]
        for index in range(32):
            states = sequences[index % 2]
            frames = []
            for state in states:
                frames.append({"bands": [{"geometry_state": state, "geometry_target_valid": True}] * 3})
            clips.append({"parent_id": f"p{index % 8}", "frames": frames})
        key, geometry, evaluable, parents = coverage.coverage({"clips": clips})
        self.assertEqual(evaluable, 32)
        self.assertEqual(parents, 8)
        self.assertEqual(set(geometry), set(coverage.TRANSITIONS))
        self.assertTrue(all(value >= 8 for value in key.values()))

    def test_outputs_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exists"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
                coverage.exclusive_write(path, b"y")

    def test_lexical_repo_boundary_accepts_artifacts_junction_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = common.resolve_inside(root, "artifacts.local/evidence/value.json")
            self.assertTrue(resolved.is_relative_to(root))
            with self.assertRaisesRegex(ValueError, "path leaves repository"):
                common.resolve_inside(root, "../escape.json")

    def test_key_generator_writes_exactly_32_private_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {
                "schema": keygen.REQUEST_SCHEMA,
                "producer_sha256": keygen.sha256_file(Path(keygen.__file__)),
                "outputs": {"key": "key.bin", "receipt": "receipt.json"},
            }
            keygen.build(root, request, Path(keygen.__file__), random_bytes=lambda size: bytes([7]) * size)
            self.assertEqual((root / "key.bin").read_bytes(), bytes([7]) * 32)
            self.assertNotIn("070707", (root / "receipt.json").read_text(encoding="utf-8"))

    def test_no_model_or_optimizer_construction(self) -> None:
        for name in (
            "derive_p3_r0_2_1_bonn_private_targets.py",
            "seal_p3_r0_2_1_target_bundle.py",
            "produce_p3_r0_2_1_aggregate_coverage_receipt.py",
            "generate_p3_r0_2_1_sealing_key.py",
        ):
            source = (BASE / name).read_text(encoding="utf-8")
            for forbidden in ("torch.optim", "AdamW(", "load_state_dict", "optimizer ="):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
