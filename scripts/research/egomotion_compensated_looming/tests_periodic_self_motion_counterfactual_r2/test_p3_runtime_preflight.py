from __future__ import annotations

import copy
from unittest import mock
import unittest

import numpy as np

from ..periodic_self_motion_counterfactual_r2 import p3_runtime_preflight_r0 as runtime


class P3RuntimePreflightTests(unittest.TestCase):
    def test_exact_eight_identity_lock(self) -> None:
        manifest = runtime.identity_manifest()
        self.assertEqual(manifest["identity_count"], 8)
        self.assertEqual(len(manifest["identities"]), 8)
        self.assertEqual(manifest["worker_profiles"], [4, 8])
        self.assertEqual(manifest["prohibited_worker_profiles"], [12, 16])
        self.assertEqual(
            manifest["seeds"][0]["numeric_seed_uint64"],
            1727242067111453576,
        )
        self.assertEqual(
            manifest["seeds"][1]["numeric_seed_uint64"],
            18409799703140433944,
        )
        self.assertEqual(
            {item["arm"] for item in manifest["identities"][:6]},
            set(runtime.FACTORIAL_ARMS),
        )
        self.assertEqual(
            {item["arm"] for item in manifest["identities"][6:]},
            set(runtime.GUARDRAIL_ARMS),
        )
        self.assertTrue(
            all(item["frame_count"] == 602 for item in manifest["identities"])
        )
        self.assertTrue(
            all(item["pair_count"] == 601 for item in manifest["identities"])
        )

    def test_factorial_and_guardrail_seed_discriminator_is_not_aliased(self) -> None:
        factorial = runtime._seed_record("FACTORIAL")
        guardrail = runtime._seed_record("GUARDRAIL")
        self.assertNotEqual(factorial["token_sha256"], guardrail["token_sha256"])
        self.assertNotEqual(
            factorial["numeric_seed_uint64"],
            guardrail["numeric_seed_uint64"],
        )
        with self.assertRaisesRegex(runtime.InvalidPreflight, "KIND"):
            runtime._seed_record("factorial")

    def test_identity_mutations_fail_closed(self) -> None:
        mutated = runtime.identity_manifest()
        mutated["identities"][0]["frame_count"] = 601
        with self.assertRaisesRegex(runtime.InvalidPreflight, "DRIFT"):
            runtime.validate_identity_manifest(mutated)
        mutated = copy.deepcopy(runtime.identity_manifest())
        mutated["seeds"][0]["token"] = mutated["seeds"][0]["token"].lower()
        with self.assertRaisesRegex(runtime.InvalidPreflight, "DRIFT"):
            runtime.validate_identity_manifest(mutated)

    def test_only_frozen_worker_profiles_are_accepted(self) -> None:
        self.assertEqual(runtime.WORKER_PROFILES, (4, 8))
        for workers in (1, 12, 16):
            with self.assertRaisesRegex(runtime.InvalidPreflight, "WORKER"):
                runtime.run_profile(
                    runtime.transport.repo_root() / "missing.json",
                    runtime.transport.repo_root() / "unused",
                    workers,
                    1,
                )

    def test_preflight_scenes_use_distinct_exact_seeds(self) -> None:
        factorial = runtime._build_scene("FACTORIAL")
        guardrail = runtime._build_scene("GUARDRAIL")
        self.assertEqual(
            factorial["numeric_seed_uint64"],
            1727242067111453576,
        )
        self.assertEqual(
            guardrail["numeric_seed_uint64"],
            18409799703140433944,
        )
        self.assertNotEqual(
            factorial["scene_geometry_sha256"],
            guardrail["scene_geometry_sha256"],
        )

    def test_static_identity_renders_once_without_caching_r3_pairs(self) -> None:
        identity = runtime.identity_manifest()["identities"][0]
        rgb = np.zeros((runtime.geometry.HEIGHT, runtime.geometry.WIDTH, 3), dtype=np.uint8)
        mask = np.ones((runtime.geometry.HEIGHT, runtime.geometry.WIDTH), dtype=bool)
        row = {"pair_index": 0}
        with (
            mock.patch.object(runtime, "_render_frame", return_value=(rgb, mask)) as render,
            mock.patch.object(runtime.transport, "evaluate_pair", return_value=row) as evaluate,
            mock.patch.object(
                runtime,
                "_native_thread_guard",
                return_value=runtime._expected_native_thread_guard(),
            ),
        ):
            receipt = runtime._evaluate_identity(identity)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(evaluate.call_count, runtime.PAIR_COUNT)
        self.assertEqual(
            receipt["render_execution"],
            {
                "strategy": "STATIC_IDENTICAL_POSE_SINGLE_RENDER",
                "render_invocations": 1,
                "render_reuse_hits": runtime.FRAME_COUNT - 1,
            },
        )

    def test_worker_thread_guard_is_observed_not_only_declared(self) -> None:
        runtime._initialize_worker()
        expected = str(runtime.NATIVE_THREADS)
        self.assertEqual(runtime.os.environ["OMP_NUM_THREADS"], expected)
        self.assertEqual(runtime.os.environ["OPENBLAS_NUM_THREADS"], expected)
        self.assertEqual(runtime.os.environ["MKL_NUM_THREADS"], expected)
        self.assertEqual(runtime.os.environ["NUMEXPR_NUM_THREADS"], expected)
        self.assertEqual(runtime.os.environ["VECLIB_MAXIMUM_THREADS"], expected)
        self.assertEqual(runtime.os.environ["BLIS_NUM_THREADS"], expected)
        self.assertEqual(runtime.cv2.getNumThreads(), runtime.NATIVE_THREADS)


if __name__ == "__main__":
    unittest.main()
