import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from host_reference_pipeline import build_pipeline


class FakeInterpreter:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeInterpreter.last_kwargs = kwargs

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"shape": np.array([1, 320, 320, 3]), "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 1}]


class HostReferencePipelineTest(unittest.TestCase):
    def test_thread_count_is_bound_to_interpreter_and_identity(self):
        parent = ModuleType("ai_edge_litert")
        interpreter_module = ModuleType("ai_edge_litert.interpreter")
        interpreter_module.Interpreter = FakeInterpreter
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.touch()
            with patch.dict(
                sys.modules,
                {
                    "ai_edge_litert": parent,
                    "ai_edge_litert.interpreter": interpreter_module,
                },
            ):
                pipeline = build_pipeline(model, "record_only", num_threads=4)

        self.assertEqual(pipeline.num_threads, 4)
        self.assertIn("TFLITE_THREADS_4", pipeline.identity)
        self.assertEqual(FakeInterpreter.last_kwargs["num_threads"], 4)

    def test_non_positive_thread_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.touch()
            with self.assertRaisesRegex(ValueError, "positive"):
                build_pipeline(model, "record_only", num_threads=0)


if __name__ == "__main__":
    unittest.main()
