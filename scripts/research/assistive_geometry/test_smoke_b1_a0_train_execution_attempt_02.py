import unittest
from unittest.mock import patch

import torch

from scripts.research.assistive_geometry import smoke_b1_a0_train_execution_attempt_02 as wrapper


class A0TrainExecutionAttempt02Test(unittest.TestCase):
    def test_wrapper_forces_cpu_checkpoint_restore(self) -> None:
        observed = {}

        def fake_load(*args, **kwargs):
            observed.update(kwargs)
            return {}

        def fake_main() -> int:
            torch.load("checkpoint.pt", map_location="cuda", weights_only=False)
            return 7

        with patch.object(wrapper.attempt_01, "main", side_effect=fake_main):
            with patch.object(torch, "load", side_effect=fake_load):
                self.assertEqual(7, wrapper.main())
        self.assertEqual("cpu", observed["map_location"])


if __name__ == "__main__":
    unittest.main()
