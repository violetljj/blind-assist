from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from scripts.research.assistive_geometry.assistive_geometry_model import AssistiveTaskHeads
from scripts.research.assistive_geometry.export_assistive_geometry_onnx import (
    ExternalCameraAssistiveGeometry,
    OUTPUT_NAMES,
    parity_summary,
)


class DummyPretrained(nn.Module):
    def forward_with_adapters(self, image, adapters, cams):
        del adapters, cams
        feature = image.mean(dim=1, keepdim=True).repeat(1, 48, 1, 1)
        return [feature, feature, feature, feature]


class DummyScale(nn.Module):
    def forward(self, feature, camera):
        del camera
        return torch.ones(feature.shape[0], device=feature.device)


class DummyMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = DummyPretrained()
        self.daa1 = nn.Identity()
        self.daa2 = nn.Identity()
        self.daa3 = nn.Identity()
        self.daa4 = nn.Identity()
        self.sfh = DummyScale()
        self.max_depth = 2.0


class DummyGeometry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metric_depthart = DummyMetric()
        self.assistive_heads = AssistiveTaskHeads()

    @staticmethod
    def _decode(features, output_hw):
        del output_hw
        return features[0][:, :1], features[0]


class AssistiveGeometryExportTests(unittest.TestCase):
    def test_external_wrapper_preserves_all_raw_outputs(self) -> None:
        model = DummyGeometry().eval()
        wrapper = ExternalCameraAssistiveGeometry(model).eval()
        image = torch.randn(1, 3, 12, 9)
        camera = torch.zeros(1, 4, 3, 3)
        outputs = wrapper(image, camera, camera, camera, camera)
        self.assertEqual(len(outputs), len(OUTPUT_NAMES))
        self.assertEqual(tuple(outputs[0].shape), (1, 1, 12, 9))
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 12, 9))
        self.assertEqual(tuple(outputs[2].shape), (1, 3))
        self.assertEqual(tuple(outputs[3].shape), (1, 3, 3))
        self.assertEqual(tuple(outputs[4].shape), (1, 3))

    def test_parity_summary_is_exact_for_same_tensors(self) -> None:
        values = tuple(torch.randn(1, index + 1) for index in range(len(OUTPUT_NAMES)))
        result = parity_summary(values, tuple(value.clone() for value in values))
        self.assertTrue(all(item["max_abs"] == 0.0 for item in result.values()))

    def test_parity_summary_rejects_output_drift(self) -> None:
        values = tuple(torch.zeros(1, index + 1) for index in range(len(OUTPUT_NAMES)))
        drift = list(value.clone() for value in values)
        drift[2][0, 0] = 1e-3
        with self.assertRaises(ValueError):
            parity_summary(values, tuple(drift))


if __name__ == "__main__":
    unittest.main()
