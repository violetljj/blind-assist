#!/usr/bin/env python3
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0 import (
    classify_pair,
    lzf_decompress,
    object_row,
    oriented_box_points,
    read_pcd_xyz,
)


def literal_lzf(payload: bytes) -> bytes:
    output = bytearray()
    for start in range(0, len(payload), 32):
        chunk = payload[start : start + 32]
        output.append(len(chunk) - 1)
        output.extend(chunk)
    return bytes(output)


class SensorSupportBiasCanaryTest(unittest.TestCase):
    def test_lzf_literal_roundtrip(self) -> None:
        payload = bytes(range(100))
        encoded = literal_lzf(payload)
        self.assertEqual(lzf_decompress(encoded, len(payload)), payload)

    def test_binary_compressed_pcd_field_major_xyz(self) -> None:
        points = [(1.0, 2.0, 3.0), (-4.0, 5.5, 6.0), (7.0, 8.0, -9.0)]
        raw = b"".join(struct.pack("<f", point[axis]) for axis in range(3) for point in points)
        encoded = literal_lzf(raw)
        header = (
            b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
            b"COUNT 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA binary_compressed\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pcd"
            path.write_bytes(header + struct.pack("<II", len(encoded), len(raw)) + encoded)
            decoded, metadata = read_pcd_xyz(path)
        self.assertEqual(decoded, points)
        self.assertEqual(metadata["finite_points"], 3)

    def test_oriented_box_uses_length_x_width_y(self) -> None:
        box = {"cx": 0.0, "cy": 0.0, "cz": 0.0, "l": 4.0, "w": 2.0, "h": 2.0, "rot_z": 0.0}
        selected = oriented_box_points([(1.9, 0.9, 0.9), (2.1, 0.0, 0.0), (0.0, 1.1, 0.0)], box)
        self.assertEqual(selected, [(1.9, 0.9, 0.9)])

    def test_object_four_class_rules(self) -> None:
        frame = {"frame_index": 0, "frame_stem": "000000"}
        item_3d = {
            "label_id": "pedestrian:1",
            "box": {"cx": 0, "cy": 0, "cz": 0, "l": 2, "w": 2, "h": 2, "rot_z": 0},
            "attributes": {"interpolated": True, "num_points": 3},
        }
        item_2d = {"attributes": {"occlusion": "Fully_visible"}}
        supported = object_row("pedestrian:1", frame, item_3d, item_2d, [(0, 0, 0), (0.1, 0, 0)], [(0, 0.1, 0)], 3)
        sparse = object_row("pedestrian:1", frame, item_3d, item_2d, [(0, 0, 0)], [], 3)
        empty = object_row("pedestrian:1", frame, item_3d, item_2d, [], [], 3)
        missing = object_row("pedestrian:1", frame, None, item_2d, [], [], 3)
        self.assertEqual([supported["classification"], sparse["classification"], empty["classification"], missing["classification"]],
                         ["sensor-supported", "abstained", "annotation-only", "abstained"])

    def test_pair_precedence(self) -> None:
        self.assertEqual(classify_pair({"classification": "sensor-supported"}, {"classification": "sensor-supported"})[0], "sensor-supported")
        self.assertEqual(classify_pair({"classification": "sensor-supported"}, {"classification": "annotation-only"})[0], "annotation-only")
        self.assertEqual(classify_pair({"classification": "annotation-only"}, {"classification": "abstained"})[0], "abstained")
        self.assertEqual(classify_pair({"classification": "abstained"}, {"classification": "invalid"})[0], "invalid")


if __name__ == "__main__":
    unittest.main()
