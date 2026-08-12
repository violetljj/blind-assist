from __future__ import annotations

import ast
import hashlib
import struct
import zipfile
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts.research.assistive_geometry import (
    ag_r2_cross_sensor_confirmation as package,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    eth3d_source as source,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    ContractError,
)

PARENT = "plant_scene_2"


def _png(array: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


def _rgb_payload(index: int) -> bytes:
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[:, :, 0] = index
    bgr[:, :, 1] = 20
    bgr[:, :, 2] = 30
    return _png(bgr)


def _depth_payload() -> bytes:
    return _png(np.asarray(((0, 1249, 1250), (30000, 30001, 5000)), dtype=np.uint16))


def _archive_row(path: Path, *, kind: str = "RGBD_TRAINING_ARCHIVE", parent_id: str = PARENT) -> dict[str, object]:
    if kind == "RGBD_TRAINING_ARCHIVE":
        filename = f"{parent_id}_rgbd.zip"
        url = f"https://www.eth3d.net/data/slam/datasets/{filename}"
    elif kind == "IMU_ARCHIVE":
        filename = f"{parent_id}_imu.zip"
        url = f"https://www.eth3d.net/data/slam/datasets/{filename}"
    else:
        filename = "camera_imu_calib_radtan.zip"
        url = f"https://www.eth3d.net/data/slam/{filename}"
    assert path.name == filename
    payload = path.read_bytes()
    return {
        "parent_id": parent_id,
        "kind": kind,
        "url": url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def _associated_rows(count: int = 24) -> list[str]:
    return [
        f"{(index + 1) * 1_000_000_000} rgb/{index:06d}.png "
        f"{(index + 1) * 1_000_000_000} depth/{index:06d}.png\n"
        for index in range(count)
    ]


def _write_rgbd(
    root: Path,
    *,
    associated_rows: Iterable[str] | None = None,
    count: int = 24,
    extra_members: Iterable[tuple[str, bytes]] = (),
    groundtruth: str = "# synthetic\n",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{PARENT}_rgbd.zip"
    rows = list(_associated_rows(count) if associated_rows is None else associated_rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{PARENT}/associated.txt", "".join(rows))
        archive.writestr(f"{PARENT}/rgb.txt", "# synthetic\n")
        archive.writestr(f"{PARENT}/depth.txt", "# synthetic\n")
        archive.writestr(f"{PARENT}/calibration.txt", "100 110 1.0 0.5\n")
        archive.writestr(f"{PARENT}/groundtruth.txt", groundtruth)
        for index in range(count):
            archive.writestr(f"{PARENT}/rgb/{index:06d}.png", _rgb_payload(index))
            archive.writestr(f"{PARENT}/depth/{index:06d}.png", _depth_payload())
        for name, payload in extra_members:
            archive.writestr(name, payload)
    return path


def _open_rgbd(path: Path, *, observer=None, budget: source.ArchiveBudget | None = None):
    binding = source.ArchiveBinding.from_manifest_row(_archive_row(path))
    verified = source.verify_archive_binding(path.parent, binding)
    return source.preflight_archive(
        verified,
        budget=budget or source.ArchiveBudget(max_members=1000, max_total_uncompressed_bytes=4 << 20),
        observer=observer,
    )


def _write_imu(
    root: Path,
    *,
    imu_rows: str,
    mocap_scale: str = "2",
    mocap_anchor_seconds: str = "0",
    mocap_offset_seconds: str = "0.25",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return _write_plain_zip(
        root / f"{PARENT}_imu.zip",
        [
            (f"{PARENT}/imu.txt", imu_rows.encode()),
            (
                f"{PARENT}/sequence_calibration.txt",
                (
                    f"mocap_time_scale {mocap_scale}\n"
                    f"mocap_time_anchor_seconds {mocap_anchor_seconds}\n"
                    f"mocap_time_offset_seconds {mocap_offset_seconds}\n"
                ).encode(),
            ),
        ],
    )


def _write_camera_imu_calibration(root: Path, *, lines: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return _write_plain_zip(root / "camera_imu_calib_radtan.zip", [("synthetic/chain.txt", lines.encode())])


def _calibration_binding(
    camera_key: str = "camera_from_imu",
    *,
    minimum_imu_samples: int = 5,
) -> source.CalibrationMemberBinding:
    return source.CalibrationMemberBinding(
        member="synthetic/chain.txt",
        camera_from_imu_key=camera_key,
        mocap_time_scale_key="mocap_time_scale",
        mocap_time_anchor_seconds_key="mocap_time_anchor_seconds",
        mocap_time_offset_seconds_key="mocap_time_offset_seconds",
        camera_timestamp_to_seconds="INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
        imu_timestamp_to_seconds="INTEGER_NANOSECONDS_TIMES_1E_MINUS_9",
        imu_clock_domain="CAMERA_CLOCK_NO_MOCAP_TRANSFORM",
        groundtruth_timestamp_unit="SECONDS",
        maximum_pose_bracket_seconds=Decimal("1.1"),
        imu_half_window_seconds=Decimal("0.05"),
        minimum_imu_samples=minimum_imu_samples,
    )


def _open_archive(path: Path, *, kind: str, parent_id: str, observer=None):
    binding = source.ArchiveBinding.from_manifest_row(
        _archive_row(path, kind=kind, parent_id=parent_id)
    )
    verified = source.verify_archive_binding(path.parent, binding)
    return source.preflight_archive(
        verified,
        budget=source.ArchiveBudget(max_members=1000, max_total_uncompressed_bytes=4 << 20),
        observer=observer,
    )


def _write_plain_zip(path: Path, members: Iterable[tuple[str, bytes]], *, compression=zipfile.ZIP_STORED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return path


def _mark_first_member_encrypted(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    local = payload.find(b"PK\x03\x04")
    central = payload.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    local_flags = struct.unpack_from("<H", payload, local + 6)[0]
    central_flags = struct.unpack_from("<H", payload, central + 8)[0]
    struct.pack_into("<H", payload, local + 6, local_flags | 0x1)
    struct.pack_into("<H", payload, central + 8, central_flags | 0x1)
    path.write_bytes(payload)


def _corrupt_member_crc(path: Path, member: str) -> None:
    with zipfile.ZipFile(path) as archive:
        local_offset = archive.getinfo(member).header_offset
    payload = bytearray(path.read_bytes())
    local_crc = struct.unpack_from("<I", payload, local_offset + 14)[0]
    struct.pack_into("<I", payload, local_offset + 14, local_crc ^ 0xFFFFFFFF)
    cursor = 0
    found = False
    while True:
        cursor = payload.find(b"PK\x01\x02", cursor)
        if cursor < 0:
            break
        name_length = struct.unpack_from("<H", payload, cursor + 28)[0]
        extra_length = struct.unpack_from("<H", payload, cursor + 30)[0]
        comment_length = struct.unpack_from("<H", payload, cursor + 32)[0]
        name_start = cursor + 46
        name = bytes(payload[name_start : name_start + name_length]).decode("utf-8")
        if name == member:
            central_crc = struct.unpack_from("<I", payload, cursor + 16)[0]
            struct.pack_into("<I", payload, cursor + 16, central_crc ^ 0xFFFFFFFF)
            found = True
            break
        cursor = name_start + name_length + extra_length + comment_length
    assert found
    path.write_bytes(payload)


def _error_code(callable_) -> str:
    with pytest.raises(ContractError) as raised:
        callable_()
    return raised.value.code


def test_binding_bytes_and_sha_fail_before_zip_enumeration(tmp_path: Path) -> None:
    path = _write_rgbd(tmp_path)
    wrong = _archive_row(path)
    wrong["sha256"] = "0" * 64
    with patch.object(source.zipfile, "ZipFile") as zip_constructor:
        assert _error_code(lambda: source.verify_archive_binding(tmp_path, wrong)) == "F2_ARCHIVE_SHA_MISMATCH"
        zip_constructor.assert_not_called()

    wrong = _archive_row(path)
    wrong["bytes"] = int(wrong["bytes"]) + 1
    with patch.object(source.zipfile, "ZipFile") as zip_constructor:
        assert _error_code(lambda: source.verify_archive_binding(tmp_path, wrong)) == "F2_ARCHIVE_BYTES_MISMATCH"
        zip_constructor.assert_not_called()


@pytest.mark.parametrize(
    "unsafe_name,expected",
    [
        ("../escape", "F2_ZIP_MEMBER_DIRECTORY_ESCAPE"),
        ("/absolute", "F2_ZIP_MEMBER_ABSOLUTE"),
        ("C:/absolute", "F2_ZIP_MEMBER_ABSOLUTE"),
        ("folder\\member", "F2_ZIP_MEMBER_BACKSLASH"),
        ("folder/./member", "F2_ZIP_MEMBER_DIRECTORY_ESCAPE"),
        ("folder//member", "F2_ZIP_MEMBER_DIRECTORY_ESCAPE"),
    ],
)
def test_preflight_rejects_unsafe_member_names(tmp_path: Path, unsafe_name: str, expected: str) -> None:
    path = _write_plain_zip(tmp_path / f"{PARENT}_rgbd.zip", [(unsafe_name, b"x")])
    if "\\" in unsafe_name:
        # ZipInfo normalizes the host separator while writing on Windows.  Make
        # the synthetic central/local names unsafe without changing length.
        payload = path.read_bytes().replace(unsafe_name.replace("\\", "/").encode(), unsafe_name.encode())
        path.write_bytes(payload)
    verified = source.verify_archive_binding(tmp_path, _archive_row(path))
    assert _error_code(
        lambda: source.preflight_archive(verified, budget=source.ArchiveBudget())
    ) == expected


def test_preflight_rejects_casefold_duplicate_encryption_and_budgets(tmp_path: Path) -> None:
    duplicate_root = tmp_path / "duplicate"
    duplicate = _write_plain_zip(
        duplicate_root / f"{PARENT}_rgbd.zip",
        [("root/Member.txt", b"a"), ("root/member.txt", b"b")],
    )
    verified = source.verify_archive_binding(duplicate_root, _archive_row(duplicate))
    assert _error_code(
        lambda: source.preflight_archive(verified, budget=source.ArchiveBudget())
    ) == "F2_ZIP_MEMBER_CASEFOLD_DUPLICATE"

    encrypted_root = tmp_path / "encrypted"
    encrypted = _write_plain_zip(encrypted_root / f"{PARENT}_rgbd.zip", [("root/member", b"x")])
    _mark_first_member_encrypted(encrypted)
    verified = source.verify_archive_binding(encrypted_root, _archive_row(encrypted))
    assert _error_code(
        lambda: source.preflight_archive(verified, budget=source.ArchiveBudget())
    ) == "F2_ZIP_MEMBER_ENCRYPTED"

    count_root = tmp_path / "count"
    counted = _write_plain_zip(
        count_root / f"{PARENT}_rgbd.zip",
        [("root/a", b"a"), ("root/b", b"b")],
    )
    verified = source.verify_archive_binding(count_root, _archive_row(counted))
    assert _error_code(
        lambda: source.preflight_archive(
            verified,
            budget=source.ArchiveBudget(max_members=1),
        )
    ) == "F2_ZIP_MEMBER_COUNT_BUDGET"

    size_root = tmp_path / "size"
    sized = _write_plain_zip(size_root / f"{PARENT}_rgbd.zip", [("root/a", b"a" * 100)])
    verified = source.verify_archive_binding(size_root, _archive_row(sized))
    assert _error_code(
        lambda: source.preflight_archive(
            verified,
            budget=source.ArchiveBudget(max_member_uncompressed_bytes=50),
        )
    ) == "F2_ZIP_MEMBER_SIZE_BUDGET"

    total_root = tmp_path / "total"
    totaled = _write_plain_zip(
        total_root / f"{PARENT}_rgbd.zip",
        [("root/a", b"a" * 40), ("root/b", b"b" * 40)],
    )
    verified = source.verify_archive_binding(total_root, _archive_row(totaled))
    assert _error_code(
        lambda: source.preflight_archive(
            verified,
            budget=source.ArchiveBudget(
                max_member_uncompressed_bytes=50,
                max_total_uncompressed_bytes=70,
            ),
        )
    ) == "F2_ZIP_TOTAL_SIZE_BUDGET"

    bomb_root = tmp_path / "bomb"
    bomb = _write_plain_zip(
        bomb_root / f"{PARENT}_rgbd.zip",
        [("root/a", b"0" * 100_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    verified = source.verify_archive_binding(bomb_root, _archive_row(bomb))
    assert _error_code(
        lambda: source.preflight_archive(
            verified,
            budget=source.ArchiveBudget(max_compression_ratio=2.0),
        )
    ) == "F2_ZIP_COMPRESSION_BOMB"


def test_member_crc_mismatch_is_rejected_when_phase_authorized_payload_is_read(tmp_path: Path) -> None:
    path = _write_rgbd(tmp_path)
    _corrupt_member_crc(path, f"{PARENT}/associated.txt")
    with _open_rgbd(path) as archive:
        assert _error_code(
            lambda: source.freeze_parent_roster(archive, parent_id=PARENT)
        ) == "F2_ZIP_MEMBER_READ_FAILED"


def test_imu_member_contract_is_bound_to_parent_root(tmp_path: Path) -> None:
    path = _write_plain_zip(
        tmp_path / f"{PARENT}_imu.zip",
        [(f"{PARENT}/imu.txt", b"1 0 0 0\n"), (f"{PARENT}/sequence_calibration.txt", b"1 0 0\n")],
    )
    binding = source.ArchiveBinding.from_manifest_row(_archive_row(path, kind="IMU_ARCHIVE"))
    verified = source.verify_archive_binding(tmp_path, binding)
    with source.preflight_archive(verified, budget=source.ArchiveBudget()) as archive:
        assert source.validate_eth3d_member_contract(archive) == f"{PARENT}/"


def test_roster_is_exact_disjoint_and_independent_of_associated_line_order(tmp_path: Path) -> None:
    ordered_path = _write_rgbd(tmp_path / "ordered")
    reversed_path = _write_rgbd(tmp_path / "reversed", associated_rows=reversed(_associated_rows()))
    events: list[source.ReadEvent] = []
    with _open_rgbd(ordered_path, observer=events.append) as ordered_archive:
        ordered = source.freeze_parent_roster(ordered_archive, parent_id=PARENT)
    with _open_rgbd(reversed_path) as reversed_archive:
        reversed_roster = source.freeze_parent_roster(reversed_archive, parent_id=PARENT)

    assert ordered.eligible_count == 24
    assert len(ordered.calibration) == len(ordered.score) == 12
    assert {row.frame_id for row in ordered.calibration}.isdisjoint(
        {row.frame_id for row in ordered.score}
    )
    assert ordered.as_dict() == reversed_roster.as_dict()
    assert all(row.rgb_timestamp == str(int(row.rgb_timestamp)) for row in ordered.calibration + ordered.score)
    first = ordered.calibration[0]
    expected = hashlib.sha256(
        (
            f"{package.PROTOCOL_ID}\n{PARENT}\n{first.rgb_timestamp}\n"
            f"{first.rgb_relpath}\n{first.depth_timestamp}\n{first.depth_relpath}"
        ).encode()
    ).hexdigest().upper()
    assert first.rank_token == expected == first.frame_id
    assert [(event.phase, event.purpose, event.member) for event in events] == [
        (
            source.SourcePhase.ROSTER_METADATA,
            "FREEZE_ASSOCIATED_ROSTER",
            f"{PARENT}/associated.txt",
        )
    ]


def test_roster_rejects_short_malformed_and_ambiguous_metadata(tmp_path: Path) -> None:
    short = _write_rgbd(tmp_path / "short", count=23)
    with _open_rgbd(short) as archive:
        assert _error_code(
            lambda: source.freeze_parent_roster(archive, parent_id=PARENT)
        ) == "F2_ROSTER_INSUFFICIENT_ELIGIBLE"

    nonfinite_rows = _associated_rows()
    nonfinite_rows[0] = "NaN rgb/000000.png 1000000000 depth/000000.png\n"
    nonfinite = _write_rgbd(tmp_path / "nonfinite", associated_rows=nonfinite_rows)
    with _open_rgbd(nonfinite) as archive:
        assert _error_code(
            lambda: source.freeze_parent_roster(archive, parent_id=PARENT)
        ) == "F2_ASSOCIATED_TIMESTAMP_TOKEN"

    mismatch_rows = _associated_rows()
    mismatch_rows[0] = "1000000000 rgb/000000.png 1000000000 depth/000001.png\n"
    mismatch = _write_rgbd(tmp_path / "mismatch", associated_rows=mismatch_rows)
    with _open_rgbd(mismatch) as archive:
        assert _error_code(
            lambda: source.freeze_parent_roster(archive, parent_id=PARENT)
        ) == "F2_ASSOCIATED_BASENAME_MISMATCH"


def test_prediction_and_source_arrays_are_read_only_and_phase_fenced(tmp_path: Path) -> None:
    path = _write_rgbd(tmp_path)
    events: list[source.ReadEvent] = []
    with _open_rgbd(path, observer=events.append) as archive:
        adapter = source.Eth3dParentSource(archive, parent_id=PARENT)
        calibration_id = adapter.roster.calibration[0].frame_id
        score_id = adapter.roster.score[0].frame_id

        before = len(events)
        prediction = adapter.read_prediction_input(
            score_id,
            phase=source.SourcePhase.RAW_SCORE_PREDICTION,
        )
        prediction_events = events[before:]
        assert prediction.rgb_hwc_u8.dtype == np.uint8
        assert prediction.rgb_hwc_u8.shape == (2, 3, 3)
        assert prediction.rgb_hwc_u8[0, 0, :2].tolist() == [30, 20]
        assert prediction.K.tolist() == [[100.0, 0.0, 1.0], [0.0, 110.0, 0.5], [0.0, 0.0, 1.0]]
        assert not prediction.rgb_hwc_u8.flags.writeable and not prediction.K.flags.writeable
        assert all(event.phase is source.SourcePhase.RAW_SCORE_PREDICTION for event in prediction_events)
        assert {event.purpose for event in prediction_events} == {"RAW_SCORE_RGB", "PINHOLE_INTRINSICS"}
        assert not any("depth/" in event.member for event in prediction_events)
        assert _error_code(
            lambda: archive.read_member_bytes(
                adapter.roster.score[0].depth_member,
                phase=source.SourcePhase.RAW_SCORE_PREDICTION,
                purpose="RAW_SCORE_RGB",
                max_bytes=archive.budget.max_member_uncompressed_bytes,
            )
        ) == "F2_SOURCE_MEMBER_PHASE_FIREWALL"

        assert _error_code(
            lambda: adapter.read_prediction_input(
                calibration_id,
                phase=source.SourcePhase.RAW_SCORE_PREDICTION,
            )
        ) == "F2_PREDICTION_FRAME_NOT_SCORE_ROLE"
        assert _error_code(
            lambda: adapter.read_prediction_input(
                score_id,
                phase=source.SourcePhase.SCORE_SOURCE,
            )
        ) == "F2_PREDICTION_PHASE_FORBIDDEN"
        assert _error_code(
            lambda: adapter.read_source_arrays(
                score_id,
                phase=source.SourcePhase.CALIBRATION_SOURCE,
            )
        ) == "F2_SOURCE_FRAME_ROLE_MISMATCH"

        calibration = adapter.read_source_arrays(
            calibration_id,
            phase=source.SourcePhase.CALIBRATION_SOURCE,
        )
        score = adapter.read_source_arrays(
            score_id,
            phase=source.SourcePhase.SCORE_SOURCE,
        )
        assert calibration.role == "CALIBRATION" and score.role == "SCORE"
        assert calibration.depth_m_hw.dtype == np.float32
        assert calibration.depth_known_hw.dtype == np.bool_
        assert calibration.depth_known_hw.tolist() == [[False, False, True], [True, False, True]]
        assert np.isnan(calibration.depth_m_hw[0, 0])
        assert float(calibration.depth_m_hw[1, 0]) == pytest.approx(6.0)
        assert not calibration.depth_m_hw.flags.writeable
        assert not calibration.depth_known_hw.flags.writeable
        with pytest.raises(ValueError):
            calibration.depth_m_hw[0, 0] = 1.0

    assert any(event.purpose == "CALIBRATION_SOURCE_DEPTH" for event in events)
    assert any(event.purpose == "SCORE_SOURCE_DEPTH" for event in events)


def _motion_fixture(tmp_path: Path):
    groundtruth = "".join(
        f"{index} {index} 0 0 0 0 0 1\n" for index in range(50)
    )
    associated = [
        f"{index * 1_000_000_000 + 500_000_000} rgb/{index:06d}.png "
        f"{index * 1_000_000_000 + 500_000_000} depth/{index:06d}.png\n"
        for index in range(24)
    ]
    rgbd = _write_rgbd(
        tmp_path / "rgbd",
        groundtruth=groundtruth,
        associated_rows=associated,
    )
    imu_rows: list[str] = []
    # Five samples are associated with every integer image timestamp under the
    # explicit 0.05-second window; acceleration is +Y in the IMU frame.
    for index in range(24):
        camera_timestamp = Decimal(index) + Decimal("0.5")
        for offset in ("-0.04", "-0.02", "0", "0.02", "0.04"):
            timestamp_seconds = camera_timestamp + Decimal(offset)
            timestamp_nanoseconds = int(timestamp_seconds / Decimal("1e-9"))
            imu_rows.append(f"{timestamp_nanoseconds} 0 0 0 0 9.81 0\n")
    imu = _write_imu(tmp_path / "imu", imu_rows="".join(imu_rows))
    # camera_from_imu rotates +Y(IMU) to +Z(camera).
    calibration = _write_camera_imu_calibration(
        tmp_path / "calibration",
        lines=(
            "camera_from_imu "
            "1 0 0 0 "
            "0 0 -1 0 "
            "0 1 0 0 "
            "0 0 0 1\n"
        ),
    )
    return rgbd, imu, calibration


def test_pose_and_gravity_are_role_fenced_interpolated_and_read_only(tmp_path: Path) -> None:
    rgbd_path, imu_path, calibration_path = _motion_fixture(tmp_path)
    events: list[source.ReadEvent] = []
    binding = _calibration_binding()
    with (
        _open_rgbd(rgbd_path, observer=events.append) as rgbd,
        _open_archive(
            imu_path,
            kind="IMU_ARCHIVE",
            parent_id=PARENT,
            observer=events.append,
        ) as imu,
        _open_archive(
            calibration_path,
            kind="CAMERA_IMU_CALIBRATION_ARCHIVE",
            parent_id="ALL_THREE_SESSIONS",
            observer=events.append,
        ) as calibration,
    ):
        adapter = source.Eth3dParentSource(rgbd, parent_id=PARENT)
        calibration_frame = adapter.roster.calibration[0]
        result = adapter.read_pose_and_gravity(
            calibration_frame.frame_id,
            phase=source.SourcePhase.CALIBRATION_SOURCE,
            imu_archive=imu,
            calibration_archive=calibration,
            calibration_binding=binding,
        )
        camera_seconds = float(
            source.camera_timestamp_nanoseconds_to_seconds(calibration_frame.rgb_timestamp)
        )
        expected_mocap_seconds = 2.0 * camera_seconds + 0.25
        assert result.role == "CALIBRATION"
        assert result.camera_timestamp_seconds == source.canonical_timestamp(str(camera_seconds))
        assert result.mocap_timestamp_seconds == source.canonical_timestamp(
            str(Decimal(2) * Decimal(str(camera_seconds)) + Decimal("0.25"))
        )
        assert result.camera_to_world.dtype == np.float64
        assert result.camera_to_world[:3, 3].tolist() == pytest.approx(
            [expected_mocap_seconds, 0.0, 0.0]
        )
        assert result.gravity_up_camera_xyz.tolist() == pytest.approx([0.0, 0.0, 1.0])
        assert result.imu_sample_count >= 5
        assert not result.camera_to_world.flags.writeable
        assert not result.gravity_up_camera_xyz.flags.writeable

        score_frame = adapter.roster.score[0]
        score_result = adapter.read_pose_and_gravity(
            score_frame.frame_id,
            phase=source.SourcePhase.SCORE_SOURCE,
            imu_archive=imu,
            calibration_archive=calibration,
            calibration_binding=binding,
        )
        assert score_result.role == "SCORE"
        assert score_result.imu_sample_count >= 5
        assert _error_code(
            lambda: adapter.read_pose_and_gravity(
                score_frame.frame_id,
                phase=source.SourcePhase.CALIBRATION_SOURCE,
                imu_archive=imu,
                calibration_archive=calibration,
                calibration_binding=binding,
            )
        ) == "F2_SOURCE_FRAME_ROLE_MISMATCH"
        assert _error_code(
            lambda: adapter.read_pose_and_gravity(
                score_frame.frame_id,
                phase=source.SourcePhase.RAW_SCORE_PREDICTION,
                imu_archive=imu,
                calibration_archive=calibration,
                calibration_binding=binding,
            )
        ) == "F2_POSE_GRAVITY_PHASE_FORBIDDEN"

    source_events = [
        event
        for event in events
        if event.phase in {source.SourcePhase.CALIBRATION_SOURCE, source.SourcePhase.SCORE_SOURCE}
    ]
    assert source_events
    assert {event.phase for event in source_events} == {
        source.SourcePhase.CALIBRATION_SOURCE,
        source.SourcePhase.SCORE_SOURCE,
    }
    assert {event.purpose for event in source_events} == {
        "CALIBRATION_CAMERA_TO_WORLD",
        "CALIBRATION_IMU_GRAVITY",
        "CALIBRATION_SEQUENCE_CALIBRATION",
        "CALIBRATION_CAMERA_IMU_CALIBRATION",
        "SCORE_CAMERA_TO_WORLD",
        "SCORE_IMU_GRAVITY",
        "SCORE_SEQUENCE_CALIBRATION",
        "SCORE_CAMERA_IMU_CALIBRATION",
    }


def test_pose_and_gravity_fail_closed_on_time_quaternion_imu_and_binding_ambiguity(tmp_path: Path) -> None:
    rgbd_path, imu_path, calibration_path = _motion_fixture(tmp_path)
    good_binding = _calibration_binding()
    with (
        _open_rgbd(rgbd_path) as rgbd,
        _open_archive(imu_path, kind="IMU_ARCHIVE", parent_id=PARENT) as imu,
        _open_archive(
            calibration_path,
            kind="CAMERA_IMU_CALIBRATION_ARCHIVE",
            parent_id="ALL_THREE_SESSIONS",
        ) as calibration,
    ):
        adapter = source.Eth3dParentSource(rgbd, parent_id=PARENT)
        frame = adapter.roster.calibration[0]
        assert _error_code(
            lambda: adapter.read_pose_and_gravity(
                frame.frame_id,
                phase=source.SourcePhase.CALIBRATION_SOURCE,
                imu_archive=imu,
                calibration_archive=calibration,
                calibration_binding=_calibration_binding("missing_key"),
            )
        ) == "F2_IMU_CALIBRATION_KEY_AMBIGUOUS_OR_MISSING"
        assert _error_code(
            lambda: adapter.read_pose_and_gravity(
                frame.frame_id,
                phase=source.SourcePhase.CALIBRATION_SOURCE,
                imu_archive=imu,
                calibration_archive=calibration,
                calibration_binding=_calibration_binding(minimum_imu_samples=1000),
            )
        ) == "F2_IMU_INSUFFICIENT_ASSOCIATED_SAMPLES"

    bad_groundtruth = _write_rgbd(
        tmp_path / "bad_pose",
        groundtruth="0 0 0 0 0 0 0 0\n50 50 0 0 0 0 0 1\n",
    )
    with (
        _open_rgbd(bad_groundtruth) as rgbd,
        _open_archive(imu_path, kind="IMU_ARCHIVE", parent_id=PARENT) as imu,
        _open_archive(
            calibration_path,
            kind="CAMERA_IMU_CALIBRATION_ARCHIVE",
            parent_id="ALL_THREE_SESSIONS",
        ) as calibration,
    ):
        adapter = source.Eth3dParentSource(rgbd, parent_id=PARENT)
        frame = adapter.roster.calibration[0]
        assert _error_code(
            lambda: adapter.read_pose_and_gravity(
                frame.frame_id,
                phase=source.SourcePhase.CALIBRATION_SOURCE,
                imu_archive=imu,
                calibration_archive=calibration,
                calibration_binding=good_binding,
            )
        ) == "F2_GROUNDTRUTH_QUATERNION"


def test_source_module_has_no_producer_metric_or_reducer_import() -> None:
    tree = ast.parse(Path(source.__file__).read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        forbidden in imported.lower()
        for imported in imports
        for forbidden in ("producer", "metrics", "reducer")
    )
