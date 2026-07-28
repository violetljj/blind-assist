from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any

import mvsec_target_chunks as parser_core

from mvsec_identity_evidence_r2 import (
    ImageMetadataLedger,
    StageLedger,
    classify_exception,
    pairing_diagnostics,
    sha256_file,
    write_pairing_diagnostic_ledger,
)


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def inspect_capture_r2(
    core: Any,
    *,
    repo: Path,
    namespace: Path,
    capture: dict[str, Any],
    geometry_source: dict[str, Any],
) -> dict[str, Any]:
    capture_root = namespace / capture["capture_id"]
    capture_root.mkdir(parents=True, exist_ok=True)
    stage = StageLedger(capture_root / "stage_ledger.jsonl")
    metadata = ImageMetadataLedger(
        capture_root / "image_metadata_ledger.jsonl"
    )
    active_stage = "CAPTURE"
    parsed_chunk_count = 0
    parsed_image_count = 0
    metadata_binding = None
    pairing_binding = None
    chunk_plan_binding = None

    def begin(name: str, details: dict[str, Any] | None = None) -> None:
        nonlocal active_stage
        active_stage = name
        stage.append(stage=name, status="BEGIN", details=details)

    def passed(name: str, details: dict[str, Any] | None = None) -> None:
        stage.append(stage=name, status="PASS", details=details)

    try:
        stage.append(
            stage="CAPTURE",
            status="BEGIN",
            details={"capture_id": capture["capture_id"]},
        )
        ledger = core.RangeLedger(capture_root / "range_ledger.jsonl")
        reader = core.ExactRangeReader(
            url=capture["data_bag"]["url"],
            expected_bytes=capture["data_bag"]["bytes"],
            expected_etag=capture["data_bag"]["etag"],
            maximum_bytes=capture["transport"]["maximum_remote_bytes"],
            ledger=ledger,
        )

        begin("HEAD_IDENTITY")
        head = reader.head()
        passed(
            "HEAD_IDENTITY",
            {
                "content_length": head["content_length"],
                "etag": head["etag"],
                "accept_ranges": head["accept_ranges"],
            },
        )

        def fetch(start: int, end: int, label: str) -> bytes:
            stage_name = f"RANGE_{label.upper().replace('-', '_')}"
            begin(
                stage_name,
                {
                    "start": start,
                    "end": end,
                    "requested_bytes": end - start + 1,
                    "range_ledger_rows_before": ledger.rows,
                    "range_ledger_bytes_before": ledger.bytes,
                },
            )
            body = reader.fetch(start, end, label)
            passed(
                stage_name,
                {
                    "body_bytes": len(body),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "range_ledger_rows_after": ledger.rows,
                    "range_ledger_bytes_after": ledger.bytes,
                },
            )
            return body

        opening = fetch(0, 8191, "opening")
        begin("OPENING_PARSE")
        if opening[: len(parser_core.MAGIC)] != parser_core.MAGIC:
            raise core.IdentityFailure("ROSBAG_MAGIC")
        bag_header, _, opening_cursor = parser_core.record(
            opening,
            len(parser_core.MAGIC),
        )
        index_pos = parser_core.u64(bag_header["index_pos"])
        connection_count = parser_core.u32(bag_header["conn_count"])
        chunk_count = parser_core.u32(bag_header["chunk_count"])
        passed(
            "OPENING_PARSE",
            {
                "record_end_cursor": opening_cursor,
                "index_pos": index_pos,
                "connection_count": connection_count,
                "chunk_count": chunk_count,
            },
        )

        final_index_bytes = capture["data_bag"]["bytes"] - index_pos
        if (
            index_pos <= 0
            or final_index_bytes <= 0
            or final_index_bytes
            > capture["transport"]["maximum_index_bytes"]
        ):
            raise core.IdentityFailure("FINAL_INDEX_BOUND")
        final_index = fetch(
            index_pos,
            capture["data_bag"]["bytes"] - 1,
            "final-index",
        )
        begin("FINAL_INDEX_PARSE")
        connections, chunks = parser_core.parse_final_index(
            final_index,
            expected_connection_count=connection_count,
            expected_chunk_count=chunk_count,
        )
        passed(
            "FINAL_INDEX_PARSE",
            {
                "connection_count": len(connections),
                "chunk_count": len(chunks),
            },
        )

        begin("CONNECTION_SELECT")
        matching = [
            row
            for row in connections.values()
            if row["topic"] == core.IMAGE_TOPIC
        ]
        if (
            len(matching) != 1
            or matching[0]["type"] != core.IMAGE_TYPE
            or matching[0]["md5sum"] != core.IMAGE_MD5
        ):
            raise core.IdentityFailure("IMAGE_CONNECTION_IDENTITY")
        connection = matching[0]
        passed(
            "CONNECTION_SELECT",
            {
                "connection_id": connection["connection_id"],
                "topic": connection["topic"],
                "type": connection["type"],
                "md5sum": connection["md5sum"],
            },
        )

        begin("TARGET_CHUNK_PLAN")
        selected_chunks = parser_core.select_target_chunks(
            chunks,
            connection_id=connection["connection_id"],
            window_start_ns=capture["window"]["start_ns"],
            window_end_ns=capture["window"]["end_ns"],
        )
        plan_path = capture_root / "target_chunk_plan.json"
        _write_json_exclusive(
            plan_path,
            {
                "schema_version": "rcle_mvsec_target_chunk_plan_r2.v1",
                "capture_id": capture["capture_id"],
                "window": capture["window"],
                "connection_id": connection["connection_id"],
                "selected_chunks": selected_chunks,
                "selection_rule": (
                    "OVERLAPPING_IMAGE_BEARING_PLUS_NEAREST_PREVIOUS_AND_NEXT"
                ),
            },
        )
        chunk_plan_binding = {
            "path": plan_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(plan_path),
            "rows": len(selected_chunks),
        }
        passed("TARGET_CHUNK_PLAN", chunk_plan_binding)

        messages = []
        chunk_rows = []
        for plan_ordinal, selected in enumerate(selected_chunks):
            position = selected["chunk_pos"]
            prefix = fetch(
                position,
                position + 3,
                "chunk-header-length",
            )
            begin(
                "CHUNK_HEADER_LENGTH_PARSE",
                {
                    "plan_ordinal": plan_ordinal,
                    "chunk_index": selected["chunk_index"],
                    "chunk_pos": position,
                },
            )
            header_size = struct.unpack("<I", prefix)[0]
            passed(
                "CHUNK_HEADER_LENGTH_PARSE",
                {"header_size": header_size},
            )
            header_and_size = fetch(
                position + 4,
                position + 4 + header_size + 3,
                "chunk-header",
            )
            begin(
                "CHUNK_HEADER_PARSE",
                {
                    "plan_ordinal": plan_ordinal,
                    "chunk_index": selected["chunk_index"],
                    "chunk_pos": position,
                },
            )
            chunk_header = parser_core.fields(
                header_and_size[:header_size]
            )
            compressed_size = struct.unpack_from(
                "<I",
                header_and_size,
                header_size,
            )[0]
            if chunk_header["op"][0] != parser_core.OP_CHUNK:
                raise core.IdentityFailure("CHUNK_POSITION")
            compression = chunk_header["compression"].decode("ascii")
            declared_uncompressed_bytes = parser_core.u32(
                chunk_header["size"]
            )
            passed(
                "CHUNK_HEADER_PARSE",
                {
                    "compression": compression,
                    "compressed_bytes": compressed_size,
                    "declared_uncompressed_bytes": (
                        declared_uncompressed_bytes
                    ),
                },
            )
            data_start = position + 4 + header_size + 4
            compressed = fetch(
                data_start,
                data_start + compressed_size - 1,
                "chunk-payload",
            )
            begin(
                "CHUNK_DECOMPRESS",
                {
                    "plan_ordinal": plan_ordinal,
                    "chunk_index": selected["chunk_index"],
                    "chunk_pos": position,
                    "compression": compression,
                    "compressed_bytes": compressed_size,
                },
            )
            uncompressed = parser_core.decompress_chunk(
                compression=compression,
                compressed=compressed,
                declared_uncompressed_bytes=declared_uncompressed_bytes,
            )
            uncompressed_sha256 = hashlib.sha256(uncompressed).hexdigest()
            passed(
                "CHUNK_DECOMPRESS",
                {
                    "uncompressed_bytes": len(uncompressed),
                    "uncompressed_sha256": uncompressed_sha256,
                },
            )

            begin(
                "RECORD_SCAN",
                {
                    "plan_ordinal": plan_ordinal,
                    "chunk_index": selected["chunk_index"],
                    "chunk_pos": position,
                    "connection_id": connection["connection_id"],
                    "uncompressed_bytes": len(uncompressed),
                },
            )
            chunk_messages = []
            cursor = 0
            record_ordinal = 0
            operation_counts: dict[int, int] = {}
            while cursor < len(uncompressed):
                record_offset = cursor
                try:
                    header, data, cursor = parser_core.record(
                        uncompressed,
                        cursor,
                    )
                    operation = header["op"][0]
                    operation_counts[operation] = (
                        operation_counts.get(operation, 0) + 1
                    )
                    if operation == parser_core.OP_MSG_DATA:
                        record_connection_id = parser_core.u32(
                            header["conn"]
                        )
                        if (
                            record_connection_id
                            == connection["connection_id"]
                        ):
                            message = parser_core.parse_sensor_image(
                                data,
                                bag_timestamp_ns=parser_core.time_ns(
                                    header["time"]
                                ),
                            )
                            metadata.append(
                                message,
                                chunk_index=selected["chunk_index"],
                                chunk_pos=position,
                                record_ordinal=record_ordinal,
                                record_offset=record_offset,
                                serialized_bytes=len(data),
                            )
                            chunk_messages.append(message)
                            parsed_image_count += 1
                except BaseException as error:
                    stage.append(
                        stage="RECORD_SCAN",
                        status="FAIL",
                        details={
                            "plan_ordinal": plan_ordinal,
                            "chunk_index": selected["chunk_index"],
                            "chunk_pos": position,
                            "record_ordinal": record_ordinal,
                            "record_offset": record_offset,
                            "terminal_cursor": cursor,
                            "parsed_image_count": parsed_image_count,
                            **classify_exception(error),
                        },
                    )
                    raise
                record_ordinal += 1
            parsed_chunk_count += 1
            timestamps = [
                row.header_timestamp_ns for row in chunk_messages
            ]
            passed(
                "RECORD_SCAN",
                {
                    "record_count": record_ordinal,
                    "terminal_cursor": cursor,
                    "image_message_count": len(chunk_messages),
                    "timestamp_min_ns": min(timestamps) if timestamps else None,
                    "timestamp_max_ns": max(timestamps) if timestamps else None,
                    "operation_counts": {
                        str(key): value
                        for key, value in sorted(operation_counts.items())
                    },
                },
            )
            messages.extend(chunk_messages)
            chunk_rows.append(
                {
                    "chunk_index": selected["chunk_index"],
                    "chunk_pos": position,
                    "start_ns": selected["start_ns"],
                    "end_ns": selected["end_ns"],
                    "compression": compression,
                    "compressed_bytes": compressed_size,
                    "compressed_sha256": hashlib.sha256(
                        compressed
                    ).hexdigest(),
                    "uncompressed_bytes": len(uncompressed),
                    "uncompressed_sha256": uncompressed_sha256,
                    "record_count": record_ordinal,
                    "image_message_count": len(chunk_messages),
                }
            )

        begin("MESSAGE_DEDUP_ORDER")
        unique = {
            (message.header_timestamp_ns, message.serialized_sha256): message
            for message in messages
        }
        ordered = [unique[key] for key in sorted(unique)]
        passed(
            "MESSAGE_DEDUP_ORDER",
            {
                "raw_message_count": len(messages),
                "unique_message_count": len(ordered),
                "exact_duplicate_count": len(messages) - len(ordered),
            },
        )

        begin("GEOMETRY_TIMESTAMPS")
        geometry = core.geometry_timestamps(
            geometry_source,
            start_ns=capture["window"]["start_ns"],
            end_ns=capture["window"]["end_ns"],
        )
        expected_geometry_count = capture["window"]["geometry_frame_count"]
        if len(geometry) != expected_geometry_count:
            raise core.IdentityFailure("GEOMETRY_FRAME_COUNT_IDENTITY")
        passed(
            "GEOMETRY_TIMESTAMPS",
            {
                "count": len(geometry),
                "minimum_ns": min(geometry),
                "maximum_ns": max(geometry),
            },
        )

        metadata_binding = metadata.binding(repo)
        begin("PAIRING_PREFLIGHT")
        pair_summary = pairing_diagnostics(
            ordered,
            geometry_timestamps_ns=geometry,
            maximum_delta_ns=capture["pairing"]["maximum_abs_delta_ns"],
        )
        pairing_path = capture_root / "pairing_diagnostic_ledger.jsonl"
        pairing_binding = write_pairing_diagnostic_ledger(
            pairing_path,
            ordered,
            geometry_timestamps_ns=geometry,
            maximum_delta_ns=capture["pairing"]["maximum_abs_delta_ns"],
        )
        pairing_binding["path"] = pairing_path.relative_to(repo).as_posix()
        passed(
            "PAIRING_PREFLIGHT",
            {
                **pair_summary,
                "image_metadata_ledger": metadata_binding,
                "pairing_diagnostic_ledger": pairing_binding,
            },
        )

        begin("PAIRING")
        paired = parser_core.pair_window(
            ordered,
            geometry_timestamps_ns=geometry,
            maximum_delta_ns=capture["pairing"]["maximum_abs_delta_ns"],
        )
        passed(
            "PAIRING",
            {
                "selected_count": len(paired["selected"]),
                "retained_count": len(paired["retained"]),
                "maximum_abs_delta_ns": paired["maximum_abs_delta_ns"],
            },
        )

        begin("LAYOUT_IDENTITY")
        retained = paired["retained"]
        frame_ids = {message.frame_id for message in retained}
        encodings = {message.encoding for message in retained}
        layouts = {
            (
                message.height,
                message.width,
                message.step,
                message.is_bigendian,
                len(message.payload),
            )
            for message in retained
        }
        if (
            len(frame_ids) != 1
            or next(iter(frame_ids)) == ""
            or encodings != {"mono8"}
            or layouts != {(260, 346, 346, 0, 89_960)}
        ):
            raise core.IdentityFailure("IMAGE_LAYOUT_IDENTITY")
        passed(
            "LAYOUT_IDENTITY",
            {
                "frame_id": next(iter(frame_ids)),
                "encoding": "mono8",
                "layout": [260, 346, 346, 0, 89_960],
            },
        )

        begin("PAYLOAD_MATERIALIZATION")
        frame_root = capture_root / "frames"
        frame_root.mkdir(parents=True, exist_ok=False)
        rows = []
        selected_set = {
            message.header_timestamp_ns for message in paired["selected"]
        }
        for message in retained:
            if (
                message.header_timestamp_ns
                == paired["before"].header_timestamp_ns
            ):
                role = "GUARD_BEFORE"
            elif (
                message.header_timestamp_ns
                == paired["after"].header_timestamp_ns
            ):
                role = "GUARD_AFTER"
            elif message.header_timestamp_ns in selected_set:
                role = "SELECTED"
            else:
                raise AssertionError("UNREACHABLE_FRAME_ROLE")
            relative = (
                Path(capture["capture_id"])
                / "frames"
                / f"{message.header_timestamp_ns}.mono8"
            )
            output = namespace / relative
            descriptor = os.open(
                output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(message.payload)
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(
                {
                    "role": role,
                    "header_timestamp_ns": message.header_timestamp_ns,
                    "bag_timestamp_ns": message.bag_timestamp_ns,
                    "sequence": message.sequence,
                    "frame_id": message.frame_id,
                    "encoding": message.encoding,
                    "height": message.height,
                    "width": message.width,
                    "step": message.step,
                    "serialized_sha256": message.serialized_sha256,
                    "payload_path": relative.as_posix(),
                    "payload_sha256": message.payload_sha256,
                    "payload_bytes": len(message.payload),
                }
            )
        ledger_path = capture_root / "frame_ledger.jsonl"
        descriptor = os.open(
            ledger_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        passed(
            "PAYLOAD_MATERIALIZATION",
            {
                "materialized_payload_files": len(rows),
                "materialized_payload_bytes": sum(
                    row["payload_bytes"] for row in rows
                ),
                "frame_ledger_sha256": sha256_file(ledger_path),
            },
        )
        stage.append(stage="CAPTURE", status="PASS")
        stage_binding = stage.binding(repo)
        return {
            "capture_id": capture["capture_id"],
            "window": capture["window"],
            "head": head,
            "bag": {
                "index_pos": index_pos,
                "connection_count": connection_count,
                "chunk_count": chunk_count,
                "final_index_bytes": final_index_bytes,
            },
            "image_connection": connection,
            "selected_chunks": chunk_rows,
            "geometry_timestamp_count": len(geometry),
            "selected_frame_count": len(paired["selected"]),
            "guard_frame_count": 2,
            "maximum_abs_pair_delta_ns": paired[
                "maximum_abs_delta_ns"
            ],
            "frame_id": next(iter(frame_ids)),
            "encoding": "mono8",
            "layout": {
                "height": 260,
                "width": 346,
                "step": 346,
                "payload_bytes": 89_960,
            },
            "frame_ledger": {
                "path": ledger_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
            },
            "range_ledger": ledger.binding(repo),
            "stage_ledger": stage_binding,
            "image_metadata_ledger": metadata_binding,
            "pairing_diagnostic_ledger": pairing_binding,
            "target_chunk_plan": chunk_plan_binding,
            "materialized_payload_files": len(rows),
            "materialized_payload_bytes": sum(
                row["payload_bytes"] for row in rows
            ),
            "image_decode_calls": 0,
            "rectification_calls": 0,
            "rgb_algorithm_calls": 0,
        }
    except BaseException as error:
        if metadata_binding is None:
            metadata_binding = metadata.binding(repo)
        stage.append(
            stage=active_stage,
            status="FAIL",
            details={
                **classify_exception(error),
                "parsed_chunk_count": parsed_chunk_count,
                "parsed_image_count": parsed_image_count,
                "image_metadata_ledger": metadata_binding,
                "pairing_diagnostic_ledger": pairing_binding,
                "target_chunk_plan": chunk_plan_binding,
            },
        )
        stage.append(
            stage="CAPTURE",
            status="FAIL",
            details=classify_exception(error),
        )
        stage.close()
        raise
