package com.linnan.blindassist.ustrf

import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.roundToInt

data class UstrfSyntheticTemporalGeometryReplayMetrics(
    val sequenceCount: Int,
    val admittedPairCount: Int,
    val rejectedPairCount: Int,
    val expectedStaticSequenceCount: Int,
    val matchedStaticSequenceCount: Int,
    val visibilityGapSequenceCount: Int,
    val falseDropSequenceCount: Int,
    val expectedDropSequenceCount: Int,
    val detectedDropSequenceCount: Int,
    val geometryEvidenceCount: Int
)

/**
 * JVM-only reader for the analytic benchmark's dependency-free CSV/TSV projection.
 *
 * This is intentionally a replay harness, not a device Adapter: it only accepts the benchmark's
 * pinned 64x48 registered-depth profile and feeds each pair through the same metric-depth and
 * temporal contracts used by the USTRF safety kernel.
 */
class UstrfSyntheticTemporalGeometryReplay {
    private val adapter = UstrfMetricDepthGeometryAdapter(
        UstrfMetricDepthGeometryAdapterConfig(sampleStridePx = 1, minimumDepthMeters = .20f, maximumDepthMeters = 5f)
    )
    private val dropProposer = UstrfGroundVisibilityDropProposer(
        UstrfGroundVisibilityDropConfig(sampleStridePx = 1, minimumExpectedForwardMeters = .20f)
    )

    fun replay(root: Path): UstrfSyntheticTemporalGeometryReplayMetrics {
        val lines = Files.readAllLines(root.resolve("kotlin_replay.tsv"))
        require(lines.size >= 2) { "benchmark replay manifest has no rows" }
        require(lines.first().split('\t') == HEADER) { "unexpected benchmark replay header" }
        var admitted = 0
        var rejected = 0
        var expectedStatic = 0
        var matchedStatic = 0
        var gaps = 0
        var falseDrops = 0
        var expectedDrops = 0
        var detectedDrops = 0
        var evidenceCount = 0
        lines.drop(1).forEach { line ->
            val values = line.split('\t')
            require(values.size == HEADER.size) { "malformed benchmark replay row" }
            val row = HEADER.indices.associate { HEADER[it] to values[it] }
            val first = frame(row, 0)
            val second = frame(row, 1)
            val firstDepth = depth(root, row.getValue("frame0_depth_csv_path"), first)
            val secondDepth = depth(root, row.getValue("frame1_depth_csv_path"), second)
            val firstResult = adapter.project(admission(first), firstDepth, intrinsics(), extrinsics(), ground(first), first.capturedAtNs)
            val secondResult = adapter.project(admission(second), secondDepth, intrinsics(), extrinsics(), ground(second), second.capturedAtNs)
            if (firstResult !is UstrfMetricDepthGeometryAdapterResult.Available || secondResult !is UstrfMetricDepthGeometryAdapterResult.Available) {
                rejected += 1
                return@forEach
            }
            admitted += 1
            evidenceCount += firstResult.admittedEvidenceCount + secondResult.admittedEvidenceCount
            val proposedDrops = listOf(
                dropProposer.propose(admission(first), firstDepth, intrinsics(), extrinsics(), ground(first), first.capturedAtNs),
                dropProposer.propose(admission(second), secondDepth, intrinsics(), extrinsics(), ground(second), second.capturedAtNs)
            ).filterIsInstance<UstrfGroundVisibilityDropProposal.Available>().sumOf { it.evidence.size }
            val temporal = UstrfTemporalGeometryConsistency().compare(
                firstResult.packet,
                secondResult.packet,
                UstrfVerifiedPoseDelta(
                    first, second,
                    row.getValue("pose_forward_meters").toFloat(),
                    row.getValue("pose_lateral_meters").toFloat(),
                    row.getValue("pose_yaw_radians").toFloat(),
                    row.getValue("pose_verified").toBooleanStrict()
                ),
                second.capturedAtNs
            )
            if (row.getValue("expected_temporal_static_match").toBooleanStrict()) {
                expectedStatic += 1
                val expectedKind = when (row.getValue("target_kind")) {
                    "lower_body" -> UstrfGeometryKind.OCCUPIED
                    "head" -> UstrfGeometryKind.HEAD_OBSTACLE
                    else -> error("static match row must name a body/head target")
                }
                if (temporal is UstrfTemporalGeometryConsistencyResult.Available && temporal.matches.any { it.current.kind == expectedKind }) {
                    matchedStatic += 1
                }
            }
            if (row.getValue("expected_must_not_emit_drop").toBooleanStrict()) {
                gaps += 1
                if (firstResult.packet.evidence.any { it.kind == UstrfGeometryKind.DROP } || secondResult.packet.evidence.any { it.kind == UstrfGeometryKind.DROP }) {
                    falseDrops += 1
                }
            }
            if (row.getValue("expected_must_emit_drop").toBooleanStrict()) {
                expectedDrops += 1
                if (proposedDrops > 0) detectedDrops += 1
            }
        }
        return UstrfSyntheticTemporalGeometryReplayMetrics(
            lines.size - 1, admitted, rejected, expectedStatic, matchedStatic, gaps, falseDrops, expectedDrops, detectedDrops, evidenceCount
        )
    }

    private fun frame(row: Map<String, String>, index: Int): UstrfFrameStamp = UstrfFrameStamp(
        row.getValue("frame${index}_id").toLong(), row.getValue("frame${index}_captured_at_ns").toLong(), CAMERA_FRAME
    )

    private fun depth(root: Path, relativePath: String, frame: UstrfFrameStamp): UstrfRegisteredMetricDepthImage {
        val rows = Files.readAllLines(root.resolve(relativePath))
        require(rows.size == HEIGHT) { "depth csv height mismatch" }
        val values = rows.flatMap { row -> row.split(',').map(String::toFloat) }
        require(values.size == WIDTH * HEIGHT) { "depth csv width mismatch" }
        return UstrfRegisteredMetricDepthImage(
            frame, WIDTH, HEIGHT, DEPTH_FRAME, TRANSFORM_ID,
            values.map { (it * 1_000f).roundToInt().coerceAtLeast(0) }.toIntArray(),
            values.map { if (it > 0f) 1f else 0f }.toFloatArray(),
            frame.capturedAtNs + VALIDITY_NS
        )
    }

    private fun admission(frame: UstrfFrameStamp) = UstrfMetricGeometryProjectionAdmission.Available(
        frame,
        DEPTH_FRAME,
        CAMERA_FRAME,
        BODY_FRAME,
        "synthetic-mount-v1",
        "a".repeat(64),
        TRANSFORM_ID,
        frame.capturedAtNs + VALIDITY_NS
    )

    private fun intrinsics() = UstrfCameraIntrinsicsReceipt(
        CAMERA_FRAME, "synthetic-v1", WIDTH, HEIGHT, 52f, 52f, 31.5f, 23.5f,
        0L, Long.MAX_VALUE, 1f, true
    )

    private fun extrinsics() = UstrfCameraBodyFullExtrinsicsReceipt(
        CAMERA_FRAME, BODY_FRAME, UstrfVector3(0f, 1.5f, 0f), floatArrayOf(0f, 0f, 0f, 1f),
        "synthetic-mount-v1", 0L, Long.MAX_VALUE, 1f, true
    )

    private fun ground(frame: UstrfFrameStamp) = UstrfVerifiedGroundPlaneReceipt(
        frame, BODY_FRAME, UstrfVector3(0f, 1f, 0f), 0f, 1f, true, frame.capturedAtNs + VALIDITY_NS
    )

    companion object {
        private const val WIDTH = 64
        private const val HEIGHT = 48
        private const val CAMERA_FRAME = "synthetic-camera"
        private const val BODY_FRAME = "synthetic-body"
        private const val DEPTH_FRAME = "synthetic-registered-depth"
        private const val TRANSFORM_ID = "synthetic-depth-to-camera-v1"
        private const val VALIDITY_NS = 1_000_000_000L
        private val HEADER = listOf(
            "sequence_id", "target_kind", "target_lateral_m", "target_forward_m", "target_height_m",
            "expected_temporal_static_match", "expected_must_not_emit_drop", "expected_must_emit_drop",
            "frame0_id", "frame0_captured_at_ns", "frame0_camera_forward_m", "frame0_depth_csv_path",
            "frame1_id", "frame1_captured_at_ns", "frame1_camera_forward_m", "frame1_depth_csv_path",
            "pose_forward_meters", "pose_lateral_meters", "pose_yaw_radians", "pose_verified"
        )
    }
}
