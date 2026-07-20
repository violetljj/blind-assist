package com.linnan.blindassist.ustrf

import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs

data class UstrfSyntheticDynamicTtcReplayMetrics(
    val sequenceCount: Int,
    val admittedSequenceCount: Int,
    val rejectedSequenceCount: Int,
    val expectedTtcCount: Int,
    val measuredTtcCount: Int,
    val maximumTtcErrorMs: Long,
    val expectedCollisionCount: Int,
    val detectedCollisionCount: Int
)

/**
 * Dependency-free replay of analytic body-track pairs.  The manifest is deliberately restricted
 * to a theory benchmark: it tests exact pair binding, ego compensation and TTC against known
 * values but supplies neither detector quality nor a production safety authorization.
 */
class UstrfSyntheticDynamicTtcReplay(
    private val promoter: UstrfEgoCompensatedMotionPromoter = UstrfEgoCompensatedMotionPromoter(),
    private val ttcEstimator: UstrfTtcEstimator = UstrfTtcEstimator(),
    private val collisionRadiusMeters: Float = .75f
) {
    fun replay(root: Path): UstrfSyntheticDynamicTtcReplayMetrics {
        val lines = Files.readAllLines(root.resolve("kotlin_dynamic_ttc_replay.tsv"))
        require(lines.size >= 2) { "dynamic TTC replay manifest has no rows" }
        val header = lines.first().split('\t')
        require(header == HEADER) { "unexpected dynamic TTC replay header" }
        var admitted = 0; var rejected = 0; var expectedTtc = 0; var measuredTtc = 0
        var maxTtcError = 0L; var expectedCollisions = 0; var detectedCollisions = 0
        lines.drop(1).forEach { line ->
            val values = line.split('\t')
            require(values.size == HEADER.size) { "malformed dynamic TTC replay row" }
            val row = HEADER.indices.associate { HEADER[it] to values[it] }
            val previous = frame(row, "previous")
            val current = frame(row, "current")
            val pair = UstrfDynamicTrackPair(
                row.getValue("track_id"), previous, current,
                UstrfVector2(row.float("previous_forward_m"), row.float("previous_lateral_m")),
                UstrfVector2(row.float("current_forward_m"), row.float("current_lateral_m")),
                row.float("track_confidence"), "synthetic-dynamic-manifest", current.capturedAtNs + VALIDITY_NS
            )
            val resolution = promoter.promote(
                pair,
                UstrfVerifiedPoseDelta(previous, current, row.float("pose_forward_m"), row.float("pose_lateral_m"), row.float("pose_yaw_rad"), row.bool("pose_verified")),
                current.capturedAtNs
            )
            if (!row.bool("expected_admitted")) {
                require(resolution is UstrfEgoCompensatedMotionResolution.Unavailable) { "rejected row promoted: ${row.getValue("sequence_id")}" }
                rejected += 1
                return@forEach
            }
            val available = resolution as? UstrfEgoCompensatedMotionResolution.Available
                ?: error("admitted row was rejected: ${row.getValue("sequence_id")}")
            admitted += 1
            val expectedVelocity = UstrfVector2(row.float("expected_velocity_forward_mps"), row.float("expected_velocity_lateral_mps"))
            require(abs(available.evidence.motion.relativeVelocityMetersPerSecond.forward - expectedVelocity.forward) <= .0001f)
            require(abs(available.evidence.motion.relativeVelocityMetersPerSecond.lateral - expectedVelocity.lateral) <= .0001f)
            val expectedTtcRaw = row.getValue("expected_ttc_ms")
            val estimate = ttcEstimator.estimate(available.evidence.motion, current.capturedAtNs)
            if (expectedTtcRaw.isBlank()) {
                require(estimate == null) { "static row unexpectedly has TTC: ${row.getValue("sequence_id")}" }
            } else {
                val expectedMs = expectedTtcRaw.toLong()
                val measured = requireNotNull(estimate) { "moving row has no TTC: ${row.getValue("sequence_id")}" }
                expectedTtc += 1; measuredTtc += 1
                maxTtcError = maxOf(maxTtcError, abs(measured.timeToClosestApproachMs - expectedMs))
                val expectedCollision = row.bool("expected_collision")
                if (expectedCollision) expectedCollisions += 1
                if (measured.closestDistanceMeters <= collisionRadiusMeters) detectedCollisions += 1
                require((measured.closestDistanceMeters <= collisionRadiusMeters) == expectedCollision) { "collision mismatch: ${row.getValue("sequence_id")}" }
            }
        }
        return UstrfSyntheticDynamicTtcReplayMetrics(lines.size - 1, admitted, rejected, expectedTtc, measuredTtc, maxTtcError, expectedCollisions, detectedCollisions)
    }

    private fun frame(row: Map<String, String>, prefix: String) = UstrfFrameStamp(
        row.getValue("${prefix}_frame_id").toLong(), row.getValue("${prefix}_captured_at_ns").toLong(), BODY_FRAME
    )
    private fun Map<String, String>.float(key: String) = getValue(key).toFloat()
    private fun Map<String, String>.bool(key: String) = getValue(key).toBooleanStrict()

    companion object {
        private const val BODY_FRAME = "synthetic-body"
        private const val VALIDITY_NS = 1_000_000_000L
        private val HEADER = listOf(
            "sequence_id", "track_id", "previous_frame_id", "previous_captured_at_ns", "current_frame_id", "current_captured_at_ns",
            "previous_forward_m", "previous_lateral_m", "current_forward_m", "current_lateral_m", "pose_forward_m", "pose_lateral_m", "pose_yaw_rad",
            "pose_verified", "track_confidence", "expected_admitted", "expected_velocity_forward_mps", "expected_velocity_lateral_mps",
            "expected_ttc_ms", "expected_closest_distance_m", "expected_collision"
        )
    }
}
