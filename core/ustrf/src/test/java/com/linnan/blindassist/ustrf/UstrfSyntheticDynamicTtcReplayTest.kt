package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path

class UstrfSyntheticDynamicTtcReplayTest {
    @Test
    fun replayConsumesBoundDynamicTrackManifestAndChecksTtcAndCollisionTruth() {
        val root = Files.createTempDirectory("ustrf-dynamic-ttc")
        Files.writeString(root.resolve("kotlin_dynamic_ttc_replay.tsv"), HEADER + "\n" + listOf(
            "oncoming\ttarget-a\t0\t1000000000\t1\t2000000000\t3.5\t0.0\t2.0\t0.0\t0.5\t0.0\t0.0\ttrue\t1.0\ttrue\t-1.0\t0.0\t2000\t0.0\ttrue",
            "static\ttarget-b\t2\t3000000000\t3\t4000000000\t3.5\t0.0\t3.0\t0.0\t0.5\t0.0\t0.0\ttrue\t1.0\ttrue\t0.0\t0.0\t\t\tfalse",
            "pose-rejected\ttarget-c\t4\t5000000000\t5\t6000000000\t3.5\t0.0\t2.0\t0.0\t0.5\t0.0\t0.0\tfalse\t1.0\tfalse\t-1.0\t0.0\t\t\t"
        ).joinToString("\n") + "\n")
        val metrics = UstrfSyntheticDynamicTtcReplay().replay(root)
        assertEquals(3, metrics.sequenceCount)
        assertEquals(2, metrics.admittedSequenceCount)
        assertEquals(1, metrics.rejectedSequenceCount)
        assertEquals(1, metrics.expectedTtcCount)
        assertEquals(0L, metrics.maximumTtcErrorMs)
        assertEquals(1, metrics.expectedCollisionCount)
        assertEquals(1, metrics.detectedCollisionCount)
    }

    @Test
    fun generatedGpuAuditedBundleReplaysWhenExplicitlyProvided() {
        val root = System.getenv("USTRF_SYNTHETIC_DYNAMIC_TTC_REPLAY_ROOT") ?: return
        assumeTrue(Path.of(root).resolve("kotlin_dynamic_ttc_replay.tsv").toFile().isFile)
        val metrics = UstrfSyntheticDynamicTtcReplay().replay(Path.of(root))
        assertEquals(9, metrics.sequenceCount)
        assertEquals(7, metrics.admittedSequenceCount)
        assertEquals(2, metrics.rejectedSequenceCount)
        assertEquals(6, metrics.expectedTtcCount)
        assertEquals(true, metrics.maximumTtcErrorMs <= 1L)
        assertEquals(4, metrics.expectedCollisionCount)
        assertEquals(4, metrics.detectedCollisionCount)
    }

    companion object {
        private val HEADER = listOf(
            "sequence_id", "track_id", "previous_frame_id", "previous_captured_at_ns", "current_frame_id", "current_captured_at_ns",
            "previous_forward_m", "previous_lateral_m", "current_forward_m", "current_lateral_m", "pose_forward_m", "pose_lateral_m", "pose_yaw_rad",
            "pose_verified", "track_confidence", "expected_admitted", "expected_velocity_forward_mps", "expected_velocity_lateral_mps",
            "expected_ttc_ms", "expected_closest_distance_m", "expected_collision"
        ).joinToString("\t")
    }
}
