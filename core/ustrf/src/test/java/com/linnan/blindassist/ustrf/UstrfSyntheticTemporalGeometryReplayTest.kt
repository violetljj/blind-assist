package com.linnan.blindassist.ustrf

import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertEquals
import org.junit.Test

class UstrfSyntheticTemporalGeometryReplayTest {
    @Test
    fun csvTsvReplayFeedsDepthAndTemporalContractsInBatch() {
        val root = Files.createTempDirectory("ustrf-synthetic-replay")
        try {
            val frames = Files.createDirectories(root.resolve("frames"))
            writeDepth(frames.resolve("first.csv"), forwardMeters = 3.5f, body = true)
            writeDepth(frames.resolve("second.csv"), forwardMeters = 3.0f, body = true)
            writeDepth(frames.resolve("gap-first.csv"), forwardMeters = 3.5f, body = false)
            writeDepth(frames.resolve("gap-second.csv"), forwardMeters = 3.0f, body = false)
            Files.writeString(root.resolve("kotlin_replay.tsv"), manifest())

            val metrics = UstrfSyntheticTemporalGeometryReplay().replay(root)

            assertEquals(2, metrics.sequenceCount)
            assertEquals(2, metrics.admittedPairCount)
            assertEquals(0, metrics.rejectedPairCount)
            assertEquals(1, metrics.expectedStaticSequenceCount)
            assertEquals(1, metrics.matchedStaticSequenceCount)
            assertEquals(1, metrics.visibilityGapSequenceCount)
            assertEquals(0, metrics.falseDropSequenceCount)
            assertEquals(0, metrics.expectedDropSequenceCount)
            assertEquals(0, metrics.detectedDropSequenceCount)
            assertEquals(true, metrics.geometryEvidenceCount > 0)
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun generatedEvidenceReplayIsCheckedWhenItsExplicitRootIsProvided() {
        val root = System.getenv("USTRF_SYNTHETIC_REPLAY_ROOT")?.takeIf { it.isNotBlank() } ?: return
        val metrics = UstrfSyntheticTemporalGeometryReplay().replay(Path.of(root))

        assertEquals(14, metrics.sequenceCount)
        assertEquals(14, metrics.admittedPairCount)
        assertEquals(0, metrics.rejectedPairCount)
        assertEquals(8, metrics.expectedStaticSequenceCount)
        assertEquals(8, metrics.matchedStaticSequenceCount)
        assertEquals(2, metrics.visibilityGapSequenceCount)
        assertEquals(0, metrics.falseDropSequenceCount)
        assertEquals(2, metrics.expectedDropSequenceCount)
        assertEquals(2, metrics.detectedDropSequenceCount)
    }

    private fun writeDepth(path: Path, forwardMeters: Float, body: Boolean) {
        val values = Array(48) { row -> FloatArray(64) { column ->
            val denominator = row - 23.5f
            val ground = if (denominator > 0f) 1.5f * 52f / denominator else 0f
            if (ground in .2f..5f) ground else 0f
        } }
        if (body) {
            val vertical = (23.5f - (.6f - 1.5f) * 52f / forwardMeters).toInt()
            val horizontal = 32
            for (row in vertical - 2..vertical + 2) for (column in horizontal - 2..horizontal + 2) {
                if (row in 0 until 48 && column in 0 until 64) values[row][column] = forwardMeters
            }
        }
        Files.writeString(path, values.joinToString("\n") { row -> row.joinToString(",") } + "\n")
    }

    private fun manifest(): String = listOf(
        HEADER,
        row("static", "lower_body", "true", "false", "0", "1000", "frames/first.csv", "1", "1100", "frames/second.csv"),
        row("gap", "visibility_gap", "false", "true", "2", "2000", "frames/gap-first.csv", "3", "2100", "frames/gap-second.csv")
    ).joinToString("\n", postfix = "\n")

    private fun row(
        id: String, kind: String, static: String, gap: String, firstId: String, firstTime: String, firstDepth: String,
        secondId: String, secondTime: String, secondDepth: String
    ) = listOf(
        id, kind, "0", "3.5", if (kind == "lower_body") ".6" else ".6", static, gap, "false",
        firstId, firstTime, "0", firstDepth, secondId, secondTime, ".5", secondDepth,
        ".5", "0", "0", "true"
    ).joinToString("\t")

    companion object {
        private const val HEADER = "sequence_id\ttarget_kind\ttarget_lateral_m\ttarget_forward_m\ttarget_height_m\texpected_temporal_static_match\texpected_must_not_emit_drop\texpected_must_emit_drop\tframe0_id\tframe0_captured_at_ns\tframe0_camera_forward_m\tframe0_depth_csv_path\tframe1_id\tframe1_captured_at_ns\tframe1_camera_forward_m\tframe1_depth_csv_path\tpose_forward_meters\tpose_lateral_meters\tpose_yaw_radians\tpose_verified"
    }
}
