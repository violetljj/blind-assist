package com.linnan.blindassist.hftf.metricdepth

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class KnownHeightGroundPipelineTest {
    @Test
    fun rejectsMissingGroundSupport() {
        val result = KnownHeightGroundPipeline.evaluate(
            FloatArray(640 * 480) { Float.NaN }, 640, 480, 320.0, 320.0, 320.0, 240.0, 1.0,
        )
        assertEquals(
            "INSUFFICIENT_GROUND_CANDIDATES",
            (result as KnownHeightGroundPipeline.Unknown).reason,
        )
    }

    @Test
    fun recoversSyntheticHorizontalGround() {
        val width = 640
        val height = 480
        val depth = FloatArray(width * height)
        for (row in 0 until height) {
            val groundRow = maxOf(row, 264)
            val z = 320.0 / (groundRow - 240.0)
            for (column in 0 until width) depth[row * width + column] = z.toFloat()
        }
        val result = KnownHeightGroundPipeline.evaluate(
            depth, width, height, 320.0, 320.0, 320.0, 240.0, 1.0,
        )
        assertTrue(result.toString(), result is KnownHeightGroundPipeline.Valid)
        result as KnownHeightGroundPipeline.Valid
        assertTrue(abs(result.relativeHeight - 1.0) < 1e-4)
        assertTrue(abs(abs(result.normal[1]) - 1.0) < 1e-4)
    }
}
