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

    @Test
    fun optimizedWorkspaceMatchesFrozenReference() {
        val width = 640
        val height = 480
        val depth = FloatArray(width * height)
        val random = java.util.Random(9917L)
        for (row in 0 until height) {
            val groundRow = maxOf(row, 264)
            val base = 320.0 / (groundRow - 240.0)
            for (column in 0 until width) {
                depth[row * width + column] = when {
                    (row * width + column) % 97 == 0 -> Float.NaN
                    (row * width + column) % 131 == 0 -> (base * 1.35).toFloat()
                    else -> (base * (1.0 + (random.nextDouble() - 0.5) * 0.004)).toFloat()
                }
            }
        }
        val reference = KnownHeightGroundPipeline.evaluateGeometryReference(
            depth, width, height, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
        )
        val optimized = KnownHeightGroundPipeline.evaluateGeometry(
            depth, width, height, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
        )
        assertGeometryEquals(reference, optimized)
        assertGeometryEquals(
            KnownHeightGroundPipeline.evaluateGeometryReference(
                FloatArray(width * height) { Float.NaN }, width, height,
                320.0, 320.0, 320.0, 240.0, 1.0,
            ),
            KnownHeightGroundPipeline.evaluateGeometry(
                FloatArray(width * height) { Float.NaN }, width, height,
                320.0, 320.0, 320.0, 240.0, 1.0,
            ),
        )
    }

    private fun assertGeometryEquals(expected: Any, actual: Any) {
        assertEquals(expected.javaClass, actual.javaClass)
        if (expected is KnownHeightGroundPipeline.Unknown && actual is KnownHeightGroundPipeline.Unknown) {
            assertEquals(expected.reason, actual.reason)
            return
        }
        expected as KnownHeightGroundPipeline.Geometry
        actual as KnownHeightGroundPipeline.Geometry
        assertEquals(expected.relativeHeight, actual.relativeHeight, 1e-12)
        assertEquals(expected.normalizedMedianResidual, actual.normalizedMedianResidual, 1e-12)
        assertEquals(expected.inlierFraction, actual.inlierFraction, 0.0)
        expected.normal.indices.forEach { assertEquals(expected.normal[it], actual.normal[it], 1e-12) }
        expected.features.indices.forEach { assertEquals(expected.features[it], actual.features[it], 1e-12) }
    }
}
