package com.linnan.blindassist.vision

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MobileFloorHeightTest {
    @Test fun twentyFrameCausalMedianRecoversTiltedFloorAndIgnoresOneBadFrame() {
        val calibration = MobileFloorHeight()
        val height = 1.42; val pitch = Math.toRadians(30.0); val roll = Math.toRadians(7.0)
        val n = doubleArrayOf(sin(roll) * cos(pitch), cos(roll) * cos(pitch), sin(pitch))
        val edge = tan(Math.PI / 8)
        val raw = DoubleArray(64)
        for (zone in 0 until 64) {
            val x = -edge + (zone % 8 + .5) * 2 * edge / 8
            val y = -edge + (zone / 8 + .5) * 2 * edge / 8
            val norm = sqrt(x * x + y * y + 1)
            raw[63 - zone] = 1000 * height * norm / (n[0] * x + n[1] * y + n[2])
        }
        repeat(19) {
            val progress = calibration.add(if (it == 5) DoubleArray(64) { 800.0 } else raw, IntArray(64) { 5 })
            assertFalse(progress.complete); assertNull(progress.calibration)
        }
        val progress = calibration.add(raw, IntArray(64) { 5 })
        assertTrue(progress.complete)
        val result = progress.calibration!!
        assertEquals(height, result.heightM, 1e-9)
        assertEquals(-30.0, result.pitchDeg, 1e-8)
        assertEquals(7.0, result.rollDeg, 1e-8)
        assertEquals(32, result.inliers)
        calibration.reset()
        assertEquals(1, calibration.add(raw, IntArray(64)).frameCount)
    }

    @Test fun invalidStatusesProduceExplicitCompletedFailure() {
        val calibration = MobileFloorHeight()
        var progress = calibration.add(DoubleArray(64) { 1400.0 }, IntArray(64))
        repeat(19) { progress = calibration.add(DoubleArray(64) { 1400.0 }, IntArray(64)) }
        assertTrue(progress.complete); assertNull(progress.calibration)
    }
}
