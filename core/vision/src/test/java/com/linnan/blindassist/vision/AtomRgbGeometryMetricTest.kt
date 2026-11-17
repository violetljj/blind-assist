package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Direction A metric chain (ToF scale -> own ground x eye height -> previous) on synthetic scenes. */
class AtomRgbGeometryMetricTest {
    private val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    private val side = 160
    private fun geometry() = AtomRgbGeometry(side, side, AtomRgbGeometry.Calibration(160.0, 160.0, side / 2.0, side / 2.0, identity))

    /** Level camera (pitch 0, roll 0: corridor = camera), flat floor at [eye] metres; model depth = truth / [s]. */
    private fun floorScene(eye: Double, s: Double) = FloatArray(side * side) { i ->
        val dy = (i / side - side / 2.0) / 160.0
        if (dy > 1e-3) (eye / dy / s).toFloat() else Float.NaN
    }

    @Test fun floorScaleRecoversEyeHeightOverPredictedPlaneDistance() {
        val s = geometry().floorScale(floorScene(1.5, 1.25), 0.0, 0.0, 1.5)!!
        assertEquals(1.25, s, 1e-3)
    }

    @Test fun floorScaleRejectsImplausibleScale() {
        // Predicted floor 3x too near: s = 3 is outside [0.5, 2].
        assertNull(geometry().floorScale(floorScene(1.5, 3.0), 0.0, 0.0, 1.5))
    }

    @Test fun tofScaleIsMedianRangeRatioOverValidZones() {
        val g = geometry()
        val depth = FloatArray(side * side) { 2f }                          // frontal wall, model says 2 m
        val factors = g.zoneDisparity(FloatArray(side * side) { .5f }).opticalDepthFactor
        val raw = DoubleArray(64) { Double.NaN }
        for (zone in 0 until 64) raw[63 - zone] = 1.1 * 2.0 / factors[zone] * 1000   // true wall at 2.2 m
        assertEquals(1.1, g.tofScale(depth, raw, IntArray(64) { 5 })!!, 2e-3)
        // Fewer than five valid zones: no ToF scale.
        val status = IntArray(64) { if (it < 4) 5 else 255 }
        assertNull(g.tofScale(depth, raw, status))
    }

    @Test fun chainOrderIsTofThenFloorThenPrevious() {
        val g = geometry()
        val valid = BooleanArray(side * side) { true }
        val floor = floorScene(1.5, 1.25)
        val noTof = DoubleArray(64) { 0.0 }; val bad = IntArray(64) { 255 }
        val first = g.processMetric(floor, valid, noTof, bad, 0.0, 0.0, 1.5)
        assertEquals(AtomRgbGeometry.Anchor.FLOOR, first.anchor)
        assertEquals(1.25, first.fit!!.a, 1e-3); assertEquals(0.0, first.fit.b, 0.0)
        val empty = FloatArray(side * side) { Float.NaN }
        val second = g.processMetric(empty, valid, noTof, bad, 0.0, 0.0, 1.5)
        assertEquals(AtomRgbGeometry.Anchor.PREVIOUS, second.anchor)
        assertEquals(first.fit.a, second.fit!!.a, 0.0)
        val wall = FloatArray(side * side) { 2f }
        val factors = g.zoneDisparity(FloatArray(side * side) { .5f }).opticalDepthFactor
        val raw = DoubleArray(64); for (zone in 0 until 64) raw[63 - zone] = .9 * 2.0 / factors[zone] * 1000
        val third = g.processMetric(wall, valid, raw, IntArray(64) { 5 }, 0.0, 0.0, 1.5)
        assertEquals(AtomRgbGeometry.Anchor.TOF, third.anchor)
        assertEquals(.9, third.fit!!.a, 2e-3)
        assertEquals(AtomRgbGeometry.Anchor.NONE, geometry().processMetric(empty, valid, noTof, bad, 0.0, 0.0, 1.5).anchor)
    }

    @Test fun scaledReadoutPlacesWallAtScaledDistance() {
        // Wall 1.0 m by the model, ToF says 1.5 m: scaled wall at 1.5 m is inside the 2 m boxes with its median distance.
        val g = geometry()
        val wall = FloatArray(side * side) { 1f }
        val factors = g.zoneDisparity(FloatArray(side * side) { 1f }).opticalDepthFactor
        val raw = DoubleArray(64); for (zone in 0 until 64) raw[63 - zone] = 1.5 / factors[zone] * 1000
        val r = g.processMetric(wall, BooleanArray(side * side) { true }, raw, IntArray(64) { 5 }, 0.0, 0.0, 1.5)
        assertTrue(r.head.alarm && r.body.alarm)
        assertEquals(1.5, r.body.medianDepthM, 1e-2)
    }
}
