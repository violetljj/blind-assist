package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AtomRgbGeometryTest {
    private val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    private fun geometry(w: Int = 80, h: Int = 80) = AtomRgbGeometry(w, h,
        AtomRgbGeometry.Calibration(80.0, 80.0, w / 2.0, h / 2.0, identity))

    @Test fun robustAffineRecoversPositiveScaleAndRejectsInsufficientAndNegative() {
        val x = DoubleArray(20) { .2 + it * .1 }
        val y = DoubleArray(20) { 3 * x[it] + 2 }.also { it[10] += 20 }
        val fit = AtomRgbGeometry.fitAffine(y, x)!!
        assertEquals(3.0, fit.a, 1e-10)
        assertEquals(2.0, fit.b, 1e-10)
        assertNull(AtomRgbGeometry.fitAffine(y.copyOf(5), x.copyOf(5)))
        assertNull(AtomRgbGeometry.fitAffine(DoubleArray(20) { -x[it] }, x))
    }

    @Test fun rawZoneRotationAndOpticalDepthFactorsMatchAnchoringContract() {
        val g = geometry()
        val disparity = FloatArray(6400) { i -> (1 + (i % 80) * .04 + (i / 80) * .02).toFloat() }
        val zones = g.zoneDisparity(disparity)
        val raw = DoubleArray(64)
        // Generate ranges from d=2/z+.5; reversing in the implementation must recover this fit.
        for (zone in 0 until 64) raw[63 - zone] = 2000 / ((zones.disparity[zone] - .5) * zones.opticalDepthFactor[zone])
        val fit = g.tofAnchor(disparity, raw, IntArray(64) { 5 })!!
        assertEquals(2.0, fit.a, 1e-10)
        assertEquals(.5, fit.b, 1e-10)
        assertNull(g.tofAnchor(disparity, raw, IntArray(64) { 0 }))
    }

    @Test fun readoutUsesUprightMaskBothBoxesAndInclusiveBoundary() {
        val g = AtomRgbGeometry(20, 20, AtomRgbGeometry.Calibration(100.0, 100.0, 10.0, 0.0, identity))
        val disparity = FloatArray(400) { 1f }
        val all = BooleanArray(400) { true }
        val (head, body) = g.readout(disparity, all, AtomRgbGeometry.Fit(1.0, 0.0), 0.0, 0.0)
        assertEquals(400, head.pixels); assertTrue(head.alarm)
        assertEquals(0, body.pixels); assertFalse(body.alarm)
        val masked = g.readout(disparity, BooleanArray(400) { it < 149 }, AtomRgbGeometry.Fit(1.0, 0.0), 0.0, 0.0)
        assertEquals(149, masked.first.pixels); assertFalse(masked.first.alarm)
        val edge = AtomRgbGeometry(2, 2, AtomRgbGeometry.Calibration(100.0, 100.0, 0.0, -42.0, identity))
        val shared = edge.readout(FloatArray(4) { 1f }, BooleanArray(4) { true }, AtomRgbGeometry.Fit(1.0, 0.0), 0.0, 0.0)
        assertEquals(2, shared.first.pixels); assertEquals(4, shared.second.pixels)
    }

    @Test fun floorRansacRecoversKnownPlaneAndPreviousFitResets() {
        val g = geometry(80, 160)
        val disp = FloatArray(80 * 160) { i -> (4 * ((i / 80 - 80.0) / 80 / 1.35) + 2).toFloat() }
        val floor = g.floorAnchor(disp, 0.0, 0.0)!!
        assertEquals(4.0, floor.a, 1e-5); assertEquals(2.0, floor.b, 1e-5)
        val first = g.process(disp, BooleanArray(disp.size) { true }, DoubleArray(64), IntArray(64), 0.0, 0.0)
        assertEquals(AtomRgbGeometry.Anchor.FLOOR, first.anchor)
        val second = g.process(FloatArray(disp.size) { Float.NaN }, BooleanArray(disp.size), DoubleArray(64), IntArray(64), 0.0, 0.0)
        assertEquals(AtomRgbGeometry.Anchor.PREVIOUS, second.anchor)
        g.reset()
        val reset = g.process(FloatArray(disp.size) { Float.NaN }, BooleanArray(disp.size), DoubleArray(64), IntArray(64), 0.0, 0.0)
        assertEquals(AtomRgbGeometry.Anchor.NONE, reset.anchor)
    }

    @Test fun poseGateAndTemporalWindowMatchPythonRules() {
        assertTrue(AtomRgbGeometry.poseInEnvelope(10.0, -25.0))
        assertFalse(AtomRgbGeometry.poseInEnvelope(10.01, 0.0))
        assertFalse(AtomRgbGeometry.poseInEnvelope(-50.0, 25.01))
        assertTrue(AtomRgbGeometry.poseInEnvelope(Double.NaN, 90.0))
        val rotation = AtomRgbGeometry.poseRotation(-30.0, 0.0)
        assertEquals(.5, rotation[5], 1e-12)
        val temporal = AtomRgbTwoOfThree()
        assertEquals(false to false, temporal.push(true, false))
        assertEquals(true to false, temporal.push(true, true))
        assertEquals(true to false, temporal.push(false, false))
        assertEquals(false to true, temporal.push(false, true))
        temporal.reset()
        assertEquals(false to false, temporal.push(true, true))
    }
}
