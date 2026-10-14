package com.linnan.blindassist

import com.linnan.blindassist.vision.AtomRgbGeometry
import org.junit.Assert.*
import org.junit.Test

class MobileFusionDemoRulesTest {
    private fun rad(deg: Double) = Math.toRadians(deg)

    @Test fun pitchIsRemovedWhileYawAndRollStay() {
        val fitted = MobileFusionDemoRules.euler(rad(10.5), rad(14.5), rad(1.0))
        val expected = MobileFusionDemoRules.euler(rad(10.5), 0.0, rad(1.0))
        assertEquals(14.5, MobileFusionDemoRules.pitchDeg(fitted), 1e-9)
        assertArrayEquals(expected, MobileFusionDemoRules.withoutPitch(fitted), 1e-12)
        assertEquals(0.0, MobileFusionDemoRules.pitchDeg(MobileFusionDemoRules.withoutPitch(fitted)), 1e-9)
    }

    @Test fun walkStillLimitsGateRgbAndTof() {
        assertEquals(1.8, MobileFusionDemoRules.limitM(true), 0.0)
        assertEquals(0.8, MobileFusionDemoRules.limitM(false), 0.0)
        assertTrue(MobileFusionDemoRules.keepRgb(true, 1.2, 1.8))
        assertFalse(MobileFusionDemoRules.keepRgb(true, 1.2, 0.8))
        assertFalse("unmeasured RGB depth is not kept", MobileFusionDemoRules.keepRgb(true, Double.NaN, 1.8))
        assertFalse(MobileFusionDemoRules.keepTof(true, 1.2, 0.8))
        assertTrue("ToF alarm without an in-box zone is kept", MobileFusionDemoRules.keepTof(true, Double.NaN, 0.8))
        assertFalse(MobileFusionDemoRules.keepTof(false, 0.5, 1.8))
    }

    @Test fun gaitNeedsSustainedAccelerationVariation() {
        val still = MobileGaitDetector(); val walk = MobileGaitDetector()
        for (i in 0..200) {
            val t = i * 10_000_000L
            still.add(t, listOf(0.0, 0.0, 1.0 + if (i % 2 == 0) .002 else -.002))
            walk.add(t, listOf(0.0, 0.0, 1.0 + .08 * kotlin.math.sin(2 * Math.PI * 1.8 * i / 100.0)))
        }
        assertFalse(still.walking(2_000_000_000L))
        assertTrue(walk.walking(2_000_000_000L))
        assertFalse("stale IMU counts as still", walk.walking(4_000_000_000L))
    }

    @Test fun tofQueryDepthFindsLevelWallInHeadBox() {
        val geometry = AtomRgbGeometry(48, 64, AtomRgbGeometry.Calibration(40.0, 40.0, 24.0, 32.0,
            MobileFusionDemoRules.euler(0.0, 0.0, 0.0)))
        val status = IntArray(64) { 5 }
        val near = geometry.tofQueryDepth(DoubleArray(64) { 1000.0 }, status, 0.0, 0.0)
        assertEquals("radial 1 m: nearest in-box forward distance comes from an outer zone", 0.95, near.first, 0.05)
        assertTrue("level 1 m wall reaches only ~0.41 m below the eye", near.second.isNaN())
        val far = geometry.tofQueryDepth(DoubleArray(64) { 3000.0 }, status, 0.0, 0.0)
        assertTrue(far.first.isNaN() && far.second.isNaN())
        val invalid = geometry.tofQueryDepth(DoubleArray(64) { 1000.0 }, IntArray(64) { 255 }, 0.0, 0.0)
        assertTrue(invalid.first.isNaN())
        assertEquals(4, geometry.boxOutlines(0.0, 0.0).size)
    }
}
