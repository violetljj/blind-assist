package com.linnan.blindassist.vision.egogeom

import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EgoGeomBevAndContractsTest {
    @Test
    fun analyticalHostFocalMatchesPythonContract() {
        // Atom 640x480 -> 384x288 nominal fx=fy=294, cx=192, cy=144: canonical 350/294 ~ 1.190476
        val hfAtom = EgoGeomContracts.analyticalHostFocal(294.0, 294.0, 192.0, 144.0)
        assertTrue("Expected ~1.1905, got $hfAtom", abs(hfAtom - 1.190476f) < 1e-3f)

        // Wide-angle & telephoto contract clamping [0.20, 4.0]
        assertEquals(4.0f, EgoGeomContracts.analyticalHostFocal(40.0, 40.0, 192.0, 144.0), 1e-5f)
        assertEquals(0.20f, EgoGeomContracts.analyticalHostFocal(4000.0, 4000.0, 192.0, 144.0), 1e-5f)
    }

    @Test
    fun gravityAlignedRaysAreUnitNorm() {
        val rays = EgoGeomContracts.buildGravityAlignedRays(
            fx = 294f, fy = 294f, cx = 192f, cy = 144f,
            gCamX = 0.05f, gCamY = 0.96f, gCamZ = 0.27f,
        )
        val plane = EgoGeomContracts.DEPTH_FLOATS
        for (idx in intArrayOf(0, plane / 2 + 192, plane - 1)) {
            val len = sqrt(
                rays[idx] * rays[idx] +
                    rays[plane + idx] * rays[plane + idx] +
                    rays[2 * plane + idx] * rays[2 * plane + idx]
            )
            assertEquals(1.0f, len, 1e-5f)
        }
    }

    @Test
    fun packRealH3PreservesSignedNegativeDipsAndDropoutMask() {
        val hist = FloatArray(64 * 16)
        val distMm = IntArray(64) { 640 }
        val status = IntArray(64) { 5 }
        val nbTargets = IntArray(64) { 1 }
        // Zone 0: strong negative dip in bin 3; Zone 1: invalid status (dropout)
        hist[0 * 16 + 0] = 100f
        hist[0 * 16 + 3] = -40f
        status[1] = 255
        distMm[1] = 0
        nbTargets[1] = 0

        val tofS4 = EgoGeomContracts.packRealH3ToTofGridS4(hist, distMm, status, nbTargets)
        val planeS4 = EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH
        // Zone (0,0) -> valid mask ch1 == 1.0, bin3 (ch 5) strictly negative!
        assertEquals(1.0f, tofS4[1 * planeS4 + 0], 1e-6f)
        assertTrue("Expected signed negative CNH bin3 < 0", tofS4[(2 + 3) * planeS4 + 0] < -0.5f)
        // Zone (0,1) starts at col 12 -> valid mask ch1 == 0.0, cnh_valid ch18 == 1.0
        assertEquals(0.0f, tofS4[1 * planeS4 + 12], 1e-6f)
        assertEquals(1.0f, tofS4[18 * planeS4 + 12], 1e-6f)
        assertEquals(1.0f, tofS4[19 * planeS4 + 12], 1e-6f)
    }

    @Test
    fun bevV3SeparatesClearFloorFromCloseVerticalObstacleAndSupportsParallaxRefutation() {
        val hS4 = EgoGeomContracts.S4_HEIGHT
        val wS4 = EgoGeomContracts.S4_WIDTH
        val n = hS4 * wS4
        val fx = 294f; val fy = 294f; val cx = 192f; val cy = 144f
        val rZup = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
        val t0 = floatArrayOf(0f, 0f, 1.4f)
        val fwd = floatArrayOf(1f, 0f, 0f)

        // Build clean flat walkable floor 1.4 m below camera
        val depFloor = FloatArray(n) { 8.0f }
        val dhFloor = FloatArray(n) { 0.0f }
        val nrmFloor = FloatArray(n) { 0.0f }
        val lsFloor = FloatArray(n) { -1.0f }
        for (y in 0 until hS4) {
            val vPix = y * 4f + 0.5f
            val ry = (vPix - cy) / fy
            for (x in 0 until wS4) {
                val idx = y * wS4 + x
                if (ry > 0.15f) {
                    depFloor[idx] = (1.4f / ry).coerceAtMost(12f)
                    nrmFloor[idx] = 0.98f // flat floor gravity-aligned normal
                }
            }
        }

        val bev = EgoGeomTemporalBevV3()
        val clear1 = bev.step(depFloor, dhFloor, nrmFloor, lsFloor, hS4, wS4, fx, fy, cx, cy, rZup, t0, fwd)
        val clear2 = bev.step(depFloor, dhFloor, nrmFloor, lsFloor, hS4, wS4, fx, fy, cx, cy, rZup, t0, fwd)
        assertTrue("Clear floor should have ~0 risk, got ${clear2.riskGeomM6}", clear2.riskGeomM6 < 0.02f)
        assertTrue("Ground scale should recover ~1.4m, got ${clear1.groundScaleM}", abs(clear1.groundScaleM - 1.4f) < 0.25f)

        // Inject moderate vertical obstacle directly ahead at z = 1.2 m (unsaturating support to verify 2-frame boost)
        val depObs = depFloor.copyOf()
        val dhObs = dhFloor.copyOf()
        val nrmObs = nrmFloor.copyOf()
        for (y in 26..36) {
            for (x in 44..52) {
                val idx = y * wS4 + x
                depObs[idx] = 1.2f
                dhObs[idx] = 0.6f
                nrmObs[idx] = 0.05f // vertical obstacle face
            }
        }
        val obs1 = bev.step(depObs, dhObs, nrmObs, lsFloor, hS4, wS4, fx, fy, cx, cy, rZup, t0, fwd)
        val obs2 = bev.step(depObs, dhObs, nrmObs, lsFloor, hS4, wS4, fx, fy, cx, cy, rZup, t0, fwd)
        assertTrue("Obstacle frame 1 risk (${obs1.riskGeomM6}) should exceed clear (${clear2.riskGeomM6})", obs1.riskGeomM6 > 0.10f)
        assertTrue("Co-supported frame 2 risk (${obs2.riskGeomM6}) should exceed frame 1 (${obs1.riskGeomM6})", obs2.riskGeomM6 > obs1.riskGeomM6)
        assertTrue("Nearest corridor obstacle should be ~1.2m, got ${obs2.nearestCorridorM}", obs2.nearestCorridorM in 0.8f..1.6f)
    }
}
