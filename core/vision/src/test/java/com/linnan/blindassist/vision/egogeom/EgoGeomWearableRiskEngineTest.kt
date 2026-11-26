package com.linnan.blindassist.vision.egogeom

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EgoGeomWearableRiskEngineTest {

    private val hS4 = EgoGeomContracts.S4_HEIGHT
    private val wS4 = EgoGeomContracts.S4_WIDTH
    private val fx = 294f
    private val fy = 294f
    private val cx = 192f
    private val cy = 144f
    private val rZup = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
    private val fwd = floatArrayOf(1f, 0f, 0f)

    private data class SynthFrame(
        val dep: FloatArray,
        val dh: FloatArray,
        val nx: FloatArray,
        val nd: FloatArray,
        val nz: FloatArray,
        val ls: FloatArray,
    )

    private fun makeConeScene(
        obsZ: Float?,
        obsNx: Float,
        obsNz: Float,
        obsY0: Int = 24,
        obsY1: Int = 38,
        obsX0: Int = 43,
        obsX1: Int = 53,
    ): SynthFrame {
        val n = hS4 * wS4
        val dep = FloatArray(n) { 8.0f }
        val dh = FloatArray(n) { 0.0f }
        val nx = FloatArray(n) { 0.0f }
        val nd = FloatArray(n) { 0.0f }
        val nz = FloatArray(n) { 1.0f }
        val ls = FloatArray(n) { -1.0f }
        for (y in 0 until hS4) {
            val ry = (y * 4.0f + 0.5f - cy) / fy
            if (ry > 0.28f) {
                val zFl = min(12.0f, 1.4f / ry)
                for (x in 0 until wS4) {
                    val idx = y * wS4 + x
                    dep[idx] = zFl
                    nx[idx] = 0.0f
                    nd[idx] = 0.98f
                    nz[idx] = 0.19f
                }
            }
        }
        if (obsZ != null) {
            for (y in obsY0 until obsY1) {
                for (x in obsX0 until obsX1) {
                    val idx = y * wS4 + x
                    dep[idx] = obsZ
                    dh[idx] = 1.3f
                    nx[idx] = obsNx
                    nd[idx] = 0.05f
                    nz[idx] = obsNz
                }
            }
        }
        return SynthFrame(dep, dh, nx, nd, nz, ls)
    }

    private fun makeLoomingGray(scaleZoom: Float): FloatArray {
        val out = FloatArray(hS4 * wS4)
        for (y in 0 until hS4) {
            val vy = (y - 36.0) / scaleZoom.toDouble() + 36.0
            for (x in 0 until wS4) {
                val ux = (x - 48.0) / scaleZoom.toDouble() + 48.0
                val g = 0.5 + 0.25 * sin(0.55 * ux) * cos(0.47 * vy) + 0.15 * sin(0.23 * ux + 0.31 * vy)
                out[y * wS4 + x] = g.coerceIn(0.0, 1.0).toFloat()
            }
        }
        return out
    }

    @Test
    fun narrowAngularConeMatchesPythonGoldenAndSuppressesParallelWalls() {
        val eng = NarrowAngularConeTtc()
        val clr = makeConeScene(null, 0f, 1f)
        val rClr = eng.step(clr.dep, clr.dh, clr.nx, clr.nd, clr.nz, clr.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        assertEquals(0.0f, rClr, 1e-4f)

        eng.reset()
        val wall = makeConeScene(2.4f, 0.99f, 0.05f)
        val rWall1 = eng.step(wall.dep, wall.dh, wall.nx, wall.nd, wall.nz, wall.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        val rWall2 = eng.step(wall.dep, wall.dh, wall.nx, wall.nd, wall.nz, wall.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        assertEquals(0.074516f, rWall1, 2e-4f)
        assertEquals(0.198486f, rWall2, 2e-4f)

        eng.reset()
        val f1 = makeConeScene(3.2f, 0.05f, 0.99f)
        val f2 = makeConeScene(2.4f, 0.05f, 0.99f)
        val rFront1 = eng.step(f1.dep, f1.dh, f1.nx, f1.nd, f1.nz, f1.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        val rFront2 = eng.step(f2.dep, f2.dh, f2.nx, f2.nd, f2.nz, f2.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        assertEquals(0.269130f, rFront1, 2e-4f)
        assertEquals(0.868756f, rFront2, 2e-4f)
        assertTrue("Frontal approaching overhang ($rFront2) must strongly exceed parallel wall ($rWall2)", rFront2 > rWall2 * 3f)
    }

    @Test
    fun gatedGyroTtcMatchesPythonGoldenAndMutesStaticAndFastHeadTurn() {
        val eng = GatedGyroTtc(lag = 2)
        // Stationary scene: muted to 0.0 by walking coherence gate
        val staticImg = makeLoomingGray(1.0f)
        repeat(4) {
            val r = eng.step(staticImg, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
            assertEquals(0.0f, r, 1e-5f)
        }

        // Approaching looming sequence: matches Python golden values to < 1e-3
        eng.reset()
        val zooms = floatArrayOf(1.0f, 1.06f, 1.12f, 1.19f)
        val expected = floatArrayOf(0.0f, 0.0f, 0.516606f, 0.885338f)
        for (i in zooms.indices) {
            val r = eng.step(makeLoomingGray(zooms[i]), hS4, wS4, fx, fy, cx, cy, rZup, fwd)
            assertEquals("Frame $i mismatch", expected[i], r, 1e-3f)
        }

        // Rapid 12-deg yaw turn -> muted to 0.0 by gyroscope rotation-rate gate
        val th = 12.0 * PI / 180.0
        val c = cos(th).toFloat()
        val s = sin(th).toFloat()
        val rTurned = floatArrayOf(
            c * rZup[0] - s * rZup[3], c * rZup[1] - s * rZup[4], c * rZup[2] - s * rZup[5],
            s * rZup[0] + c * rZup[3], s * rZup[1] + c * rZup[4], s * rZup[2] + c * rZup[5],
            rZup[6], rZup[7], rZup[8],
        )
        val rTurn = eng.step(makeLoomingGray(1.26f), hS4, wS4, fx, fy, cx, cy, rTurned, fwd)
        assertEquals(0.0f, rTurn, 1e-5f)
        assertTrue("Expected rotDeg ~ 12 deg, got ${eng.lastRotDeg}", eng.lastRotDeg > 10f)
    }

    @Test
    fun neckMountProfileCapturesHighHeadOverhangWithoutRetraining() {
        // High overhang around +15 deg elevation (v ~ 144 - 294*tan(15 deg) ~ 65 -> y_s4 ~ 14..20)
        val highOvh = makeConeScene(2.2f, 0.05f, 0.99f, obsY0 = 12, obsY1 = 20, obsX0 = 43, obsX1 = 53)
        val headEng = NarrowAngularConeTtc(
            elLoDeg = WearableMountProfile.HEAD_MOUNTED.elLoDeg,
            elHiDeg = WearableMountProfile.HEAD_MOUNTED.elHiDeg,
        )
        val neckEng = NarrowAngularConeTtc(
            elLoDeg = WearableMountProfile.NECK_MOUNTED.elLoDeg,
            elHiDeg = WearableMountProfile.NECK_MOUNTED.elHiDeg,
        )
        val rHead = headEng.step(highOvh.dep, highOvh.dh, highOvh.nx, highOvh.nd, highOvh.nz, highOvh.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        val rNeck = neckEng.step(highOvh.dep, highOvh.dh, highOvh.nx, highOvh.nd, highOvh.nz, highOvh.ls, hS4, wS4, fx, fy, cx, cy, rZup, fwd)
        assertTrue("Neck mount ($rNeck) should capture +15 deg overhead hazard missed by head mount ($rHead)", rNeck > 0.20f && rHead < 0.05f)
    }

    @Test
    fun glassCorridorEvaluatorRequires2FrameConfirmationAndSupportsOptionalTofEscalation() {
        val eval = GlassCorridorRiskEvaluator()
        val clearGlass = FloatArray(GlassCorridorRiskEvaluator.S16_FLOATS) { 0.001f }
        val rClear = eval.step(clearGlass, fx, fy, cx, cy, rZup, fwd)
        assertEquals(GlassAlertTier.NONE, rClear.tier)

        val modGlass = FloatArray(GlassCorridorRiskEvaluator.S16_FLOATS) { 0.58f }
        // Frame 1: single frame should NOT trigger yet (anti-flicker)
        val r1 = eval.step(modGlass, fx, fy, cx, cy, rZup, fwd)
        assertEquals(GlassAlertTier.NONE, r1.tier)
        assertEquals(1, r1.confirmedFrames)

        // Frame 2 without ToF: confirmed CAUTION
        val r2 = eval.step(modGlass, fx, fy, cx, cy, rZup, fwd)
        assertEquals(GlassAlertTier.CAUTION, r2.tier)

        // Frame 3 WITH optional ToF corroborating near surface at 1.4 m -> escalates to HIGH
        val r3Tof = eval.step(modGlass, fx, fy, cx, cy, rZup, fwd, optionalTofNearestM = 1.4f)
        assertEquals(GlassAlertTier.HIGH, r3Tof.tier)

        // High-confidence pure-RGB glass (p >= 0.70, zero ToF) -> reaches HIGH on frame 2 without ToF
        eval.reset()
        val strongGlass = FloatArray(GlassCorridorRiskEvaluator.S16_FLOATS) { 0.92f }
        assertEquals(GlassAlertTier.NONE, eval.step(strongGlass, fx, fy, cx, cy, rZup, fwd).tier)
        assertEquals(GlassAlertTier.HIGH, eval.step(strongGlass, fx, fy, cx, cy, rZup, fwd).tier)

        // Flat horizontal floor normal (|n_down| = 0.98) -> suppressed to NONE even if raw p_glass is high
        eval.reset()
        val flatFloorNormal = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH) { 0.98f }
        assertEquals(GlassAlertTier.NONE, eval.step(strongGlass, fx, fy, cx, cy, rZup, fwd, nrmDownS4 = flatFloorNormal).tier)
        assertEquals(GlassAlertTier.NONE, eval.step(strongGlass, fx, fy, cx, cy, rZup, fwd, nrmDownS4 = flatFloorNormal).tier)
    }

    @Test
    fun wearableArbitratorDispatchesDistinctChannelsAcrossFusionModes() {
        val clear = makeConeScene(null, 0f, 1f)
        val gray = makeLoomingGray(1.0f)
        val glassHigh = FloatArray(GlassCorridorRiskEvaluator.S16_FLOATS) { 0.90f }

        // 1. Solid obstacle absent, pure-RGB glass door ahead -> GLASS_HAZARD on 2nd frame
        val arb = WearableRiskArbitrator(mode = CameraImuFusionMode.FULL_3CH)
        arb.step(0.02f, clear.dep, clear.dh, clear.nx, clear.nd, clear.nz, clear.ls, gray, fx, fy, cx, cy, rZup, fwd, glassHigh)
        val g2 = arb.step(0.02f, clear.dep, clear.dh, clear.nx, clear.nd, clear.nz, clear.ls, gray, fx, fy, cx, cy, rZup, fwd, glassHigh)
        assertEquals(WearableHazardChannel.GLASS_HAZARD, g2.primaryChannel)

        // 2. Solid obstacle high -> SOLID_OBSTACLE takes precedence in FULL_3CH
        val s1 = arb.step(0.35f, clear.dep, clear.dh, clear.nx, clear.nd, clear.nz, clear.ls, gray, fx, fy, cx, cy, rZup, fwd, null)
        assertEquals(WearableHazardChannel.SOLID_OBSTACLE, s1.primaryChannel)

        // 3. ZERO_VIO_CAMERA_IMU mode ignores solidM6ParRisk and triggers OVERHANG_LOOMING from narrow cone alone
        val arbNoVio = WearableRiskArbitrator(mode = CameraImuFusionMode.ZERO_VIO_CAMERA_IMU)
        val ovh = makeConeScene(2.4f, 0.05f, 0.99f)
        val nv = arbNoVio.step(0.0f, ovh.dep, ovh.dh, ovh.nx, ovh.nd, ovh.nz, ovh.ls, gray, fx, fy, cx, cy, rZup, fwd, null)
        assertEquals(WearableHazardChannel.OVERHANG_LOOMING, nv.primaryChannel)
        assertTrue("Expected positive fused score in zero-VIO mode, got ${nv.fusedScore}", nv.fusedScore > 0.15f)
    }
}
