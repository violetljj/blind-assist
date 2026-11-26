package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.egogeom.CameraImuFusionMode
import com.linnan.blindassist.vision.egogeom.EgoGeomContracts
import com.linnan.blindassist.vision.egogeom.EgoGeomQnnRuntimeFactory
import com.linnan.blindassist.vision.egogeom.EgoGeomTemporalBevV3
import com.linnan.blindassist.vision.egogeom.GlassAlertTier
import com.linnan.blindassist.vision.egogeom.WearableHazardChannel
import com.linnan.blindassist.vision.egogeom.WearableRiskArbitrator
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Live S24 Ultra (Hexagon v75) instrumented verification of `EgoGeomQnnRuntime` (`v4.1 pc_fp16dec_a16`, 59.1 MB)
 * + `EgoGeomTemporalBevV3` on real synchronized Atom S3R RGB + VL53L8CH H3 signed CNH fixtures.
 */
@RunWith(AndroidJUnit4::class)
class EgoGeomQnnRuntimeDeviceTest {
    private fun readRawFloats(file: File, count: Int): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size == count * 4) { "Expected ${count * 4} bytes in $file, got ${bytes.size}" }
        val out = FloatArray(count)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun qnnHtpV41PcFp16DecA16RunsZeroInterferenceAndFeedsBevV3() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as Application
        val dir = File(
            InstrumentationRegistry.getArguments().getString("egogeomFixtureRoot")
                ?: File(context.getExternalFilesDir(null) ?: context.filesDir, "egogeom-v41-smoke").absolutePath,
        )
        val omniBin = File(dir, EgoGeomQnnRuntimeFactory.OMNI_BIN)
        val rgbBin = File(dir, EgoGeomQnnRuntimeFactory.PURE_RGB_BIN)
        require(omniBin.isFile && rgbBin.isFile) { "Missing QNN binaries in $dir" }

        val imgFile = File(dir, "atom_bg.jpg")
        val bmp = requireNotNull(BitmapFactory.decodeFile(imgFile.absolutePath))
        val pixels = IntArray(bmp.width * bmp.height)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val grayS4 = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
        val rgbNchw = EgoGeomContracts.preprocessArgbToNchw(pixels, bmp.width, bmp.height, outGrayS4 = grayS4)
        bmp.recycle()

        val fx = 294f; val fy = 294f; val cx = 192f; val cy = 144f
        val rays = EgoGeomContracts.buildGravityAlignedRays(fx, fy, cx, cy)
        val hostFocal = EgoGeomContracts.analyticalHostFocal(fx.toDouble(), fy.toDouble(), cx.toDouble(), cy.toDouble())
        val tofBg = readRawFloats(File(dir, "h3_bg_tof_s4.raw"), EgoGeomContracts.TOF_S4_FLOATS)
        val tofObj = readRawFloats(File(dir, "h3_obj_tof_s4.raw"), EgoGeomContracts.TOF_S4_FLOATS)

        val pureOut = EgoGeomQnnRuntime(app, rgbBin).use { rt ->
            assertTrue(!rt.supportsTof)
            rt.infer(rgbNchw, rays, hostFocal)
        }

        var maxZeroInt = 0f
        var nanCount = 0
        val omniExecMs = ArrayList<Double>()
        var bgRoiMean = 0.0
        var objRoiMean = 0.0
        var bevBgRisk = 0f
        var bevObjRisk = 0f

        EgoGeomQnnRuntime(app, omniBin).use { rt ->
            assertTrue(rt.supportsTof)
            val zeroOut = rt.infer(rgbNchw, rays, hostFocal, tofGridS4 = null, hasTof = false)
            for (i in zeroOut.depthM.indices) {
                val d = abs(zeroOut.depthM[i] - pureOut.depthM[i])
                if (d > maxZeroInt) maxZeroInt = d
            }

            // Warmup + 15 timed live HTP executions inside Android app process
            repeat(3) { rt.infer(rgbNchw, rays, hostFocal, tofBg, hasTof = true) }
            lateinit var bgOut: com.linnan.blindassist.vision.egogeom.EgoGeomOutput
            repeat(15) {
                bgOut = rt.infer(rgbNchw, rays, hostFocal, tofBg, hasTof = true)
                omniExecMs.add(bgOut.executeMs)
            }
            val objOut = rt.infer(rgbNchw, rays, hostFocal, tofObj, hasTof = true)

            for (arr in listOf(bgOut.depthM, bgOut.deltaHM, bgOut.normalGrav, bgOut.logSigma, objOut.depthM)) {
                for (v in arr) if (!v.isFinite()) nanCount++
            }
            fun centerRoiMean(depth: FloatArray): Double {
                var sum = 0.0; var cnt = 0
                for (y in 72 until 216) {
                    val row = y * 384
                    for (x in 96 until 288) { sum += depth[row + x]; cnt++ }
                }
                return sum / cnt
            }
            bgRoiMean = centerRoiMean(bgOut.depthM)
            objRoiMean = centerRoiMean(objOut.depthM)

            val rZup = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
            val t0 = floatArrayOf(0f, 0f, 1.4f)
            val fwd = floatArrayOf(1f, 0f, 0f)
            val bev = EgoGeomTemporalBevV3()
            bev.stepFromFullOutputs(bgOut.depthM, bgOut.deltaHM, bgOut.normalGrav, bgOut.logSigma, fx, fy, cx, cy, rZup, t0, fwd, grayS4)
            bevBgRisk = bev.stepFromFullOutputs(bgOut.depthM, bgOut.deltaHM, bgOut.normalGrav, bgOut.logSigma, fx, fy, cx, cy, rZup, t0, fwd, grayS4).riskGeomM6
            bev.reset()
            bev.stepFromFullOutputs(objOut.depthM, objOut.deltaHM, objOut.normalGrav, objOut.logSigma, fx, fy, cx, cy, rZup, t0, fwd, grayS4)
            bevObjRisk = bev.stepFromFullOutputs(objOut.depthM, objOut.deltaHM, objOut.normalGrav, objOut.logSigma, fx, fy, cx, cy, rZup, t0, fwd, grayS4).riskGeomM6
        }

        omniExecMs.sort()
        val medMs = omniExecMs[omniExecMs.size / 2]
        val report = JSONObject()
            .put("model_omni", omniBin.name)
            .put("model_omni_bytes", omniBin.length())
            .put("nan_count", nanCount)
            .put("chip_zero_interference_max_abs_m", maxZeroInt.toDouble())
            .put("android_htp_omni_median_ms", medMs)
            .put("android_htp_omni_min_ms", omniExecMs.first())
            .put("android_htp_omni_max_ms", omniExecMs.last())
            .put("bg_roi_mean_m", bgRoiMean)
            .put("obj_roi_mean_m", objRoiMean)
            .put("obj_delta_m", objRoiMean - bgRoiMean)
            .put("bev_bg_m6_risk", bevBgRisk.toDouble())
            .put("bev_obj_m6_risk", bevObjRisk.toDouble())
        File(dir, "egogeom_android_device_report.json").writeText(report.toString(2))

        assertEquals("Expected 0 NaNs", 0, nanCount)
        assertEquals("Expected exact 0.0 m chip zero-interference", 0.0f, maxZeroInt, 1e-6f)
        assertTrue("Expected live Android HTP median < 35 ms, got $medMs", medMs < 35.0)
        assertTrue("Expected object insertion to reduce depth (${objRoiMean - bgRoiMean})", objRoiMean < bgRoiMean)
    }

    @Test
    fun qnnHtpV41WithG2GlassHeadRunsLiveAndFeedsWearableEngine() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as Application
        val dir = File(
            InstrumentationRegistry.getArguments().getString("egogeomG2FixtureRoot")
                ?: File(context.getExternalFilesDir(null) ?: context.filesDir, "egogeom-v41-g2").absolutePath,
        )
        val rgbG2Bin = File(dir, EgoGeomQnnRuntimeFactory.PURE_RGB_G2_BIN)
        val omniG2Bin = File(dir, EgoGeomQnnRuntimeFactory.OMNI_G2_BIN)
        require(rgbG2Bin.isFile && omniG2Bin.isFile) { "Missing G2 QNN binaries in $dir" }

        fun loadJpg(name: String): Pair<FloatArray, FloatArray> {
            val bmp = requireNotNull(BitmapFactory.decodeFile(File(dir, name).absolutePath))
            val pixels = IntArray(bmp.width * bmp.height)
            bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            val gray = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
            val rgb = EgoGeomContracts.preprocessArgbToNchw(pixels, bmp.width, bmp.height, outGrayS4 = gray)
            bmp.recycle()
            return rgb to gray
        }

        val (glassRgb, glassGray) = loadJpg("arkit_glass_door.jpg")
        val (clearRgb, clearGray) = loadJpg("coda_clear_rain.jpg")
        val fx = 294f; val fy = 294f; val cx = 192f; val cy = 144f
        val rays = EgoGeomContracts.buildGravityAlignedRays(fx, fy, cx, cy)
        val hostFocal = EgoGeomContracts.analyticalHostFocal(fx.toDouble(), fy.toDouble(), cx.toDouble(), cy.toDouble())
        val rZup = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
        val fwd = floatArrayOf(1f, 0f, 0f)

        val pureExecMs = ArrayList<Double>()
        lateinit var pureGlassOut: com.linnan.blindassist.vision.egogeom.EgoGeomOutput
        lateinit var pureClearOut: com.linnan.blindassist.vision.egogeom.EgoGeomOutput
        EgoGeomQnnRuntime(app, rgbG2Bin).use { rt ->
            assertTrue(!rt.supportsTof)
            assertTrue("Expected supportsGlass=true for Pure-RGB+G2", rt.supportsGlass)
            repeat(3) { rt.infer(glassRgb, rays, hostFocal) }
            repeat(15) {
                pureGlassOut = rt.infer(glassRgb, rays, hostFocal)
                pureExecMs.add(pureGlassOut.executeMs)
            }
            pureClearOut = rt.infer(clearRgb, rays, hostFocal)
        }

        var maxDepthDiff = 0f
        var maxGlassDiff = 0f
        var nanCount = 0
        val omniExecMs = ArrayList<Double>()
        EgoGeomQnnRuntime(app, omniG2Bin).use { rt ->
            assertTrue(rt.supportsTof)
            assertTrue("Expected supportsGlass=true for Omni+G2", rt.supportsGlass)
            val zeroOut = rt.infer(glassRgb, rays, hostFocal, tofGridS4 = null, hasTof = false)
            for (i in zeroOut.depthM.indices) {
                val d = abs(zeroOut.depthM[i] - pureGlassOut.depthM[i])
                if (d > maxDepthDiff) maxDepthDiff = d
            }
            val gPure = requireNotNull(pureGlassOut.glassProbS16)
            val gZero = requireNotNull(zeroOut.glassProbS16)
            for (i in gPure.indices) {
                val d = abs(gZero[i] - gPure[i])
                if (d > maxGlassDiff) maxGlassDiff = d
                if (!gZero[i].isFinite()) nanCount++
            }
            repeat(3) { rt.infer(glassRgb, rays, hostFocal, hasTof = false) }
            repeat(15) {
                val o = rt.infer(glassRgb, rays, hostFocal, hasTof = false)
                omniExecMs.add(o.executeMs)
            }
        }

        // Feed live S24U Hexagon v75 outputs into WearableRiskArbitrator
        fun extractS4(out: com.linnan.blindassist.vision.egogeom.EgoGeomOutput): Array<FloatArray> {
            val h4 = EgoGeomContracts.S4_HEIGHT; val w4 = EgoGeomContracts.S4_WIDTH; val n4 = h4 * w4
            val dep = FloatArray(n4); val dh = FloatArray(n4); val nx = FloatArray(n4)
            val nd = FloatArray(n4); val nz = FloatArray(n4); val ls = FloatArray(n4)
            val plane = EgoGeomContracts.DEPTH_FLOATS
            for (y in 0 until h4) {
                val sr = (y * 4) * EgoGeomContracts.WIDTH; val dr = y * w4
                for (x in 0 until w4) {
                    val si = sr + x * 4; val di = dr + x
                    dep[di] = out.depthM[si]; dh[di] = out.deltaHM[si]; ls[di] = out.logSigma[si]
                    nx[di] = out.normalGrav[si]; nd[di] = out.normalGrav[plane + si]; nz[di] = out.normalGrav[2 * plane + si]
                }
            }
            return arrayOf(dep, dh, nx, nd, nz, ls)
        }

        val arb = WearableRiskArbitrator(mode = CameraImuFusionMode.ZERO_VIO_CAMERA_IMU)
        val gMaps = extractS4(pureGlassOut)
        arb.step(0f, gMaps[0], gMaps[1], gMaps[2], gMaps[3], gMaps[4], gMaps[5], glassGray, fx, fy, cx, cy, rZup, fwd, pureGlassOut.glassProbS16)
        val resGlass = arb.step(0f, gMaps[0], gMaps[1], gMaps[2], gMaps[3], gMaps[4], gMaps[5], glassGray, fx, fy, cx, cy, rZup, fwd, pureGlassOut.glassProbS16)

        arb.reset()
        val cMaps = extractS4(pureClearOut)
        arb.step(0f, cMaps[0], cMaps[1], cMaps[2], cMaps[3], cMaps[4], cMaps[5], clearGray, fx, fy, cx, cy, rZup, fwd, pureClearOut.glassProbS16)
        val resClear = arb.step(0f, cMaps[0], cMaps[1], cMaps[2], cMaps[3], cMaps[4], cMaps[5], clearGray, fx, fy, cx, cy, rZup, fwd, pureClearOut.glassProbS16)

        pureExecMs.sort(); omniExecMs.sort()
        val rep = JSONObject()
            .put("pure_rgb_g2_median_ms", pureExecMs[pureExecMs.size / 2])
            .put("omni_g2_median_ms", omniExecMs[omniExecMs.size / 2])
            .put("nan_count", nanCount)
            .put("zero_interference_depth_m", maxDepthDiff.toDouble())
            .put("zero_interference_glass", maxGlassDiff.toDouble())
            .put("glass_door_corr_mean", resGlass.glass.corridorMeanProb.toDouble())
            .put("glass_door_tier", resGlass.glass.tier.name)
            .put("glass_door_channel", resGlass.primaryChannel.name)
            .put("clear_rain_corr_mean", resClear.glass.corridorMeanProb.toDouble())
            .put("clear_rain_tier", resClear.glass.tier.name)
        File(dir, "egogeom_v41_g2_android_device_report.json").writeText(rep.toString(2))

        assertEquals("Expected 0 NaNs", 0, nanCount)
        assertEquals("Expected 0.0 m depth zero-interference", 0.0f, maxDepthDiff, 1e-6f)
        assertEquals("Expected 0.0 glass zero-interference", 0.0f, maxGlassDiff, 1e-6f)
        assertEquals("Expected glass French doors to trigger HIGH tier", GlassAlertTier.HIGH, resGlass.glass.tier)
        assertEquals("Expected GLASS_HAZARD primary channel", WearableHazardChannel.GLASS_HAZARD, resGlass.primaryChannel)
        assertEquals("Expected rainy outdoor walkway to stay NONE", GlassAlertTier.NONE, resClear.glass.tier)
        assertTrue("Expected Pure-RGB+G2 Android median < 30 ms, got ${pureExecMs[pureExecMs.size / 2]}", pureExecMs[pureExecMs.size / 2] < 30.0)
    }
}
