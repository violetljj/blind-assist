package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** UniDepth-V2-S Atom model on the HTP against the research path (host/unidepth_atom_export_20261118.py fixtures:
 * one real Atom frame, its model-resolution inputs, research model-resolution points and research 640x480 depth).
 * Fixtures and model are supplied explicitly (instrumentation argument udFixtureRoot); nothing is downloaded. */
@RunWith(AndroidJUnit4::class)
class UniDepthSnpeRuntimeDeviceTest {
    private val n = 3 * UniDepthSnpeRuntime.H * UniDepthSnpeRuntime.W

    private fun relStats(a: FloatArray, ref: FloatArray): Pair<Double, Double> {
        val rel = DoubleArray(a.size) { abs(a[it] - ref[it]).toDouble() / maxOf(ref[it].toDouble(), 1e-3) }.sorted()
        return rel[rel.size / 2] to rel[(rel.size * 99) / 100]
    }

    @Test
    fun htpMatchesResearchPathOnRealAtomFrame() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(InstrumentationRegistry.getArguments().getString("udFixtureRoot")
            ?: File(context.filesDir, "unidepth-smoke").absolutePath)
        val report = JSONObject().put("evidence", "one real Atom frame (capture 2026-10-13), research path fixtures")
        UniDepthSnpeRuntime(context.applicationContext as Application, File(dir, UniDepthRuntimeFactory.MODEL),
            File(dir, UniDepthRuntimeFactory.RAYS)).use { runtime ->
            // 1. Same model-resolution input as the research path.
            val input = UniDepthSnpeRuntime.readRaw(File(dir, "input_rgb_658x490.raw"), n)
            val pts = runtime.executeNchw(input)
            val refPts = UniDepthSnpeRuntime.readRaw(File(dir, "torch_pts_658x490.raw"), n)
            val z = pts.copyOfRange(2 * n / 3, n); val refZ = refPts.copyOfRange(2 * n / 3, n)
            val (modelMedian, modelP99) = relStats(z, refZ)
            // 2. App preprocessing from the decoded upright PNG.
            val bitmap = requireNotNull(BitmapFactory.decodeFile(File(dir, "atom_frame_upright.png").absolutePath))
            val pre = UniDepthSnpeRuntime.preprocessNchw(bitmap)
            val preMaxAbs = pre.indices.maxOf { abs(pre[it] - input[it]).toDouble() }
            // 3. Whole app path to 640x480 against the research depth.
            val depth = runtime.estimate(bitmap)
            val refDepth = UniDepthSnpeRuntime.readRaw(File(dir, "research_depth_640x480.raw"), 480 * 640)
            val (appMedian, appP99) = relStats(depth, refDepth)
            // 4. Timing of the app path (after warm-up).
            repeat(3) { runtime.estimate(bitmap) }
            val preMs = ArrayList<Double>(); val runMs = ArrayList<Double>(); val postMs = ArrayList<Double>()
            repeat(10) {
                var t = SystemClock.elapsedRealtimeNanos()
                val x = UniDepthSnpeRuntime.preprocessNchw(bitmap); preMs.add((SystemClock.elapsedRealtimeNanos() - t) / 1e6)
                t = SystemClock.elapsedRealtimeNanos()
                val p = runtime.executeNchw(x); runMs.add((SystemClock.elapsedRealtimeNanos() - t) / 1e6)
                t = SystemClock.elapsedRealtimeNanos()
                UniDepthSnpeRuntime.postprocess(p); postMs.add((SystemClock.elapsedRealtimeNanos() - t) / 1e6)
            }
            bitmap.recycle()
            fun med(v: List<Double>) = v.sorted()[v.size / 2]
            report.put("model_z_rel_median", modelMedian).put("model_z_rel_p99", modelP99)
                .put("preprocess_max_abs_diff", preMaxAbs)
                .put("app_depth_rel_median", appMedian).put("app_depth_rel_p99", appP99)
                .put("preprocess_ms", JSONArray(preMs)).put("execute_ms", JSONArray(runMs)).put("postprocess_ms", JSONArray(postMs))
                .put("preprocess_median_ms", med(preMs)).put("execute_median_ms", med(runMs)).put("postprocess_median_ms", med(postMs))
            File(dir, "unidepth_device_check.json").writeText(report.toString(2))
            assertTrue("model z median rel $modelMedian", modelMedian < .02)
            assertTrue("preprocess max abs $preMaxAbs", preMaxAbs < 1e-3)
            assertTrue("app depth median rel $appMedian", appMedian < .02)
        }
    }
}
