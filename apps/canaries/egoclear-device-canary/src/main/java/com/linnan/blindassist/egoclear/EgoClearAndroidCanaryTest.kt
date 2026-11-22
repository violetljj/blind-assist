package com.linnan.blindassist.egoclear

import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EgoClearAndroidCanaryTest {

    @Test
    fun goldenParityAndMultiBackendBenchmark() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        val dir = File(args.getString("modelDir") ?: DEFAULT_MODEL_DIR)
        val encFile = File(dir, args.getString("encoderName") ?: "egoclear_v66_cleankd_encoder_fp16.onnx")
        val stpFile = File(dir, args.getString("stepName") ?: "egoclear_v66_cleankd_step_fp16.onnx")
        val rgbBin = File(dir, "rgbs_5x3x294x518_f32.bin")
        val goldenJsonFile = File(dir, "golden_meta.json")
        val sustainedSec = args.getString("sustainedSec")?.toIntOrNull() ?: 60
        val skipNnapi = args.getString("skipNnapi")?.toBoolean() ?: false
        val parityTol = args.getString("parityTol")?.toDoubleOrNull() ?: 1e-3
        val reportName = args.getString("reportName") ?: "egoclear_v66_android_report.json"

        assertTrue("Missing encoder: ${encFile.absolutePath}", encFile.isFile)
        assertTrue("Missing step: ${stpFile.absolutePath}", stpFile.isFile)
        assertTrue("Missing golden rgb bin: ${rgbBin.absolutePath}", rgbBin.isFile)
        assertTrue("Missing golden json: ${goldenJsonFile.absolutePath}", goldenJsonFile.isFile)

        val goldenMeta = JSONObject(goldenJsonFile.readText())
        val nFrames = goldenMeta.getInt("T")
        val kArr = FloatArray(4) { goldenMeta.getJSONArray("K").getDouble(it).toFloat() }
        val goldenCorr = Array(nFrames) { t ->
            val a = goldenMeta.getJSONArray("corr_probs").getJSONArray(t)
            FloatArray(a.length()) { a.getDouble(it).toFloat() }
        }
        val rgbFrames = loadFloatFrames(rgbBin, nFrames, EgoClearStreamingEngine.RGB_FLOATS)
        val rLevel = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
        val fwdX = floatArrayOf(1f, 0f, 0f)

        val report = JSONObject()
            .put("schema", "egoclear_v66_android_device_canary_r0")
            .put("device", deviceJson())
            .put("encoder_bytes", encFile.length())
            .put("step_bytes", stpFile.length())
            .put("total_mb", (encFile.length() + stpFile.length()) / 1_000_000.0)
            .put("memory_baseline", memoryJson())
            .put("battery_temp_c_start", batteryTempC())
            .put("thermal_status_start", thermalStatus())

        val backendsReport = JSONObject()
        var bestBackend: EgoClearStreamingEngine.Backend? = null
        var bestP50Ms = Double.POSITIVE_INFINITY

        for (backend in EgoClearStreamingEngine.Backend.values()) {
            if (backend == EgoClearStreamingEngine.Backend.HTP_V75_NPU) continue
            if (skipNnapi && backend == EgoClearStreamingEngine.Backend.NNAPI) continue
            val bObj = JSONObject()
            try {
                EgoClearStreamingEngine(encFile, stpFile, backend, EgoClearStreamingEngine.Backend.CPU_4T).use { engine ->
                    bObj.put("status", "OK")
                    bObj.put("encoder_load_ms", round2(engine.encoderLoadMs))
                    bObj.put("step_load_ms", round2(engine.stepLoadMs))
                    bObj.put("memory_after_load", memoryJson())

                    // 1. Exact 5-frame streaming golden parity check
                    engine.resetState()
                    var maxAbsErr = 0.0
                    var monoViol = 0
                    for (t in 0 until nFrames) {
                        val tPos = floatArrayOf(0.25f * t, 0f, 1.5f)
                        val ds = if (t == 0) 0f else 0.25f
                        val out = engine.step(rgbFrames[t], kArr, rLevel, tPos, fwdX, ds)
                        if (!out.monotonicOk) monoViol++
                        for (k in out.corrProbsMonotonic.indices) {
                            val err = abs(out.corrProbsMonotonic[k] - goldenCorr[t][k]).toDouble()
                            if (err > maxAbsErr) maxAbsErr = err
                        }
                    }
                    bObj.put("golden_max_abs_err", maxAbsErr)
                    bObj.put("golden_mono_violations", monoViol)

                    // 2. Warmup (5 frames) + Timed 25-frame streaming latency sweep
                    engine.resetState()
                    repeat(5) { idx ->
                        val f = idx % nFrames
                        engine.step(rgbFrames[f], kArr, rLevel, floatArrayOf(0.25f * idx, 0f, 1.5f), fwdX, if (idx == 0) 0f else 0.25f)
                    }
                    val encMs = ArrayList<Double>(25)
                    val stpMs = ArrayList<Double>(25)
                    val totMs = ArrayList<Double>(25)
                    var timedMonoViol = 0
                    for (idx in 0 until 25) {
                        val f = idx % nFrames
                        val out = engine.step(rgbFrames[f], kArr, rLevel, floatArrayOf(0.25f * (idx + 5), 0f, 1.5f), fwdX, 0.25f)
                        encMs += out.encoderMs
                        stpMs += out.stepMs
                        totMs += out.totalMs
                        if (!out.monotonicOk) timedMonoViol++
                    }
                    val totStats = latencyJson(totMs)
                    val p50 = totStats.getDouble("p50")
                    bObj.put("encoder_ms", latencyJson(encMs))
                    bObj.put("step_ms", latencyJson(stpMs))
                    bObj.put("total_ms", totStats)
                    bObj.put("effective_fps", round2(1000.0 / totMs.average()))
                    bObj.put("timed_mono_violations", timedMonoViol)
                    bObj.put("memory_after_runs", memoryJson())
                    bObj.put("thermal_status", thermalStatus())

                    if (maxAbsErr <= parityTol && monoViol == 0 && p50 < bestP50Ms) {
                        bestP50Ms = p50
                        bestBackend = backend
                    }
                }
            } catch (t: Throwable) {
                bObj.put("status", "ERROR")
                bObj.put("error", "${t.javaClass.simpleName}: ${t.message}")
            }
            backendsReport.put(backend.tag, bObj)
        }
        report.put("backends", backendsReport)

        val chosen = requireNotNull(bestBackend) { "No backend passed parity! $backendsReport" }
        report.put("best_backend", chosen.tag)

        // 3. Sustained 5 Hz real-time streaming & thermal stability test (60s @ 200ms cadence)
        val sustainedObj = JSONObject()
            .put("backend", chosen.tag)
            .put("target_sec", sustainedSec)
            .put("target_cadence_ms", 200.0)
            .put("battery_temp_c_before", batteryTempC())
            .put("thermal_status_before", thermalStatus())

        EgoClearStreamingEngine(encFile, stpFile, chosen, EgoClearStreamingEngine.Backend.CPU_4T).use { engine ->
            engine.resetState()
            // Warmup 3 frames
            repeat(3) { idx ->
                engine.step(rgbFrames[idx % nFrames], kArr, rLevel, floatArrayOf(0.25f * idx, 0f, 1.5f), fwdX, if (idx == 0) 0f else 0.25f)
            }
            val startMs = SystemClock.elapsedRealtime()
            val targetMs = sustainedSec * 1000L
            var nextSampleMs = startMs
            val samples = JSONArray()
            val encAll = ArrayList<Double>()
            val stpAll = ArrayList<Double>()
            val totAll = ArrayList<Double>()
            var frameIdx = 0
            var monoViol = 0
            var deadlineMisses = 0

            while (SystemClock.elapsedRealtime() - startMs < targetMs) {
                val frameStartMs = SystemClock.elapsedRealtime()
                val f = frameIdx % nFrames
                val out = engine.step(rgbFrames[f], kArr, rLevel, floatArrayOf(0.25f * frameIdx, 0f, 1.5f), fwdX, 0.25f)
                encAll += out.encoderMs
                stpAll += out.stepMs
                totAll += out.totalMs
                if (!out.monotonicOk) monoViol++
                if (out.totalMs > 200.0) deadlineMisses++
                frameIdx++

                val now = SystemClock.elapsedRealtime()
                if (now >= nextSampleMs) {
                    samples.put(
                        JSONObject()
                            .put("elapsed_sec", round2((now - startMs) / 1000.0))
                            .put("frames_done", frameIdx)
                            .put("recent_total_ms", round2(out.totalMs))
                            .put("pss_mb", round2(Debug.getPss() / 1024.0))
                            .put("native_heap_mb", round2(Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)))
                            .put("battery_temp_c", batteryTempC())
                            .put("thermal_status", thermalStatus()),
                    )
                    nextSampleMs += 10_000L
                }
                // Pace to realistic 5 Hz (200ms period) if faster than 200ms
                val elapsedFrame = SystemClock.elapsedRealtime() - frameStartMs
                val sleepMs = 200L - elapsedFrame
                if (sleepMs > 2L) SystemClock.sleep(sleepMs)
            }

            val actualSec = (SystemClock.elapsedRealtime() - startMs) / 1000.0
            val firstHalfP50 = percentile(totAll.take(totAll.size / 2).sorted(), 0.50)
            val secondHalfP50 = percentile(totAll.drop(totAll.size / 2).sorted(), 0.50)
            sustainedObj
                .put("actual_sec", round2(actualSec))
                .put("frames_completed", frameIdx)
                .put("achieved_hz", round2(frameIdx / actualSec))
                .put("deadline_200ms_misses", deadlineMisses)
                .put("mono_violations", monoViol)
                .put("encoder_ms", latencyJson(encAll))
                .put("step_ms", latencyJson(stpAll))
                .put("total_ms", latencyJson(totAll))
                .put("first_half_p50_ms", round2(firstHalfP50))
                .put("second_half_p50_ms", round2(secondHalfP50))
                .put("thermal_drift_pct", round2(100.0 * (secondHalfP50 - firstHalfP50) / firstHalfP50))
                .put("battery_temp_c_after", batteryTempC())
                .put("thermal_status_after", thermalStatus())
                .put("memory_after", memoryJson())
                .put("samples_10s", samples)
        }
        report.put("sustained_5hz", sustainedObj)

        // Write report JSON to world-readable target cache & emit via instrumentation status
        val outFile = File(inst.targetContext.externalCacheDir ?: inst.targetContext.cacheDir, reportName)
        outFile.writeText(report.toString(2))
        report.put("report_file", outFile.absolutePath)

        // Assert strict golden parity & zero monotonicity violations on best backend
        val bestStats = backendsReport.getJSONObject(chosen.tag)
        assertTrue("Parity error too high: ${bestStats.getDouble("golden_max_abs_err")}", bestStats.getDouble("golden_max_abs_err") <= parityTol)
        assertTrue("Monotonicity violated in sustained test", sustainedObj.getInt("mono_violations") == 0)

        inst.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    @Test
    fun htpV75NpuEndToEndBenchmark() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        val dir = File(args.getString("modelDir") ?: DEFAULT_MODEL_DIR)
        val dlcFile = File(dir, args.getString("dlcName") ?: "egoclear_v66_encoder_htp_v75_fp16.dlc")
        val stpFile = File(dir, args.getString("stepName") ?: "egoclear_v66_cleankd_step_fp16.onnx")
        val rgbBin = File(dir, "rgbs_5x3x294x518_f32.bin")
        val goldenJsonFile = File(dir, "golden_meta.json")
        val sustainedSec = args.getString("sustainedSec")?.toIntOrNull() ?: 60

        assertTrue("Missing HTP DLC: ${dlcFile.absolutePath}", dlcFile.isFile)
        assertTrue("Missing step ONNX: ${stpFile.absolutePath}", stpFile.isFile)
        assertTrue("Missing golden rgb bin", rgbBin.isFile)
        assertTrue("Missing golden json", goldenJsonFile.isFile)

        val goldenMeta = JSONObject(goldenJsonFile.readText())
        val nFrames = goldenMeta.getInt("T")
        val kArr = FloatArray(4) { goldenMeta.getJSONArray("K").getDouble(it).toFloat() }
        val goldenCorr = Array(nFrames) { t ->
            val a = goldenMeta.getJSONArray("corr_probs").getJSONArray(t)
            FloatArray(a.length()) { a.getDouble(it).toFloat() }
        }
        val rgbFrames = loadFloatFrames(rgbBin, nFrames, EgoClearStreamingEngine.RGB_FLOATS)
        val rLevel = floatArrayOf(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f)
        val fwdX = floatArrayOf(1f, 0f, 0f)

        val report = JSONObject()
            .put("schema", "egoclear_v66_android_htp_v75_npu_r0")
            .put("device", deviceJson())
            .put("dlc_bytes", dlcFile.length())
            .put("step_bytes", stpFile.length())
            .put("memory_baseline", memoryJson())
            .put("battery_temp_c_start", batteryTempC())
            .put("thermal_status_start", thermalStatus())

        EgoClearSnpeEncoder(inst.context, inst.targetContext, dlcFile).use { npuEncoder ->
            EgoClearStreamingEngine(
                encoderModel = null,
                stepModel = stpFile,
                backend = EgoClearStreamingEngine.Backend.HTP_V75_NPU,
                stepBackend = EgoClearStreamingEngine.Backend.CPU_4T,
                snpeEncoder = npuEncoder,
            ).use { engine ->
                report.put("encoder_npu_load_ms", round2(engine.encoderLoadMs))
                report.put("step_cpu_load_ms", round2(engine.stepLoadMs))
                report.put("memory_after_load", memoryJson())

                // 1. 5-frame streaming golden parity vs host FP16 ONNX
                engine.resetState()
                var maxAbsErr = 0.0
                var sumAbsErr = 0.0
                var countErr = 0
                var monoViol = 0
                for (t in 0 until nFrames) {
                    val out = engine.step(rgbFrames[t], kArr, rLevel, floatArrayOf(0.25f * t, 0f, 1.5f), fwdX, if (t == 0) 0f else 0.25f)
                    if (!out.monotonicOk) monoViol++
                    for (k in out.corrProbsMonotonic.indices) {
                        val err = abs(out.corrProbsMonotonic[k] - goldenCorr[t][k]).toDouble()
                        if (err > maxAbsErr) maxAbsErr = err
                        sumAbsErr += err
                        countErr++
                    }
                }
                report.put("golden_mae", sumAbsErr / countErr)
                report.put("golden_max_abs_err", maxAbsErr)
                report.put("golden_mono_violations", monoViol)

                // 2. Burst speed benchmark: 10 warmup + 50 timed streaming frames (unthrottled max throughput)
                engine.resetState()
                repeat(10) { idx ->
                    engine.step(rgbFrames[idx % nFrames], kArr, rLevel, floatArrayOf(0.25f * idx, 0f, 1.5f), fwdX, if (idx == 0) 0f else 0.25f)
                }
                val encBurst = ArrayList<Double>(50)
                val stpBurst = ArrayList<Double>(50)
                val totBurst = ArrayList<Double>(50)
                var burstMonoViol = 0
                for (idx in 0 until 50) {
                    val out = engine.step(rgbFrames[idx % nFrames], kArr, rLevel, floatArrayOf(0.25f * (idx + 10), 0f, 1.5f), fwdX, 0.25f)
                    encBurst += out.encoderMs
                    stpBurst += out.stepMs
                    totBurst += out.totalMs
                    if (!out.monotonicOk) burstMonoViol++
                }
                report.put(
                    "burst_50f",
                    JSONObject()
                        .put("encoder_npu_ms", latencyJson(encBurst))
                        .put("step_cpu_ms", latencyJson(stpBurst))
                        .put("total_ms", latencyJson(totBurst))
                        .put("max_fps", round2(1000.0 / totBurst.average()))
                        .put("mono_violations", burstMonoViol)
                        .put("memory_after_burst", memoryJson())
                        .put("battery_temp_c", batteryTempC())
                        .put("thermal_status", thermalStatus()),
                )

                // 3. Real-time 5 Hz (200 ms cadence) sustained 60-second streaming test!
                engine.resetState()
                val startMs = SystemClock.elapsedRealtime()
                val targetMs = sustainedSec * 1000L
                var nextSampleMs = startMs
                val samples = JSONArray()
                val encSust = ArrayList<Double>()
                val stpSust = ArrayList<Double>()
                val totSust = ArrayList<Double>()
                var frameIdx = 0
                var sustMonoViol = 0
                var deadlineMisses = 0
                val tempBefore = batteryTempC()
                val thermBefore = thermalStatus()

                while (SystemClock.elapsedRealtime() - startMs < targetMs) {
                    val fStart = SystemClock.elapsedRealtime()
                    val out = engine.step(rgbFrames[frameIdx % nFrames], kArr, rLevel, floatArrayOf(0.25f * frameIdx, 0f, 1.5f), fwdX, if (frameIdx == 0) 0f else 0.25f)
                    encSust += out.encoderMs
                    stpSust += out.stepMs
                    totSust += out.totalMs
                    if (!out.monotonicOk) sustMonoViol++
                    if (out.totalMs > 200.0) deadlineMisses++
                    frameIdx++

                    val now = SystemClock.elapsedRealtime()
                    if (now >= nextSampleMs) {
                        samples.put(
                            JSONObject()
                                .put("elapsed_sec", round2((now - startMs) / 1000.0))
                                .put("frames_done", frameIdx)
                                .put("recent_enc_ms", round2(out.encoderMs))
                                .put("recent_step_ms", round2(out.stepMs))
                                .put("recent_total_ms", round2(out.totalMs))
                                .put("pss_mb", round2(Debug.getPss() / 1024.0))
                                .put("native_heap_mb", round2(Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)))
                                .put("battery_temp_c", batteryTempC())
                                .put("thermal_status", thermalStatus()),
                        )
                        nextSampleMs += 10_000L
                    }
                    val sleepMs = 200L - (SystemClock.elapsedRealtime() - fStart)
                    if (sleepMs > 2L) SystemClock.sleep(sleepMs)
                }
                val actualSec = (SystemClock.elapsedRealtime() - startMs) / 1000.0
                val firstHalf = percentile(totSust.take(totSust.size / 2).sorted(), 0.50)
                val secondHalf = percentile(totSust.drop(totSust.size / 2).sorted(), 0.50)
                report.put(
                    "sustained_5hz",
                    JSONObject()
                        .put("target_sec", sustainedSec)
                        .put("actual_sec", round2(actualSec))
                        .put("frames_completed", frameIdx)
                        .put("achieved_hz", round2(frameIdx / actualSec))
                        .put("deadline_200ms_misses", deadlineMisses)
                        .put("mono_violations", sustMonoViol)
                        .put("encoder_npu_ms", latencyJson(encSust))
                        .put("step_cpu_ms", latencyJson(stpSust))
                        .put("total_ms", latencyJson(totSust))
                        .put("first_half_p50_ms", round2(firstHalf))
                        .put("second_half_p50_ms", round2(secondHalf))
                        .put("thermal_drift_pct", round2(100.0 * (secondHalf - firstHalf) / firstHalf))
                        .put("battery_temp_c_before", tempBefore)
                        .put("battery_temp_c_after", batteryTempC())
                        .put("thermal_status_before", thermBefore)
                        .put("thermal_status_after", thermalStatus())
                        .put("memory_after", memoryJson())
                        .put("samples_10s", samples),
                )
            }
        }

        val outFile = File(inst.targetContext.externalCacheDir ?: inst.targetContext.cacheDir, "egoclear_v66_htp_v75_npu_report.json")
        outFile.writeText(report.toString(2))
        report.put("report_file", outFile.absolutePath)

        assertTrue("NPU monotonicity violated", report.getInt("golden_mono_violations") == 0)
        assertTrue("NPU sustained monotonicity violated", report.getJSONObject("sustained_5hz").getInt("mono_violations") == 0)
        assertTrue("NPU parity MAE too high: ${report.getDouble("golden_mae")}", report.getDouble("golden_mae") < 0.02)

        inst.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun loadFloatFrames(file: File, nFrames: Int, floatsPerFrame: Int): Array<FloatArray> {
        val bytes = file.readBytes()
        val expected = nFrames * floatsPerFrame * 4
        require(bytes.size == expected) { "Expected $expected bytes in ${file.name}, got ${bytes.size}" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return Array(nFrames) {
            FloatArray(floatsPerFrame).also { bb.get(it) }
        }
    }

    private fun latencyJson(values: List<Double>): JSONObject {
        val s = values.sorted()
        return JSONObject()
            .put("mean", round2(values.average()))
            .put("p50", round2(percentile(s, 0.50)))
            .put("p95", round2(percentile(s, 0.95)))
            .put("max", round2(s.last()))
    }

    private fun percentile(sorted: List<Double>, q: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val pos = q * (sorted.size - 1)
        val lo = pos.toInt()
        val hi = minOf(lo + 1, sorted.lastIndex)
        return sorted[lo] * (1.0 - (pos - lo)) + sorted[hi] * (pos - lo)
    }

    private fun memoryJson(): JSONObject {
        val rt = Runtime.getRuntime()
        return JSONObject()
            .put("pss_mb", round2(Debug.getPss() / 1024.0))
            .put("java_heap_mb", round2((rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0)))
            .put("native_heap_mb", round2(Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)))
    }

    private fun deviceJson() = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("device", Build.DEVICE)
        .put("hardware", Build.HARDWARE)
        .put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unknown")
        .put("android_release", Build.VERSION.RELEASE)
        .put("sdk", Build.VERSION.SDK_INT)

    private fun thermalStatus(): Int {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return ctx.getSystemService(PowerManager::class.java).currentThermalStatus
    }

    private fun batteryTempC(): Double {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        return if (tenths > 0) tenths / 10.0 else -1.0
    }

    private fun round2(v: Double): Double = kotlin.math.round(v * 100.0) / 100.0

    companion object {
        const val REPORT_KEY = "egoclear_v66_android_canary_report"
        const val DEFAULT_MODEL_DIR = "/data/local/tmp/egoclear_v66"
    }
}
