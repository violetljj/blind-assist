package com.linnan.blindassist.hftf

import android.Manifest
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.vision.LatestOnlySidecar
import java.io.File
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.Collections
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraXLatestOnlyDepthDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun realYuvLatestOnlyCachedQnn() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val cachedDlc = File(requireNotNull(arguments.getString("cachedDlcPath")))
        val durationSeconds = arguments.getString("durationSeconds")?.toInt() ?: 20
        val stressSeconds = arguments.getString("stressSeconds")?.toInt() ?: 5
        val depthPeriodMs = arguments.getString("depthPeriodMs")?.toLong() ?: 500L
        val ttlMs = arguments.getString("ttlMs")?.toLong() ?: 750L
        require(durationSeconds >= 12 && stressSeconds in 3 until durationSeconds)
        require(depthPeriodMs in 250L..2_000L && ttlMs >= depthPeriodMs)
        assertTrue(cachedDlc.isFile)

        val provider = ProcessCameraProvider.getInstance(context).get(10, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val depthExecutor = Executors.newSingleThreadExecutor()
        val power = context.getSystemService(PowerManager::class.java)
        val pool = YuvFramePool(3, WIDTH, HEIGHT)
        val framesSeen = AtomicInteger()
        val imageClosed = AtomicInteger()
        val invalidFrames = AtomicInteger()
        val noBufferDrops = AtomicInteger()
        val throttled = AtomicInteger()
        val submitted = AtomicInteger()
        val stressSubmitted = AtomicInteger()
        val pacedSubmitted = AtomicInteger()
        val processed = AtomicInteger()
        val replaced = AtomicInteger()
        val fresh = AtomicInteger()
        val thermalFailClosed = AtomicInteger()
        val running = AtomicInteger()
        val maxRunning = AtomicInteger()
        val failures = Collections.synchronizedList(mutableListOf<String>())
        val copyLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val executeLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val freshAges = Collections.synchronizedList(mutableListOf<Double>())
        val rotations = Collections.synchronizedSet(mutableSetOf<Int>())
        val dimensions = Collections.synchronizedSet(mutableSetOf<String>())
        val lastPacedAt = AtomicLong(Long.MIN_VALUE)
        val startedAt = SystemClock.elapsedRealtimeNanos()
        val stressNanos = TimeUnit.SECONDS.toNanos(stressSeconds.toLong())
        val durationNanos = TimeUnit.SECONDS.toNanos(durationSeconds.toLong())
        val periodNanos = TimeUnit.MILLISECONDS.toNanos(depthPeriodMs)

        val nativeLibraryDir = arguments.getString("qnnRuntimeDir")
            ?: instrumentation.context.applicationInfo.nativeLibraryDir
        val runtime = Dav2QnnCachedContext(cachedDlc.absolutePath, nativeLibraryDir)
        val preprocessor = Dav2NativePreprocessor()
        val converter = Dav2Yuv420RgbConverter()
        val sidecar = LatestOnlySidecar<OwnedYuv420Frame, DepthReceipt>(
            executor = depthExecutor,
            maxResultAgeNanos = TimeUnit.MILLISECONDS.toNanos(ttlMs),
            process = { frame ->
                frame.started = true
                val active = running.incrementAndGet()
                maxRunning.accumulateAndGet(active, ::maxOf)
                try {
                    if (power.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) {
                        thermalFailClosed.incrementAndGet()
                        throw IllegalStateException("thermal fail-closed status=${power.currentThermalStatus}")
                    }
                    val start = SystemClock.elapsedRealtimeNanos()
                    val rgb = converter.convert(frame)
                    val output = runtime.execute(preprocessor.preprocessFp16(rgb))
                    val elapsed = elapsedMs(start)
                    executeLatencies += elapsed
                    processed.incrementAndGet()
                    DepthReceipt(frame.stage, frame.sensorTimestampNanos, elapsed,
                        output.asShortBuffer().get(0).toInt() and 0xffff)
                } finally {
                    running.decrementAndGet()
                }
            },
            onFreshResult = { result ->
                fresh.incrementAndGet()
                freshAges += result.ageNanos / 1_000_000.0
            },
            onFailure = { failure -> failures += "${failure.javaClass.simpleName}: ${failure.message}" },
            nowNanos = SystemClock::elapsedRealtimeNanos,
        )

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                ResolutionStrategy(Size(WIDTH, HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
            ).build())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
        analysis.setAnalyzer(analyzerExecutor) { image ->
            framesSeen.incrementAndGet()
            val receivedAt = SystemClock.elapsedRealtimeNanos()
            try {
                dimensions += "${image.width}x${image.height}"
                rotations += image.imageInfo.rotationDegrees
                if (receivedAt - startedAt >= durationNanos) return@setAnalyzer
                if (image.width != WIDTH || image.height != HEIGHT || image.planes.size != 3) {
                    invalidFrames.incrementAndGet()
                    return@setAnalyzer
                }
                val stage = if (receivedAt - startedAt < stressNanos) "stress" else "paced"
                if (stage == "paced" && !claimPacedSlot(lastPacedAt, receivedAt, periodNanos)) {
                    throttled.incrementAndGet()
                    return@setAnalyzer
                }
                val slot = pool.acquire()
                if (slot == null) {
                    noBufferDrops.incrementAndGet()
                    return@setAnalyzer
                }
                val copyStart = SystemClock.elapsedRealtimeNanos()
                try {
                    copyPlane(image.planes[0], slot.y, WIDTH, HEIGHT)
                    copyPlane(image.planes[1], slot.u, WIDTH / 2, HEIGHT / 2)
                    copyPlane(image.planes[2], slot.v, WIDTH / 2, HEIGHT / 2)
                    slot.width = WIDTH
                    slot.height = HEIGHT
                    slot.rotationDegrees = image.imageInfo.rotationDegrees
                    slot.sensorTimestampNanos = image.imageInfo.timestamp
                    slot.receivedAtNanos = receivedAt
                    slot.stage = stage
                    copyLatencies += elapsedMs(copyStart)
                    submitted.incrementAndGet()
                    if (stage == "stress") stressSubmitted.incrementAndGet() else pacedSubmitted.incrementAndGet()
                    sidecar.submit(slot, receivedAt)
                } catch (failure: Throwable) {
                    slot.close()
                    throw failure
                }
            } catch (failure: Throwable) {
                failures += "analyzer ${failure.javaClass.simpleName}: ${failure.message}"
            } finally {
                image.close()
                imageClosed.incrementAndGet()
            }
        }

        val thermalBefore = power.currentThermalStatus
        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            val deadline = SystemClock.elapsedRealtime() + TimeUnit.SECONDS.toMillis(durationSeconds.toLong() + 15)
            while (SystemClock.elapsedRealtimeNanos() - startedAt < durationNanos &&
                SystemClock.elapsedRealtime() < deadline) Thread.sleep(100)
        } finally {
            instrumentation.runOnMainSync { provider.unbindAll(); owner.destroy() }
            analysis.clearAnalyzer()
            analyzerExecutor.shutdown()
            analyzerExecutor.awaitTermination(5, TimeUnit.SECONDS)
            sidecar.close()
            depthExecutor.shutdown()
            depthExecutor.awaitTermination(15, TimeUnit.SECONDS)
            converter.close()
            preprocessor.close()
            runtime.close()
        }

        replaced.set(pool.replacedCount.get())
        val stale = (processed.get() - fresh.get()).coerceAtLeast(0)
        val report = JSONObject()
            .put("schema", "blindassist_camerax_latest_only_r0")
            .put("contract", JSONObject()
                .put("camera_format", "YUV_420_888")
                .put("camera_resolution", "640x480")
                .put("rotation", "clockwise imageInfo.rotationDegrees")
                .put("crop", "center 4:3 after rotation")
                .put("camera_resize", "OpenCV INTER_LINEAR to 640x480 RGB")
                .put("tensor", "frozen OpenCV cubic normalize NCHW FP16 1x3x518x686")
                .put("backpressure", "CameraX KEEP_ONLY_LATEST + one running/one replaceable pending")
                .put("depth_period_ms", depthPeriodMs)
                .put("result_ttl_ms", ttlMs))
            .put("duration_seconds", durationSeconds)
            .put("stress_seconds", stressSeconds)
            .put("frames_seen", framesSeen.get())
            .put("image_proxy_closed", imageClosed.get())
            .put("dimensions", JSONArray(dimensions.toList()))
            .put("rotations_degrees", JSONArray(rotations.toList().sorted()))
            .put("invalid_frames", invalidFrames.get())
            .put("no_buffer_drops", noBufferDrops.get())
            .put("throttled", throttled.get())
            .put("submitted", submitted.get())
            .put("stress_submitted", stressSubmitted.get())
            .put("paced_submitted", pacedSubmitted.get())
            .put("processed", processed.get())
            .put("pending_replaced", replaced.get())
            .put("fresh_results", fresh.get())
            .put("stale_results", stale)
            .put("max_concurrent_depth_tasks", maxRunning.get())
            .put("pool_available_after_close", pool.available())
            .put("thermal_status_before", thermalBefore)
            .put("thermal_status_after", power.currentThermalStatus)
            .put("thermal_fail_closed", thermalFailClosed.get())
            .put("yuv_copy_ms", latencyJson(copyLatencies))
            .put("yuv_to_fp16_plus_qnn_ms", latencyJson(executeLatencies))
            .put("fresh_result_age_ms", latencyJson(freshAges))
            .put("failures", JSONArray(failures.toList()))

        val gateFailures = mutableListOf<String>()
        if (framesSeen.get() < durationSeconds * 5) gateFailures += "camera frame rate below 5 fps"
        if (imageClosed.get() != framesSeen.get()) gateFailures += "ImageProxy leak"
        if (dimensions != setOf("640x480")) gateFailures += "unexpected camera dimensions: $dimensions"
        if (stressSubmitted.get() < 10) gateFailures += "stress arm did not submit enough frames"
        if (pacedSubmitted.get() < (durationSeconds - stressSeconds)) gateFailures += "paced arm below 1 Hz"
        if (maxRunning.get() != 1) gateFailures += "depth concurrency was ${maxRunning.get()}"
        if (pool.available() != 3) gateFailures += "owned YUV slot leak"
        if (fresh.get() == 0 || freshAges.any { it > ttlMs }) gateFailures += "TTL freshness contract failed"
        if (thermalFailClosed.get() != 0) gateFailures += "device reached severe thermal status"
        if (failures.isNotEmpty()) gateFailures += failures
        report.put("gate_pass", gateFailures.isEmpty()).put("gate_failures", JSONArray(gateFailures))
        File(context.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertTrue(gateFailures.joinToString("\n"), gateFailures.isEmpty())
    }

    private fun copyPlane(plane: ImageProxy.PlaneProxy, target: ByteArray, width: Int, height: Int) {
        val source = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (row in 0 until height) {
            val sourceStart = row * rowStride
            val targetStart = row * width
            if (pixelStride == 1) {
                source.position(sourceStart)
                source.get(target, targetStart, width)
            } else {
                for (column in 0 until width) target[targetStart + column] = source.get(sourceStart + column * pixelStride)
            }
        }
    }

    private fun claimPacedSlot(last: AtomicLong, now: Long, period: Long): Boolean {
        while (true) {
            val previous = last.get()
            if (previous != Long.MIN_VALUE && now - previous < period) return false
            if (last.compareAndSet(previous, now)) return true
        }
    }

    private fun latencyJson(values: List<Double>): JSONObject {
        if (values.isEmpty()) return JSONObject().put("count", 0)
        val sorted = values.sorted()
        fun percentile(q: Double): Double {
            val position = q * (sorted.size - 1)
            val lower = position.toInt()
            val upper = minOf(lower + 1, sorted.lastIndex)
            return sorted[lower] * (1 - position + lower) + sorted[upper] * (position - lower)
        }
        return JSONObject().put("count", values.size).put("p50", percentile(.5))
            .put("p95", percentile(.95)).put("maximum", sorted.last()).put("mean", values.average())
    }

    private fun elapsedMs(startNanos: Long) =
        (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0

    private data class DepthReceipt(val stage: String, val sensorTimestampNanos: Long,
        val executeMs: Double, val checksum: Int)

    private class YuvFramePool(capacity: Int, width: Int, height: Int) {
        private val available = ArrayDeque<OwnedYuv420Frame>()
        val replacedCount = AtomicInteger()
        init { repeat(capacity) { available.add(OwnedYuv420Frame(width, height, ::release)) } }
        @Synchronized fun acquire(): OwnedYuv420Frame? =
            if (available.isEmpty()) null else available.removeFirst().lease()
        @Synchronized private fun release(frame: OwnedYuv420Frame) {
            if (!frame.started) replacedCount.incrementAndGet()
            available.addLast(frame)
        }
        @Synchronized fun available(): Int = available.size
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
        const val REPORT_KEY = "camerax_latest_only_r0_report"
        const val REPORT_FILE = "camerax-latest-only-r0.json"
    }
}
