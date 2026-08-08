package com.linnan.blindassist.hftf

import android.Manifest
import android.os.Bundle
import android.os.Debug
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
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import com.linnan.blindassist.vision.ExpiringLatestResult
import com.linnan.blindassist.vision.LatestOnlySidecar
import com.linnan.blindassist.vision.PhaseLockedCadenceGate
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
        val includeGeometry = arguments.getString("includeGeometry")?.toBooleanStrictOrNull() ?: false
        val pipelineGeometry = arguments.getString("pipelineGeometry")?.toBooleanStrictOrNull() ?: false
        val phaseLockedCadence = arguments.getString("phaseLockedCadence")?.toBooleanStrictOrNull() ?: false
        val nativeFp16Decode = arguments.getString("nativeFp16Decode")?.toBooleanStrictOrNull() ?: false
        val nativeGeometry = arguments.getString("nativeGeometry")?.toBooleanStrictOrNull() ?: false
        val nativeDirectDepthBridge = arguments.getString("nativeDirectDepthBridge")?.toBooleanStrictOrNull() ?: false
        val directRgbBridge = arguments.getString("directRgbBridge")?.toBooleanStrictOrNull() ?: false
        require(!pipelineGeometry || includeGeometry) { "pipelineGeometry requires includeGeometry" }
        require(!nativeGeometry || includeGeometry) { "nativeGeometry requires includeGeometry" }
        require(!nativeDirectDepthBridge || pipelineGeometry && nativeFp16Decode && nativeGeometry) {
            "nativeDirectDepthBridge requires pipelineGeometry, nativeFp16Decode, and nativeGeometry"
        }
        require(durationSeconds >= 12 && stressSeconds in 3 until durationSeconds)
        require(depthPeriodMs in 200L..2_000L && ttlMs >= depthPeriodMs)
        assertTrue(cachedDlc.isFile)

        val provider = ProcessCameraProvider.getInstance(context).get(10, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val depthExecutor = Executors.newSingleThreadExecutor()
        val geometryExecutor = Executors.newSingleThreadExecutor()
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
        val nonInteractiveObservations = AtomicInteger()
        val maximumThermalStatus = AtomicInteger(power.currentThermalStatus)
        val geometryValid = AtomicInteger()
        val geometryUnknown = AtomicInteger()
        val inferenceProcessed = AtomicInteger()
        val stressInferenceProcessed = AtomicInteger()
        val pacedInferenceProcessed = AtomicInteger()
        val stressProcessed = AtomicInteger()
        val pacedProcessed = AtomicInteger()
        val running = AtomicInteger()
        val maxRunning = AtomicInteger()
        val geometryRunning = AtomicInteger()
        val maxGeometryRunning = AtomicInteger()
        val combinedRunning = AtomicInteger()
        val maxCombinedRunning = AtomicInteger()
        val failures = Collections.synchronizedList(mutableListOf<String>())
        val copyLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val executeLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val decodeAlignLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val fp16DecodeLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val alignResizeLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val directDepthBridgeLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val geometryLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val fullPipelineLatencies = Collections.synchronizedList(mutableListOf<Double>())
        val freshAges = Collections.synchronizedList(mutableListOf<Double>())
        val rotations = Collections.synchronizedSet(mutableSetOf<Int>())
        val dimensions = Collections.synchronizedSet(mutableSetOf<String>())
        val lastPacedAt = AtomicLong(Long.MIN_VALUE)
        val startedAt = SystemClock.elapsedRealtimeNanos()
        val stressNanos = TimeUnit.SECONDS.toNanos(stressSeconds.toLong())
        val durationNanos = TimeUnit.SECONDS.toNanos(durationSeconds.toLong())
        val periodNanos = TimeUnit.MILLISECONDS.toNanos(depthPeriodMs)
        val phaseLockedGate = PhaseLockedCadenceGate(periodNanos)

        fun markInferenceProcessed(stage: String) {
            inferenceProcessed.incrementAndGet()
            if (stage == "stress") stressInferenceProcessed.incrementAndGet()
            else pacedInferenceProcessed.incrementAndGet()
        }

        fun markProcessed(stage: String) {
            processed.incrementAndGet()
            if (stage == "stress") stressProcessed.incrementAndGet()
            else pacedProcessed.incrementAndGet()
        }

        val nativeLibraryDir = arguments.getString("qnnRuntimeDir")
            ?: instrumentation.context.applicationInfo.nativeLibraryDir
        val runtime = Dav2QnnCachedContext(cachedDlc.absolutePath, nativeLibraryDir)
        val preprocessor = Dav2NativePreprocessor()
        val converter = Dav2Yuv420RgbConverter()
        val resultStore = ExpiringLatestResult<DepthReceipt>(TimeUnit.MILLISECONDS.toNanos(ttlMs))
        val runtimeStatsBefore = RuntimeStats.capture()
        val memoryBefore = memoryJson()
        val rawDepth = if (nativeDirectDepthBridge) FloatArray(0) else FloatArray(Dav2PreprocessContract.PLANE)
        val alignedDepth = if (nativeDirectDepthBridge) FloatArray(0) else FloatArray(WIDTH * HEIGHT)
        val geometryPool = AlignedDepthPool(3, WIDTH * HEIGHT, nativeDirectDepthBridge)

        fun evaluateGeometry(depth: FloatArray): Any = if (nativeGeometry) {
            Dav2NativeGeometry.evaluate(
                depth, WIDTH, HEIGHT, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
            )
        } else {
            KnownHeightGroundPipeline.evaluateGeometry(
                depth, WIDTH, HEIGHT, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
            )
        }

        fun evaluateGeometry(work: OwnedAlignedDepth): Any = if (nativeDirectDepthBridge) {
            Dav2NativeGeometry.evaluateDirect(
                checkNotNull(work.directDepth), WIDTH, HEIGHT,
                320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
            )
        } else {
            evaluateGeometry(checkNotNull(work.arrayDepth))
        }

        fun deliverFinal(result: LatestOnlySidecar.Result<DepthReceipt>) {
            fresh.incrementAndGet()
            freshAges += result.ageNanos / 1_000_000.0
            resultStore.update(result.value, result.capturedAtNanos, result.completedAtNanos)
        }

        val geometrySidecar = LatestOnlySidecar<OwnedAlignedDepth, DepthReceipt>(
            executor = geometryExecutor,
            maxResultAgeNanos = TimeUnit.MILLISECONDS.toNanos(ttlMs),
            process = { work ->
                work.started = true
                val activeGeometry = geometryRunning.incrementAndGet()
                maxGeometryRunning.accumulateAndGet(activeGeometry, ::maxOf)
                val activeCombined = combinedRunning.incrementAndGet()
                maxCombinedRunning.accumulateAndGet(activeCombined, ::maxOf)
                try {
                    val geometryStart = SystemClock.elapsedRealtimeNanos()
                    val geometry = evaluateGeometry(work)
                    val geometryStatus = if (geometry is KnownHeightGroundPipeline.Geometry) {
                        geometryValid.incrementAndGet(); "VALID"
                    } else {
                        geometryUnknown.incrementAndGet(); "UNKNOWN"
                    }
                    geometryLatencies += elapsedMs(geometryStart)
                    val fullElapsed = elapsedMs(work.fullStartedAtNanos)
                    fullPipelineLatencies += fullElapsed
                    markProcessed(work.stage)
                    DepthReceipt(
                        work.stage,
                        work.sensorTimestampNanos,
                        fullElapsed,
                        geometryStatus,
                        work.checksum,
                    )
                } finally {
                    geometryRunning.decrementAndGet()
                    combinedRunning.decrementAndGet()
                }
            },
            onFreshResult = ::deliverFinal,
            onFailure = { failure -> failures += "geometry ${failure.javaClass.simpleName}: ${failure.message}" },
            nowNanos = SystemClock::elapsedRealtimeNanos,
        )

        val sidecar = LatestOnlySidecar<OwnedYuv420Frame, InferenceResult>(
            executor = depthExecutor,
            maxResultAgeNanos = TimeUnit.MILLISECONDS.toNanos(ttlMs),
            process = { frame ->
                frame.started = true
                val active = running.incrementAndGet()
                maxRunning.accumulateAndGet(active, ::maxOf)
                val activeCombined = combinedRunning.incrementAndGet()
                maxCombinedRunning.accumulateAndGet(activeCombined, ::maxOf)
                try {
                    val thermalStatus = power.currentThermalStatus
                    maximumThermalStatus.accumulateAndGet(thermalStatus, ::maxOf)
                    if (thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) {
                        thermalFailClosed.incrementAndGet()
                        throw IllegalStateException("thermal fail-closed status=$thermalStatus")
                    }
                    val fullStart = SystemClock.elapsedRealtimeNanos()
                    val inputTensor = if (directRgbBridge) {
                        preprocessor.preprocessFp16CanonicalStrictDirect(converter.convertDirect(frame))
                    } else {
                        preprocessor.preprocessFp16CanonicalStrict(converter.convert(frame))
                    }
                    val output = runtime.execute(inputTensor)
                    val qnnElapsed = elapsedMs(fullStart)
                    executeLatencies += qnnElapsed
                    markInferenceProcessed(frame.stage)
                    var geometryStatus = "NOT_REQUESTED"
                    if (pipelineGeometry) {
                        val work = checkNotNull(geometryPool.acquire()) {
                            "aligned-depth pool exhausted"
                        }
                        try {
                            val decodeStart = SystemClock.elapsedRealtimeNanos()
                            if (nativeDirectDepthBridge) {
                                preprocessor.decodeResizeFp16AlignCornersStrict(output, checkNotNull(work.directDepth))
                                directDepthBridgeLatencies += elapsedMs(decodeStart)
                            } else {
                                if (nativeFp16Decode) {
                                    preprocessor.decodeFp16ToFloatStrict(output, rawDepth)
                                } else {
                                    val shorts = output.asShortBuffer()
                                    for (index in rawDepth.indices) rawDepth[index] = halfBitsToFloat(shorts.get(index))
                                }
                                fp16DecodeLatencies += elapsedMs(decodeStart)
                                val resizeStart = SystemClock.elapsedRealtimeNanos()
                                resizeDepthAlignCorners(rawDepth, checkNotNull(work.arrayDepth))
                                alignResizeLatencies += elapsedMs(resizeStart)
                            }
                            decodeAlignLatencies += elapsedMs(decodeStart)
                            work.stage = frame.stage
                            work.sensorTimestampNanos = frame.sensorTimestampNanos
                            work.fullStartedAtNanos = fullStart
                            work.checksum = output.asShortBuffer().get(0).toInt() and 0xffff
                            InferenceResult.Geometry(work)
                        } catch (failure: Throwable) {
                            work.close()
                            throw failure
                        }
                    } else if (includeGeometry) {
                        val decodeStart = SystemClock.elapsedRealtimeNanos()
                        if (nativeFp16Decode) {
                            preprocessor.decodeFp16ToFloatStrict(output, rawDepth)
                        } else {
                            val shorts = output.asShortBuffer()
                            for (index in rawDepth.indices) rawDepth[index] = halfBitsToFloat(shorts.get(index))
                        }
                        fp16DecodeLatencies += elapsedMs(decodeStart)
                        val resizeStart = SystemClock.elapsedRealtimeNanos()
                        resizeDepthAlignCorners(rawDepth, alignedDepth)
                        alignResizeLatencies += elapsedMs(resizeStart)
                        decodeAlignLatencies += elapsedMs(decodeStart)
                        val geometryStart = SystemClock.elapsedRealtimeNanos()
                        val geometry = evaluateGeometry(alignedDepth)
                        geometryStatus = if (geometry is KnownHeightGroundPipeline.Geometry) {
                            geometryValid.incrementAndGet(); "VALID"
                        } else {
                            geometryUnknown.incrementAndGet(); "UNKNOWN"
                        }
                        geometryLatencies += elapsedMs(geometryStart)
                        val fullElapsed = elapsedMs(fullStart)
                        fullPipelineLatencies += fullElapsed
                        markProcessed(frame.stage)
                        InferenceResult.Completed(DepthReceipt(
                            frame.stage, frame.sensorTimestampNanos, fullElapsed,
                            geometryStatus, output.asShortBuffer().get(0).toInt() and 0xffff,
                        ))
                    } else {
                        val fullElapsed = elapsedMs(fullStart)
                        fullPipelineLatencies += fullElapsed
                        markProcessed(frame.stage)
                        InferenceResult.Completed(DepthReceipt(
                            frame.stage, frame.sensorTimestampNanos, fullElapsed,
                            geometryStatus, output.asShortBuffer().get(0).toInt() and 0xffff,
                        ))
                    }
                } finally {
                    running.decrementAndGet()
                    combinedRunning.decrementAndGet()
                }
            },
            onFreshResult = { result ->
                when (val value = result.value) {
                    is InferenceResult.Completed -> deliverFinal(
                        LatestOnlySidecar.Result(value.receipt, result.capturedAtNanos, result.completedAtNanos),
                    )
                    is InferenceResult.Geometry -> {
                        if (!geometrySidecar.submit(value.work, result.capturedAtNanos)) {
                            failures += "geometry sidecar closed before inference handoff"
                        }
                    }
                }
            },
            onDiscardedResult = { result ->
                if (result is InferenceResult.Geometry) result.work.close()
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
            if (!power.isInteractive) nonInteractiveObservations.incrementAndGet()
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
                if (stage == "paced") {
                    val pacedClaimed = if (phaseLockedCadence) {
                        phaseLockedGate.claim(receivedAt)
                    } else {
                        claimPacedSlot(lastPacedAt, receivedAt, periodNanos)
                    }
                    if (!pacedClaimed) {
                        throttled.incrementAndGet()
                        return@setAnalyzer
                    }
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
            geometrySidecar.close()
            geometryExecutor.shutdown()
            geometryExecutor.awaitTermination(15, TimeUnit.SECONDS)
            converter.close()
            preprocessor.close()
            runtime.close()
        }

        replaced.set(pool.replacedCount.get())
        val stale = (processed.get() - fresh.get()).coerceAtLeast(0)
        val expiryProbe = resultStore.readAt(
            SystemClock.elapsedRealtimeNanos() + 2 * TimeUnit.MILLISECONDS.toNanos(ttlMs),
        )
        val ttlExpiresToUnknown = expiryProbe is ExpiringLatestResult.State.Unknown &&
            expiryProbe.reason == ExpiringLatestResult.UnknownReason.EXPIRED
        val report = JSONObject()
            .put("schema", "blindassist_camerax_latest_only_r0")
            .put("contract", JSONObject()
                .put("camera_format", "YUV_420_888")
                .put("camera_resolution", "640x480")
                .put("rotation", "clockwise imageInfo.rotationDegrees")
                .put("crop", "center 4:3 after rotation")
                .put("camera_resize", "OpenCV INTER_LINEAR to 640x480 RGB")
                .put("tensor", "frozen OpenCV cubic normalize NCHW FP16 1x3x518x686")
                .put("geometry", "frozen known-height ground pipeline: 5000 candidates, 240 RANSAC iterations")
                .put("geometry_implementation", if (nativeGeometry) "native_cpp_parity_gated" else "canonical_kotlin")
                .put("geometry_ransac_seed", 1729)
                .put("fp16_decode", if (nativeFp16Decode) "native_bit_exact_all_patterns" else "android_half_kotlin")
                .put("depth_bridge", if (nativeDirectDepthBridge) "native_direct_decode_resize_owned_aligned" else "java_float_arrays")
                .put("rgb_bridge", if (directRgbBridge) "native_direct_bit_exact" else "kotlin_byte_array")
                .put(
                    "preprocess_route",
                    "canonical_native_official_fp32_then_integer_rnte_fp16_v1",
                )
                .put("backpressure", "CameraX KEEP_ONLY_LATEST + one running/one replaceable pending")
                .put(
                    "paced_cadence",
                    if (phaseLockedCadence) "phase_locked_deadline_skip_missed" else "last_accepted_frame_relative",
                )
                .put("depth_period_ms", depthPeriodMs)
                .put("result_ttl_ms", ttlMs))
            .put("include_geometry", includeGeometry)
            .put("pipeline_geometry", pipelineGeometry)
            .put("phase_locked_cadence", phaseLockedCadence)
            .put("native_fp16_decode", nativeFp16Decode)
            .put("native_geometry", nativeGeometry)
            .put("native_direct_depth_bridge", nativeDirectDepthBridge)
            .put("direct_rgb_bridge", directRgbBridge)
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
            .put("inference_processed", inferenceProcessed.get())
            .put("stress_inference_processed", stressInferenceProcessed.get())
            .put("paced_inference_processed", pacedInferenceProcessed.get())
            .put("stress_processed", stressProcessed.get())
            .put("paced_processed", pacedProcessed.get())
            .put("stress_processed_per_second", stressProcessed.get().toDouble() / stressSeconds)
            .put(
                "paced_processed_per_second",
                pacedProcessed.get().toDouble() / (durationSeconds - stressSeconds),
            )
            .put("processed_per_second", processed.get().toDouble() / durationSeconds)
            .put("pending_replaced", replaced.get())
            .put("geometry_pending_replaced", geometryPool.replacedCount.get())
            .put("fresh_results", fresh.get())
            .put("stale_results", stale)
            .put("ttl_expires_to_unknown", ttlExpiresToUnknown)
            .put("max_concurrent_depth_tasks", maxRunning.get())
            .put("max_concurrent_geometry_tasks", maxGeometryRunning.get())
            .put("max_concurrent_pipeline_stages", maxCombinedRunning.get())
            .put("pool_available_after_close", pool.available())
            .put("geometry_pool_available_after_close", geometryPool.available())
            .put("thermal_status_before", thermalBefore)
            .put("thermal_status_after", power.currentThermalStatus)
            .put("thermal_fail_closed", thermalFailClosed.get())
            .put("maximum_thermal_status", maximumThermalStatus.get())
            .put("noninteractive_camera_observations", nonInteractiveObservations.get())
            .put("geometry_valid", geometryValid.get())
            .put("geometry_unknown", geometryUnknown.get())
            .put("yuv_copy_ms", latencyJson(copyLatencies))
            .put("yuv_to_fp16_plus_qnn_ms", latencyJson(executeLatencies))
            .put("fp16_decode_align_ms", latencyJson(decodeAlignLatencies))
            .put("fp16_decode_ms", latencyJson(fp16DecodeLatencies))
            .put("align_corners_resize_ms", latencyJson(alignResizeLatencies))
            .put("native_direct_depth_bridge_ms", latencyJson(directDepthBridgeLatencies))
            .put("ground_geometry_ms", latencyJson(geometryLatencies))
            .put("full_depth_geometry_ms", latencyJson(fullPipelineLatencies))
            .put("fresh_result_age_ms", latencyJson(freshAges))
            .put("failures", JSONArray(failures.toList()))
            .put("memory_before", memoryBefore)
            .put("memory_after", memoryJson())
        val runtimeStatsAfter = RuntimeStats.capture()
        report.put("runtime_deltas", JSONObject()
            .put("allocated_bytes", deltaOrNull(runtimeStatsBefore.allocated, runtimeStatsAfter.allocated))
            .put("gc_count", deltaOrNull(runtimeStatsBefore.gcCount, runtimeStatsAfter.gcCount))
            .put("gc_time_ms", deltaOrNull(runtimeStatsBefore.gcTimeMs, runtimeStatsAfter.gcTimeMs)))

        val gateFailures = mutableListOf<String>()
        if (framesSeen.get() < durationSeconds * 5) gateFailures += "camera frame rate below 5 fps"
        if (imageClosed.get() != framesSeen.get()) gateFailures += "ImageProxy leak"
        if (dimensions != setOf("640x480")) gateFailures += "unexpected camera dimensions: $dimensions"
        if (stressSubmitted.get() < 10) gateFailures += "stress arm did not submit enough frames"
        if (pacedSubmitted.get() < (durationSeconds - stressSeconds)) gateFailures += "paced arm below 1 Hz"
        if (stressInferenceProcessed.get() + pacedInferenceProcessed.get() != inferenceProcessed.get()) {
            gateFailures += "inference stage accounting mismatch"
        }
        if (stressProcessed.get() + pacedProcessed.get() != processed.get()) {
            gateFailures += "final stage accounting mismatch"
        }
        if (maxRunning.get() != 1) gateFailures += "depth concurrency was ${maxRunning.get()}"
        if (pipelineGeometry && maxGeometryRunning.get() != 1) {
            gateFailures += "geometry concurrency was ${maxGeometryRunning.get()}"
        }
        if (pipelineGeometry && maxCombinedRunning.get() != 2) {
            gateFailures += "QNN/geometry overlap was not observed: max stages ${maxCombinedRunning.get()}"
        }
        if (pool.available() != 3) gateFailures += "owned YUV slot leak"
        if (geometryPool.available() != 3) gateFailures += "owned aligned-depth slot leak"
        if (fresh.get() == 0 || freshAges.any { it > ttlMs }) gateFailures += "TTL freshness contract failed"
        if (!ttlExpiresToUnknown) gateFailures += "expired depth did not become UNKNOWN"
        if (thermalFailClosed.get() != 0) gateFailures += "device reached severe thermal status"
        if (nonInteractiveObservations.get() != 0) gateFailures += "screen was not continuously interactive"
        if (includeGeometry && geometryValid.get() + geometryUnknown.get() != processed.get()) {
            gateFailures += "geometry did not run for every processed frame"
        }
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

    private fun memoryJson(): JSONObject {
        val runtime = Runtime.getRuntime()
        return JSONObject().put("pss_kib", Debug.getPss())
            .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun deltaOrNull(before: Long, after: Long): Any =
        if (before >= 0 && after >= before) after - before else JSONObject.NULL

    private fun resizeDepthAlignCorners(input: FloatArray, output: FloatArray) {
        for (row in 0 until HEIGHT) {
            val sy = row.toDouble() * (Dav2PreprocessContract.OUTPUT_HEIGHT - 1) / (HEIGHT - 1)
            val y0 = sy.toInt()
            val y1 = minOf(y0 + 1, Dav2PreprocessContract.OUTPUT_HEIGHT - 1)
            val fy = sy - y0
            for (column in 0 until WIDTH) {
                val sx = column.toDouble() * (Dav2PreprocessContract.OUTPUT_WIDTH - 1) / (WIDTH - 1)
                val x0 = sx.toInt()
                val x1 = minOf(x0 + 1, Dav2PreprocessContract.OUTPUT_WIDTH - 1)
                val fx = sx - x0
                val top = input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) +
                    input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                val bottom = input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) +
                    input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                output[row * WIDTH + column] = (top * (1 - fy) + bottom * fy).toFloat()
            }
        }
    }

    private sealed interface InferenceResult {
        data class Completed(val receipt: DepthReceipt) : InferenceResult
        data class Geometry(val work: OwnedAlignedDepth) : InferenceResult
    }

    private data class DepthReceipt(val stage: String, val sensorTimestampNanos: Long,
        val executeMs: Double, val geometryStatus: String, val checksum: Int)

    private data class RuntimeStats(val allocated: Long, val gcCount: Long, val gcTimeMs: Long) {
        companion object {
            fun capture() = RuntimeStats(stat("art.gc.bytes-allocated"), stat("art.gc.gc-count"), stat("art.gc.gc-time"))
            private fun stat(name: String) = Debug.getRuntimeStat(name)?.toLongOrNull() ?: -1L
        }
    }

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

    private class AlignedDepthPool(capacity: Int, elements: Int, direct: Boolean) {
        private val available = ArrayDeque<OwnedAlignedDepth>()
        val replacedCount = AtomicInteger()
        init { repeat(capacity) { available.add(OwnedAlignedDepth(elements, direct, ::release)) } }
        @Synchronized fun acquire(): OwnedAlignedDepth? =
            if (available.isEmpty()) null else available.removeFirst().lease()
        @Synchronized private fun release(work: OwnedAlignedDepth) {
            if (!work.started) replacedCount.incrementAndGet()
            available.addLast(work)
        }
        @Synchronized fun available(): Int = available.size
    }

    private class OwnedAlignedDepth(
        elements: Int,
        direct: Boolean,
        private val onRelease: (OwnedAlignedDepth) -> Unit,
    ) : AutoCloseable {
        val arrayDepth = if (direct) null else FloatArray(elements)
        val directDepth = if (direct) {
            ByteBuffer.allocateDirect(elements * 4).order(java.nio.ByteOrder.nativeOrder())
        } else null
        var stage = ""
        var sensorTimestampNanos = 0L
        var fullStartedAtNanos = 0L
        var checksum = 0
        var started = false
        private var leased = false

        fun lease(): OwnedAlignedDepth {
            check(!leased)
            leased = true
            started = false
            stage = ""
            return this
        }

        override fun close() {
            if (!leased) return
            leased = false
            onRelease(this)
        }
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
