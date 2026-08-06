package com.linnan.blindassist.timing

import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.os.Trace
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.camera.AtomS3rMjpegFrameSource
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.RuntimeObjectDetectorFactory
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AtomS3rAndroidE2eTimingTest {
    @Test
    fun recordsOptInRealDeviceEndToEndBaseline() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val durationSeconds = arguments
            .getString(ARG_DURATION_SECONDS)?.toLongOrNull() ?: 0L
        assumeTrue("Pass -e $ARG_DURATION_SECONDS to run the hardware benchmark", durationSeconds > 0L)
        val endpoint = arguments.getString(ARG_ENDPOINT) ?: DEFAULT_ENDPOINT
        val decodeSampleSize = arguments.getString(ARG_DECODE_SAMPLE_SIZE)
            ?.toIntOrNull() ?: 1
        val maxFrameAgeMs = arguments.getString(ARG_MAX_FRAME_AGE_MS)
            ?.toLongOrNull() ?: 0L
        val context = instrumentation.targetContext
        val detector = RuntimeObjectDetectorFactory.create(context)
        assertTrue("Production detector must be ready: ${detector.statusMessage}", detector.isReady)
        val feedback = FeedbackController(context)
        val coordinator = AssistSessionCoordinator(feedbackGateway = feedback)
        val source = AtomS3rMjpegFrameSource(
            endpoint = endpoint,
            decodeSampleSize = decodeSampleSize,
            maxFrameAgeMs = maxFrameAgeMs
        )
        val outputDir = requireNotNull(context.getExternalFilesDir("atoms3r-android-e2e"))
        val rowsFile = File(outputDir, "frames-${System.currentTimeMillis()}.jsonl")
        val summaryFile = File(outputDir, rowsFile.nameWithoutExtension + "-summary.json")
        val completed = CountDownLatch(1)
        val frames = AtomicInteger()
        val errors = AtomicInteger()
        val terminalError = AtomicReference<Throwable?>(null)
        val sequenceGaps = AtomicLong()
        var previousSequence: Long? = null
        val startedNs = SystemClock.elapsedRealtimeNanos()
        val deadlineNs = startedNs + TimeUnit.SECONDS.toNanos(durationSeconds)
        val initialPssKb = Debug.getPss().toLong()
        val deadlineThread = Thread({
            try {
                Thread.sleep(TimeUnit.SECONDS.toMillis(durationSeconds))
                completed.countDown()
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }, "atoms3r-e2e-deadline").apply { isDaemon = true }

        coordinator.startSession(SystemClock.elapsedRealtime())
        rowsFile.bufferedWriter().use { writer ->
            try {
                deadlineThread.start()
                source.start(
                    previewView = null,
                    onFrame = { frame ->
                        try {
                            val timing = requireNotNull(frame.externalTiming) {
                                "AtomS3R frame missing external timing"
                            }
                            val transport = requireNotNull(frame.externalTransportDiagnostics) {
                                "AtomS3R frame missing transport diagnostics"
                            }
                            val stamp = requireNotNull(frame.frameStamp)
                            require(stamp.clockDomain ==
                                FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_MAPPED_TO_ANDROID) {
                                "UDP clock mapping is required for end-to-end latency"
                            }
                            previousSequence?.let { previous ->
                                if (stamp.frameId > previous + 1L) {
                                    sequenceGaps.addAndGet(stamp.frameId - previous - 1L)
                                }
                            }
                            previousSequence = stamp.frameId
                            val detectorCallStartNs = SystemClock.elapsedRealtimeNanos()
                            var detected = detector.detect(frame)
                            val detectorCallCompleteNs = SystemClock.elapsedRealtimeNanos()
                            val detectorTiming = requireNotNull(detected.stageTiming) {
                                "Detector stage timing is required for R2 latency decomposition"
                            }
                            if (detected.sourceRanging == null && frame.rangingSample != null) {
                                detected = detected.copy(sourceRanging = frame.rangingSample)
                            }
                            val decisionAtNs = SystemClock.elapsedRealtimeNanos()
                            Trace.beginSection(TRACE_RISK_DECISION)
                            val result = try {
                                coordinator.processFrame(
                                    detectorFrame = detected,
                                    profile = AlertProfile.STANDARD,
                                    scenario = AssistScenario.GENERAL,
                                    nowMs = stamp.capturedAtNs / NANOS_PER_MILLISECOND,
                                    decisionAtNs = decisionAtNs,
                                    dualLoopDecisionClockDomain =
                                        FrameClockDomain.ANDROID_ELAPSED_REALTIME
                                )
                            } finally {
                                Trace.endSection()
                            }
                            val riskCompleteNs = SystemClock.elapsedRealtimeNanos()
                            val captureAndroidNs = stamp.capturedAtNs
                            val tof = frame.rangingSample
                            writer.appendLine(JSONObject().apply {
                                put("schema", "blindassist_atoms3r_android_latency_decomposition_r2_frame_v1")
                                put("frame_sequence", stamp.frameId)
                                put("device_capture_ns", timing.deviceCaptureNs)
                                put("device_jpeg_ready_ns", timing.deviceJpegReadyNs)
                                putNullable("device_send_start_ns", timing.deviceSendStartNs)
                                put("android_capture_mapped_ns", captureAndroidNs)
                                put("android_first_byte_ns", timing.androidFirstByteNs)
                                put("android_jpeg_complete_ns", timing.androidJpegCompleteNs)
                                put("android_decode_start_ns", timing.androidDecodeStartNs)
                                put("android_decode_complete_ns", timing.androidDecodeCompleteNs)
                                put("android_rgba_complete_ns", timing.androidRgbaCompleteNs)
                                put("detector_call_start_ns", detectorCallStartNs)
                                put("preprocess_start_ns", detectorTiming.preprocessStartNs)
                                put("preprocess_complete_ns", detectorTiming.preprocessCompleteNs)
                                put("qnn_enqueue_ns", detectorTiming.qnnEnqueueNs)
                                put("qnn_enqueue_semantics", "host_interpreter_run_entry")
                                put("qnn_complete_ns", detectorTiming.qnnCompleteNs)
                                put("output_read_complete_ns", detectorTiming.outputReadCompleteNs)
                                put("postprocess_complete_ns", detectorTiming.postprocessCompleteNs)
                                put("detector_call_complete_ns", detectorCallCompleteNs)
                                put("risk_decision_at_ns", decisionAtNs)
                                put("risk_complete_ns", riskCompleteNs)
                                putNullable("speech_request_ns", result.feedbackDecision.speechRequestAtNs)
                                putNullable("vibration_request_ns", result.feedbackDecision.vibrationRequestAtNs)
                                put("clock_offset_ns", timing.deviceMinusAndroidNs)
                                put("clock_sync_rtt_ns", timing.clockSyncRttNs)
                                put("clock_sync_error_bound_ns", timing.clockSyncErrorBoundNs)
                                put("capture_to_jpeg_complete_ms", ms(timing.androidJpegCompleteNs - captureAndroidNs))
                                put("capture_to_decode_complete_ms", ms(timing.androidDecodeCompleteNs - captureAndroidNs))
                                put("decode_duration_ms", ms(timing.androidDecodeCompleteNs - timing.androidDecodeStartNs))
                                put("rgba_duration_ms", ms(timing.androidRgbaCompleteNs - timing.androidDecodeCompleteNs))
                                put("jpeg_size_bytes", transport.jpegSizeBytes)
                                putNullable("wifi_rssi_dbm", transport.wifiRssiDbm)
                                putNullable("previous_frame_sequence", transport.previousFrameSequence)
                                putNullable(
                                    "previous_response_write_duration_ms",
                                    transport.previousResponseWriteDurationNs?.let(::ms)
                                )
                                put("android_body_read_calls", transport.androidBodyReadCalls)
                                put("android_max_body_read_gap_ms", ms(transport.androidMaxBodyReadGapNs))
                                put(
                                    "android_first_byte_to_jpeg_complete_ms",
                                    ms(timing.androidJpegCompleteNs - timing.androidFirstByteNs)
                                )
                                put("jpeg_complete_to_decode_start_ms", ms(timing.androidDecodeStartNs - timing.androidJpegCompleteNs))
                                put("decode_complete_to_detector_call_ms", ms(detectorCallStartNs - timing.androidDecodeCompleteNs))
                                put("detector_entry_to_preprocess_ms", ms(detectorTiming.preprocessStartNs - detectorCallStartNs))
                                put("preprocess_ms", ms(detectorTiming.preprocessCompleteNs - detectorTiming.preprocessStartNs))
                                val drawStartNs = detectorTiming.preprocessLetterboxDrawStartNs
                                val drawCompleteNs = detectorTiming.preprocessLetterboxDrawCompleteNs
                                val pixelsCompleteNs = detectorTiming.preprocessBitmapPixelsCompleteNs
                                val inputWriteCompleteNs = detectorTiming.preprocessInputWriteCompleteNs
                                if (drawStartNs != null && drawCompleteNs != null &&
                                    pixelsCompleteNs != null && inputWriteCompleteNs != null
                                ) {
                                    put("preprocess_letterbox_draw_ms", ms(drawCompleteNs - drawStartNs))
                                    put("preprocess_bitmap_get_pixels_ms", ms(pixelsCompleteNs - drawCompleteNs))
                                    put("preprocess_input_write_ms", ms(inputWriteCompleteNs - pixelsCompleteNs))
                                }
                                put("preprocess_to_qnn_enqueue_ms", ms(detectorTiming.qnnEnqueueNs - detectorTiming.preprocessCompleteNs))
                                put("qnn_execute_ms", ms(detectorTiming.qnnCompleteNs - detectorTiming.qnnEnqueueNs))
                                put("output_read_ms", ms(detectorTiming.outputReadCompleteNs - detectorTiming.qnnCompleteNs))
                                put("postprocess_ms", ms(detectorTiming.postprocessCompleteNs - detectorTiming.outputReadCompleteNs))
                                put("postprocess_to_risk_start_ms", ms(decisionAtNs - detectorTiming.postprocessCompleteNs))
                                put("risk_ms", ms(riskCompleteNs - decisionAtNs))
                                put("detector_total_ms", ms(detectorCallCompleteNs - detectorCallStartNs))
                                put("frame_age_at_first_byte_ms", ms(timing.androidFirstByteNs - captureAndroidNs))
                                put("frame_age_at_decode_start_ms", ms(timing.androidDecodeStartNs - captureAndroidNs))
                                put("frame_age_at_preprocess_start_ms", ms(detectorTiming.preprocessStartNs - captureAndroidNs))
                                put("frame_age_at_qnn_enqueue_ms", ms(detectorTiming.qnnEnqueueNs - captureAndroidNs))
                                put("frame_age_at_postprocess_complete_ms", ms(detectorTiming.postprocessCompleteNs - captureAndroidNs))
                                put("frame_age_at_risk_ready_ms", ms(riskCompleteNs - captureAndroidNs))
                                put("capture_to_risk_complete_ms", ms(riskCompleteNs - captureAndroidNs))
                                putNullable("tof_timestamp_ns", tof?.sampledAtNs)
                                putNullable("tof_age_at_jpeg_ready_ns", tof?.ageAtFrameReadyNs)
                                putNullable("tof_range_mm", tof?.rangeMm)
                                put("tof_valid", tof?.valid ?: false)
                                put("detection_count", detected.detections.size)
                                put("feedback_reason", result.feedbackDecision.reason.name)
                                put("speech_triggered", result.feedbackDecision.speechTriggered)
                                put("vibration_triggered", result.feedbackDecision.vibrationTriggered)
                            }.toString())
                            writer.flush()
                            frames.incrementAndGet()
                            if (SystemClock.elapsedRealtimeNanos() >= deadlineNs) completed.countDown()
                        } catch (error: Throwable) {
                            errors.incrementAndGet()
                            terminalError.compareAndSet(null, error)
                            completed.countDown()
                        } finally {
                            frame.close()
                        }
                    },
                    onStarted = {},
                    onError = {
                        errors.incrementAndGet()
                        terminalError.compareAndSet(null, it)
                        completed.countDown()
                    }
                )
                assertTrue(
                    "Timed out before completing AtomS3R benchmark",
                    completed.await(durationSeconds + STARTUP_GRACE_SECONDS, TimeUnit.SECONDS)
                )
            } finally {
                source.shutdown()
                detector.close()
                feedback.shutdown()
            }
        }

        val endedNs = SystemClock.elapsedRealtimeNanos()
        val sourceDiagnostics = source.diagnostics()
        summaryFile.writeText(JSONObject().apply {
            put("schema", "blindassist_atoms3r_android_latency_decomposition_r2_summary_v1")
            put("development_only", true)
            put("physical_speech_onset", "NOT_EVALUABLE")
            put("physical_vibration_onset", "NOT_EVALUABLE")
            put("device_model", Build.MODEL)
            put("device_product", Build.PRODUCT)
            put("android_sdk", Build.VERSION.SDK_INT)
            put("endpoint", endpoint)
            put("decode_sample_size", decodeSampleSize)
            put("max_frame_age_ms", maxFrameAgeMs)
            put("requested_duration_seconds", durationSeconds)
            put("actual_duration_ms", ms(endedNs - startedNs))
            put("frames", frames.get())
            put("sequence_gap_frames", sequenceGaps.get())
            put("errors", errors.get())
            put("source_packets_read", sourceDiagnostics.packetsRead)
            put("source_latest_packet_overwrites", sourceDiagnostics.latestPacketOverwrites)
            put("source_reconnects", sourceDiagnostics.reconnects)
            put("source_stream_errors", sourceDiagnostics.streamErrors)
            put("source_stale_packets_dropped", sourceDiagnostics.stalePacketsDropped)
            put("clock_sync_successes", sourceDiagnostics.clockSyncSuccesses)
            put("clock_sync_failures", sourceDiagnostics.clockSyncFailures)
            put("initial_pss_kb", initialPssKb)
            put("final_pss_kb", Debug.getPss())
            put("frames_jsonl", rowsFile.absolutePath)
        }.toString(2))

        assertTrue("Expected at least one processed frame", frames.get() > 0)
        terminalError.get()?.let { throw AssertionError("Hardware benchmark failed", it) }
        assertTrue("Hardware benchmark reported errors", errors.get() == 0)
    }

    private fun JSONObject.putNullable(name: String, value: Any?) {
        put(name, value ?: JSONObject.NULL)
    }

    private fun ms(ns: Long): Double = ns / 1_000_000.0

    private companion object {
        const val ARG_DURATION_SECONDS = "atoms3rE2eDurationSeconds"
        const val ARG_ENDPOINT = "atoms3rEndpoint"
        const val ARG_DECODE_SAMPLE_SIZE = "atoms3rDecodeSampleSize"
        const val ARG_MAX_FRAME_AGE_MS = "atoms3rMaxFrameAgeMs"
        const val DEFAULT_ENDPOINT = "http://192.168.5.11"
        const val STARTUP_GRACE_SECONDS = 30L
        const val NANOS_PER_MILLISECOND = 1_000_000L
        const val TRACE_RISK_DECISION = "BlindAssist.RiskDecision"
    }
}
