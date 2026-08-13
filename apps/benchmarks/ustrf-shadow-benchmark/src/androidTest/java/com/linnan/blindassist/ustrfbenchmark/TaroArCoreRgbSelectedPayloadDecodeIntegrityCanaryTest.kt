package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Intent
import android.graphics.ImageFormat
import android.media.Image
import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.DeadlineExceededException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.ResourceExhaustedException
import com.linnan.blindassist.ustrf.TaroPoseDiverseFrameSelector
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest
import kotlin.math.roundToInt

/**
 * Device-only delayed-decode integrity canary for exact TARO reference/selected payload pairs.
 *
 * ARCore Image is closed immediately after YUV copy. Only then may the frozen decoder read the
 * owned bytes. Decoder receipts contain identity and hashes only; RGBA bytes never enter history,
 * a model, risk fusion, guidance, or the default app.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreRgbSelectedPayloadDecodeIntegrityCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun exactOwnedReferenceAndSelectedPayloads_decodeDeterministicallyAfterImageClose() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("taroRgbDecodeFrameAttempts")
            ?.toIntOrNull()
            ?.coerceIn(MINIMUM_FRAME_ATTEMPTS, MAXIMUM_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as UstrfArCoreBenchmarkActivity
        assertTrue("ARCore benchmark GL host did not initialize", activity.awaitGlReady())
        assertTrue("ARCore benchmark texture was not allocated", activity.cameraTextureName() != 0)
        val availability = awaitArCoreAvailability(activity)
        assertEquals(
            "ARCore must be supported and installed for this isolated canary",
            ArCoreApk.Availability.SUPPORTED_INSTALLED,
            availability
        )

        val selector = TaroPoseDiverseFrameSelector(enabled = true)
        val history = TaroOwnedRgbPayloadHistory(
            maximumRetainedAgeNs = MAXIMUM_RETAINED_AGE_NS,
            maximumRetainedBytes = MAXIMUM_RETAINED_BYTES
        )
        val decoder = TaroOwnedRgbPayloadDecoder()
        var cameraImageAvailableCount = 0
        var cameraImageNotYetAvailableCount = 0
        var sourceBoundPayloadCopyCount = 0
        var availableSelectionCount = 0
        var exactSelectedPayloadLookupCount = 0
        var decodedPairCount = 0
        var deterministicSelectedReplayCount = 0
        var sourceIdentityMismatchCount = 0
        var deterministicHashMismatchCount = 0
        var decodedDimensionMismatchCount = 0
        var maximumTransientRgbaBytes = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        val individualDecodeLatencyMs = mutableListOf<Double>()
        val pairDecodeLatencyMs = mutableListOf<Double>()
        val distinctReferenceRgbaHashes = linkedSetOf<String>()
        val distinctSelectedRgbaHashes = linkedSetOf<String>()
        val poseAdmissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        val decodeFailureCounts = linkedMapOf<String, Int>()
        val resourceErrorCounts = linkedMapOf<String, Int>()
        var session: Session? = null
        try {
            activity.runOnGlThreadAndWait(timeoutSeconds = DEVICE_TEST_TIMEOUT_SECONDS) {
                session = Session(activity).also { created ->
                    created.configure(Config(created).apply {
                        updateMode = Config.UpdateMode.BLOCKING
                    })
                    created.setCameraTextureName(activity.cameraTextureName())
                    created.resume()
                }
                val poseAdapter = TaroArCoreAnchorPoseAdmissionAdapter(
                    session = requireNotNull(session),
                    sessionToken = SESSION_TOKEN
                )
                try {
                    repeat(frameAttempts) { attempt ->
                        val frame = requireNotNull(session).update()
                        val sourceFrame = UstrfFrameStamp(
                            frameId = attempt.toLong(),
                            capturedAtNs = frame.timestamp,
                            coordinateFrame = ARCORE_CAMERA_FRAME
                        )
                        val poseAdmission = poseAdapter.observe(frame, sourceFrame)
                        if (poseAdmission !is TaroArCoreAnchorPoseAdmission.Available) {
                            poseAdmission as TaroArCoreAnchorPoseAdmission.Unavailable
                            increment(poseAdmissionFailureCounts, poseAdmission.failure.name)
                            history.reset()
                            Thread.sleep(FRAME_SETTLE_MS)
                            return@repeat
                        }

                        val image = try {
                            frame.acquireCameraImage()
                        } catch (_: NotYetAvailableException) {
                            cameraImageNotYetAvailableCount++
                            null
                        } catch (_: DeadlineExceededException) {
                            increment(resourceErrorCounts, "DEADLINE_EXCEEDED")
                            null
                        } catch (_: ResourceExhaustedException) {
                            increment(resourceErrorCounts, "RESOURCE_EXHAUSTED")
                            null
                        }
                        val referencePayload = image?.use { cameraImage ->
                            cameraImageAvailableCount++
                            copyOwnedPayload(cameraImage, sourceFrame, poseAdmission)
                        }
                        // image.use has completed: every decode below is detached from ARCore Image.
                        if (referencePayload == null) {
                            Thread.sleep(FRAME_SETTLE_MS)
                            return@repeat
                        }
                        sourceBoundPayloadCopyCount++
                        history.advanceTo(sourceFrame.capturedAtNs)
                        val selection = selector.select(
                            referenceFrame = sourceFrame,
                            referencePose = poseAdmission.cameraPose,
                            bufferedFrames = history.bufferedPoseFrames()
                        )
                        when (selection) {
                            is TaroPoseDiverseSelection.Available -> {
                                availableSelectionCount++
                                val selectedPayload = history.lookupExact(selection.selectedFrame)
                                if (selectedPayload != null) {
                                    exactSelectedPayloadLookupCount++
                                    val pairDecodeStartedNs = System.nanoTime()
                                    val selectedFirst = decodeOrNull(
                                        decoder,
                                        selectedPayload,
                                        individualDecodeLatencyMs,
                                        decodeFailureCounts
                                    )
                                    val selectedReplay = decodeOrNull(
                                        decoder,
                                        selectedPayload,
                                        individualDecodeLatencyMs,
                                        decodeFailureCounts
                                    )
                                    val referenceDecoded = decodeOrNull(
                                        decoder,
                                        referencePayload,
                                        individualDecodeLatencyMs,
                                        decodeFailureCounts
                                    )
                                    if (selectedFirst != null && selectedReplay != null && referenceDecoded != null) {
                                        pairDecodeLatencyMs +=
                                            (System.nanoTime() - pairDecodeStartedNs).toDouble() / 1_000_000.0
                                        decodedPairCount++
                                        maximumTransientRgbaBytes = maxOf(
                                            maximumTransientRgbaBytes,
                                            selectedFirst.rgbaByteCount,
                                            selectedReplay.rgbaByteCount,
                                            referenceDecoded.rgbaByteCount
                                        )
                                        if (
                                            selectedFirst.sourceFrame != selection.selectedFrame ||
                                            selectedReplay.sourceFrame != selection.selectedFrame ||
                                            referenceDecoded.sourceFrame != sourceFrame
                                        ) {
                                            sourceIdentityMismatchCount++
                                        }
                                        if (selectedFirst.rgbaSha256 == selectedReplay.rgbaSha256) {
                                            deterministicSelectedReplayCount++
                                        } else {
                                            deterministicHashMismatchCount++
                                        }
                                        if (
                                            selectedFirst.widthPx != selectedPayload.imageWidthPx ||
                                            selectedFirst.heightPx != selectedPayload.imageHeightPx ||
                                            referenceDecoded.widthPx != referencePayload.imageWidthPx ||
                                            referenceDecoded.heightPx != referencePayload.imageHeightPx
                                        ) {
                                            decodedDimensionMismatchCount++
                                        }
                                        distinctSelectedRgbaHashes += selectedFirst.rgbaSha256
                                        distinctReferenceRgbaHashes += referenceDecoded.rgbaSha256
                                    }
                                }
                                maximumSelectedTranslationM = maxOf(
                                    maximumSelectedTranslationM,
                                    selection.translationM
                                )
                                maximumSelectedYawDeltaRad = maxOf(
                                    maximumSelectedYawDeltaRad,
                                    selection.yawDeltaRad
                                )
                                minimumSelectedGapNs = minOf(minimumSelectedGapNs, selection.gapNs)
                                maximumSelectedGapNs = maxOf(maximumSelectedGapNs, selection.gapNs)
                            }
                            is TaroPoseDiverseSelection.Unavailable ->
                                increment(selectionFailureCounts, selection.failure.name)
                        }
                        history.append(referencePayload)
                        Thread.sleep(FRAME_SETTLE_MS)
                    }
                } finally {
                    poseAdapter.close()
                    session?.pause()
                    session?.close()
                }
            }
        } finally {
            history.close()
            activity.runOnUiThread { activity.finish() }
        }

        val decodeIntegrityGatePass = decodedPairCount >= MINIMUM_DECODED_PAIRS &&
            decodedPairCount == exactSelectedPayloadLookupCount &&
            deterministicSelectedReplayCount == decodedPairCount &&
            sourceIdentityMismatchCount == 0 &&
            deterministicHashMismatchCount == 0 &&
            decodedDimensionMismatchCount == 0 &&
            decodeFailureCounts.values.sum() == 0 &&
            distinctReferenceRgbaHashes.size >= MINIMUM_DISTINCT_REFERENCE_HASHES &&
            distinctSelectedRgbaHashes.size >= MINIMUM_DISTINCT_SELECTED_HASHES
        val resourceGatePass = resourceErrorCounts.values.sum() == 0
        val selectionWindowGatePass = availableSelectionCount > 0 &&
            minimumSelectedGapNs >= MINIMUM_SELECTION_GAP_NS &&
            maximumSelectedGapNs <= MAXIMUM_SELECTION_GAP_NS
        val observedMotionGatePass = maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
            maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_rgb_selected_payload_decode_integrity_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("frame_attempts", frameAttempts)
            .put("camera_image_available_count", cameraImageAvailableCount)
            .put("camera_image_not_yet_available_count", cameraImageNotYetAvailableCount)
            .put("source_bound_payload_copy_count", sourceBoundPayloadCopyCount)
            .put("available_selection_count", availableSelectionCount)
            .put("exact_selected_payload_lookup_count", exactSelectedPayloadLookupCount)
            .put("decoded_pair_count", decodedPairCount)
            .put("deterministic_selected_replay_count", deterministicSelectedReplayCount)
            .put("source_identity_mismatch_count", sourceIdentityMismatchCount)
            .put("deterministic_hash_mismatch_count", deterministicHashMismatchCount)
            .put("decoded_dimension_mismatch_count", decodedDimensionMismatchCount)
            .put("distinct_reference_rgba_hash_count", distinctReferenceRgbaHashes.size)
            .put("distinct_selected_rgba_hash_count", distinctSelectedRgbaHashes.size)
            .put("maximum_transient_rgba_bytes", maximumTransientRgbaBytes)
            .put("individual_decode_latency_ms", JSONObject()
                .put("sample_count", individualDecodeLatencyMs.size)
                .put("p50", percentileOrNull(individualDecodeLatencyMs, .50))
                .put("p95", percentileOrNull(individualDecodeLatencyMs, .95)))
            .put("three_decode_pair_integrity_latency_ms", JSONObject()
                .put("sample_count", pairDecodeLatencyMs.size)
                .put("p50", percentileOrNull(pairDecodeLatencyMs, .50))
                .put("p95", percentileOrNull(pairDecodeLatencyMs, .95)))
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { availableSelectionCount > 0 } ?: JSONObject.NULL)
            .put("pose_admission_failure_counts", JSONObject(poseAdmissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("decode_failure_counts", JSONObject(decodeFailureCounts as Map<*, *>))
            .put("resource_error_counts", JSONObject(resourceErrorCounts as Map<*, *>))
            .put("decode_integrity_gate_pass", decodeIntegrityGatePass)
            .put("resource_gate_pass", resourceGatePass)
            .put("selection_window_gate_pass", selectionWindowGatePass)
            .put("observed_motion_gate_pass", observedMotionGatePass)
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("decode_after_image_close", true)
                .put("rgba_retained_in_history", false)
                .put("model_inference_run", false)
                .put("task_evidence_gain_proven", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_RGB_SELECTED_PAYLOAD_DECODE_INTEGRITY_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("Owned reference/selected payload decode integrity failed", decodeIntegrityGatePass)
        assertTrue("ARCore camera-image resource errors occurred", resourceGatePass)
        assertTrue("Decoded RGB selection fell outside the frozen 150ms to 1s window", selectionWindowGatePass)
        assertTrue(
            "Decoded RGB selections did not observe at least 2cm translation or 2 degrees yaw",
            observedMotionGatePass
        )
    }

    private fun decodeOrNull(
        decoder: TaroOwnedRgbPayloadDecoder,
        payload: TaroOwnedRgbPayload,
        latencyMs: MutableList<Double>,
        failures: MutableMap<String, Int>
    ): TaroOwnedRgbDecodeReceipt? {
        val startedNs = System.nanoTime()
        return try {
            decoder.decode(payload).also {
                latencyMs += (System.nanoTime() - startedNs).toDouble() / 1_000_000.0
            }
        } catch (error: IllegalArgumentException) {
            increment(failures, "ILLEGAL_ARGUMENT:${error.message.orEmpty()}")
            null
        }
    }

    private fun copyOwnedPayload(
        image: Image,
        sourceFrame: UstrfFrameStamp,
        admission: TaroArCoreAnchorPoseAdmission.Available
    ): TaroOwnedRgbPayload {
        require(image.format == ImageFormat.YUV_420_888)
        val digest = MessageDigest.getInstance("SHA-256")
        val planes = image.planes.map { plane ->
            val input = plane.buffer.duplicate()
            val bytes = ByteArray(input.remaining())
            input.get(bytes)
            digest.update(bytes)
            TaroOwnedYuvPlane(plane.rowStride, plane.pixelStride, bytes)
        }
        return TaroOwnedRgbPayload(
            sourceFrame = sourceFrame,
            anchorPose = admission.cameraPose,
            imageWidthPx = image.width,
            imageHeightPx = image.height,
            imageFormat = image.format,
            planes = planes,
            contentSha256 = digest.digest().joinToString("") { "%02x".format(it) }
        )
    }

    private fun awaitArCoreAvailability(activity: UstrfArCoreBenchmarkActivity): ArCoreApk.Availability {
        var availability = ArCoreApk.Availability.UNKNOWN_CHECKING
        repeat(AVAILABILITY_ATTEMPTS) {
            availability = ArCoreApk.getInstance().checkAvailability(activity)
            if (!availability.isTransient) return availability
            Thread.sleep(AVAILABILITY_RETRY_MS)
        }
        return availability
    }

    private fun increment(counts: MutableMap<String, Int>, key: String) {
        counts[key] = (counts[key] ?: 0) + 1
    }

    private fun percentileOrNull(values: List<Double>, percentile: Double): Any {
        if (values.isEmpty()) return JSONObject.NULL
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * percentile).roundToInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private companion object {
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val MINIMUM_FRAME_ATTEMPTS = 300
        const val DEFAULT_FRAME_ATTEMPTS = 600
        const val MAXIMUM_FRAME_ATTEMPTS = 1_800
        const val DEVICE_TEST_TIMEOUT_SECONDS = 180L
        const val FRAME_SETTLE_MS = 10L
        const val MINIMUM_DECODED_PAIRS = 20
        const val MINIMUM_DISTINCT_REFERENCE_HASHES = 10
        const val MINIMUM_DISTINCT_SELECTED_HASHES = 5
        const val MINIMUM_SELECTION_GAP_NS = 150_000_000L
        const val MAXIMUM_SELECTION_GAP_NS = 1_000_000_000L
        const val MAXIMUM_RETAINED_AGE_NS = 1_000_000_000L
        const val MAXIMUM_RETAINED_BYTES = 32L * 1024L * 1024L
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "rgb-selected-payload-decode-integrity-r0"
        const val REPORT_KEY = "taro_arcore_rgb_selected_payload_decode_integrity"
        const val TAG = "UstrfShadowBenchmark"
    }
}
