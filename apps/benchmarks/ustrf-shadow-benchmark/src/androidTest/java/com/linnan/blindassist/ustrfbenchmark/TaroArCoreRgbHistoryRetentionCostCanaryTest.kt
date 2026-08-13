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
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest
import kotlin.math.roundToInt

/**
 * Device-only TARO canary for owned RGB history and selection-to-payload identity.
 *
 * Each admitted current ARCore Frame has its YUV planes copied before Image.close(). The owned
 * payload is bounded by both one second of source time and 32 MiB, and the frozen selector can
 * resolve a historical payload only through exact UstrfFrameStamp identity. No pixels are decoded,
 * inferred, fused, exposed to guidance, or wired into the default app.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreRgbHistoryRetentionCostCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun copiedYuvHistory_isBoundedAndSelectorIdentityResolvesExactly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frameAttempts = InstrumentationRegistry.getArguments()
            .getString("taroRgbHistoryFrameAttempts")
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
        var cameraImageAvailableCount = 0
        var cameraImageNotYetAvailableCount = 0
        var sourceBoundPayloadCopyCount = 0
        var availableSelectionCount = 0
        var exactSelectedPayloadLookupCount = 0
        var selectedPayloadLookupMissCount = 0
        var selectionReferenceIdentityMismatchCount = 0
        var selectedReceiptIdentityMismatchCount = 0
        var totalCopiedBytes = 0L
        var peakRetainedBytes = 0L
        var peakRetainedEntryCount = 0
        var maximumRetainedAgeObservedNs = 0L
        var ageEvictionCount = 0
        var byteCapEvictionCount = 0
        var continuityResetCount = 0
        var continuityResetEvictionCount = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        var imageWidthPx: Int? = null
        var imageHeightPx: Int? = null
        var imageFormat: Int? = null
        var finalRetainedEntryCount = 0
        var finalRetainedBytes = 0L
        val copyAppendSelectLatencyMs = mutableListOf<Double>()
        val distinctContentHashes = linkedSetOf<String>()
        val selectionReceiptSamples = JSONArray()
        val poseAdmissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        val resourceErrorCounts = linkedMapOf<String, Int>()
        var session: Session? = null
        try {
            activity.runOnGlThreadAndWait(
                timeoutSeconds = maxOf(20L, frameAttempts * FRAME_SETTLE_MS / 1_000L + 25L)
            ) {
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
                            val reset = history.reset()
                            if (reset.evictedEntryCount > 0) {
                                continuityResetCount++
                                continuityResetEvictionCount += reset.evictedEntryCount
                            }
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
                        image?.use { cameraImage ->
                            cameraImageAvailableCount++
                            val operationStartedNs = System.nanoTime()
                            val payload = copyOwnedPayload(cameraImage, sourceFrame, poseAdmission)
                            imageWidthPx = payload.imageWidthPx
                            imageHeightPx = payload.imageHeightPx
                            imageFormat = payload.imageFormat
                            sourceBoundPayloadCopyCount++
                            totalCopiedBytes += payload.byteCount
                            distinctContentHashes += payload.contentSha256

                            val advance = history.advanceTo(sourceFrame.capturedAtNs)
                            ageEvictionCount += advance.ageEvictionCount
                            val selection = selector.select(
                                referenceFrame = sourceFrame,
                                referencePose = poseAdmission.cameraPose,
                                bufferedFrames = history.bufferedPoseFrames()
                            )
                            when (selection) {
                                is TaroPoseDiverseSelection.Available -> {
                                    availableSelectionCount++
                                    if (selection.referenceFrame != sourceFrame) {
                                        selectionReferenceIdentityMismatchCount++
                                    }
                                    val selectedPayload = history.lookupExact(selection.selectedFrame)
                                    if (selectedPayload == null) {
                                        selectedPayloadLookupMissCount++
                                    } else {
                                        exactSelectedPayloadLookupCount++
                                        if (selectedPayload.receipt.sourceFrame != selection.selectedFrame) {
                                            selectedReceiptIdentityMismatchCount++
                                        }
                                        if (selectionReceiptSamples.length() < MAXIMUM_RECEIPT_SAMPLES) {
                                            selectionReceiptSamples.put(JSONObject()
                                                .put("reference_frame_id", sourceFrame.frameId)
                                                .put("reference_timestamp_ns", sourceFrame.capturedAtNs)
                                                .put("selected_frame_id", selectedPayload.sourceFrame.frameId)
                                                .put("selected_timestamp_ns", selectedPayload.sourceFrame.capturedAtNs)
                                                .put("selected_content_sha256", selectedPayload.contentSha256)
                                                .put("selected_byte_count", selectedPayload.byteCount)
                                                .put("selected_anchor_world_frame", selectedPayload.anchorPose.worldFrame))
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

                            val append = history.append(payload)
                            ageEvictionCount += append.ageEvictionCount
                            byteCapEvictionCount += append.byteCapEvictionCount
                            peakRetainedBytes = maxOf(peakRetainedBytes, append.retainedBytes)
                            peakRetainedEntryCount = maxOf(peakRetainedEntryCount, append.retainedEntryCount)
                            val oldestTimestampNs = history.oldestRetainedTimestampNs
                            if (oldestTimestampNs != null) {
                                maximumRetainedAgeObservedNs = maxOf(
                                    maximumRetainedAgeObservedNs,
                                    sourceFrame.capturedAtNs - oldestTimestampNs
                                )
                            }
                            copyAppendSelectLatencyMs +=
                                (System.nanoTime() - operationStartedNs).toDouble() / 1_000_000.0
                        }
                        Thread.sleep(FRAME_SETTLE_MS)
                    }
                } finally {
                    poseAdapter.close()
                    session?.pause()
                    session?.close()
                }
            }
            finalRetainedEntryCount = history.retainedEntryCount
            finalRetainedBytes = history.retainedBytes
        } finally {
            history.close()
            activity.runOnUiThread { activity.finish() }
        }

        val identityGatePass = sourceBoundPayloadCopyCount >= MINIMUM_SOURCE_BOUND_PAYLOADS &&
            distinctContentHashes.size >= MINIMUM_DISTINCT_CONTENT_HASHES &&
            availableSelectionCount >= MINIMUM_AVAILABLE_SELECTIONS &&
            exactSelectedPayloadLookupCount == availableSelectionCount &&
            selectedPayloadLookupMissCount == 0 &&
            selectionReferenceIdentityMismatchCount == 0 &&
            selectedReceiptIdentityMismatchCount == 0
        val boundedHistoryGatePass = peakRetainedBytes <= MAXIMUM_RETAINED_BYTES &&
            finalRetainedBytes <= MAXIMUM_RETAINED_BYTES &&
            maximumRetainedAgeObservedNs <= MAXIMUM_RETAINED_AGE_NS
        val resourceGatePass = resourceErrorCounts.values.sum() == 0
        val selectionWindowGatePass = availableSelectionCount > 0 &&
            minimumSelectedGapNs >= MINIMUM_SELECTION_GAP_NS &&
            maximumSelectedGapNs <= MAXIMUM_SELECTION_GAP_NS
        val observedMotionGatePass = maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
            maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_rgb_history_retention_cost_v1")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("frame_attempts", frameAttempts)
            .put("camera_image_available_count", cameraImageAvailableCount)
            .put("camera_image_not_yet_available_count", cameraImageNotYetAvailableCount)
            .put("source_bound_payload_copy_count", sourceBoundPayloadCopyCount)
            .put("distinct_content_hash_count", distinctContentHashes.size)
            .put("available_selection_count", availableSelectionCount)
            .put("exact_selected_payload_lookup_count", exactSelectedPayloadLookupCount)
            .put("selected_payload_lookup_miss_count", selectedPayloadLookupMissCount)
            .put("selection_reference_identity_mismatch_count", selectionReferenceIdentityMismatchCount)
            .put("selected_receipt_identity_mismatch_count", selectedReceiptIdentityMismatchCount)
            .put("selection_receipt_samples", selectionReceiptSamples)
            .put("image_width_px", imageWidthPx ?: JSONObject.NULL)
            .put("image_height_px", imageHeightPx ?: JSONObject.NULL)
            .put("image_format", imageFormat ?: JSONObject.NULL)
            .put("total_copied_bytes", totalCopiedBytes)
            .put("maximum_retained_age_ns", MAXIMUM_RETAINED_AGE_NS)
            .put("maximum_retained_bytes", MAXIMUM_RETAINED_BYTES)
            .put("maximum_retained_age_observed_ns", maximumRetainedAgeObservedNs)
            .put("peak_retained_bytes", peakRetainedBytes)
            .put("peak_retained_entry_count", peakRetainedEntryCount)
            .put("final_retained_bytes", finalRetainedBytes)
            .put("final_retained_entry_count", finalRetainedEntryCount)
            .put("age_eviction_count", ageEvictionCount)
            .put("byte_cap_eviction_count", byteCapEvictionCount)
            .put("continuity_reset_count", continuityResetCount)
            .put("continuity_reset_eviction_count", continuityResetEvictionCount)
            .put("copy_append_select_latency_ms", JSONObject()
                .put("sample_count", copyAppendSelectLatencyMs.size)
                .put("p50", percentileOrNull(copyAppendSelectLatencyMs, .50))
                .put("p95", percentileOrNull(copyAppendSelectLatencyMs, .95)))
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { availableSelectionCount > 0 } ?: JSONObject.NULL)
            .put("pose_admission_failure_counts", JSONObject(poseAdmissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("resource_error_counts", JSONObject(resourceErrorCounts as Map<*, *>))
            .put("identity_gate_pass", identityGatePass)
            .put("bounded_history_gate_pass", boundedHistoryGatePass)
            .put("resource_gate_pass", resourceGatePass)
            .put("selection_window_gate_pass", selectionWindowGatePass)
            .put("observed_motion_gate_pass", observedMotionGatePass)
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("image_closed_before_next_session_update", true)
                .put("owned_yuv_history_validated", identityGatePass && boundedHistoryGatePass)
                .put("pixels_decoded_or_inferred", false)
                .put("task_evidence_gain_proven", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_RGB_HISTORY_RETENTION_COST_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("Owned RGB payload identities did not resolve exactly", identityGatePass)
        assertTrue("Owned RGB history exceeded its age or byte bound", boundedHistoryGatePass)
        assertTrue("ARCore camera-image resource errors occurred", resourceGatePass)
        assertTrue("RGB history selection fell outside the frozen 150ms to 1s window", selectionWindowGatePass)
        assertTrue(
            "RGB history selections did not observe at least 2cm translation or 2 degrees yaw",
            observedMotionGatePass
        )
    }

    private fun copyOwnedPayload(
        image: Image,
        sourceFrame: UstrfFrameStamp,
        admission: TaroArCoreAnchorPoseAdmission.Available
    ): TaroOwnedRgbPayload {
        require(image.format == ImageFormat.YUV_420_888) {
            "expected YUV_420_888 but received ${image.format}"
        }
        val digest = MessageDigest.getInstance("SHA-256")
        val planes = image.planes.map { plane ->
            val input = plane.buffer.duplicate()
            val bytes = ByteArray(input.remaining())
            input.get(bytes)
            digest.update(bytes)
            TaroOwnedYuvPlane(
                rowStrideBytes = plane.rowStride,
                pixelStrideBytes = plane.pixelStride,
                bytes = bytes
            )
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
        const val FRAME_SETTLE_MS = 33L
        const val MINIMUM_SOURCE_BOUND_PAYLOADS = 30
        const val MINIMUM_DISTINCT_CONTENT_HASHES = 10
        const val MINIMUM_AVAILABLE_SELECTIONS = 20
        const val MINIMUM_SELECTION_GAP_NS = 150_000_000L
        const val MAXIMUM_SELECTION_GAP_NS = 1_000_000_000L
        const val MAXIMUM_RETAINED_AGE_NS = 1_000_000_000L
        const val MAXIMUM_RETAINED_BYTES = 32L * 1024L * 1024L
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val MAXIMUM_RECEIPT_SAMPLES = 6
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "rgb-history-retention-cost-r0"
        const val REPORT_KEY = "taro_arcore_rgb_history_retention_cost"
        const val TAG = "UstrfShadowBenchmark"
    }
}
