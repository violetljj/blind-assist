package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Context
import android.content.Intent
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.Surface
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
import com.linnan.blindassist.vision.DetectorExecutionBackend
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.InputStream
import java.security.MessageDigest
import kotlin.math.roundToInt

/** One opaque-scene shard of the pre-locked TARO frozen-YOLO positive-evidence shadow. */
@RunWith(AndroidJUnit4::class)
class TaroArCoreYoloPositiveEvidenceSceneCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun frozenYolo_comparesPassiveAndPoseDiverseAtTheSameExtraFrameBudget() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val sceneId = arguments.getString(ARG_SCENE_ID)?.takeIf(SCENE_ID_REGEX::matches)
        assertTrue("taroSceneId must be an opaque 1..32 character identifier", sceneId != null)
        val exactSceneId = requireNotNull(sceneId)
        val targetReferences = arguments.getString(ARG_TARGET_REFERENCES)
            ?.toIntOrNull()
            ?.coerceIn(MINIMUM_EVALUABLE_REFERENCES, MAXIMUM_EVALUABLE_REFERENCES)
            ?: DEFAULT_TARGET_REFERENCES
        val frameAttempts = arguments.getString(ARG_FRAME_ATTEMPTS)
            ?.toIntOrNull()
            ?.coerceIn(MINIMUM_FRAME_ATTEMPTS, MAXIMUM_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val detector = TfliteYoloDetector(
            context = instrumentation.targetContext,
            executionBackend = DetectorExecutionBackend.CPU_XNNPACK
        )
        val detectorReadyAtStart = detector.isReady
        val detectorStatusAtStart = detector.statusMessage
        val modelSha256 = assetSha256(instrumentation.targetContext, TfliteYoloDetector.MODEL_ASSET)
        val labelsSha256 = assetSha256(instrumentation.targetContext, TfliteYoloDetector.LABELS_ASSET)
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .putExtra(
                    UstrfArCoreBenchmarkActivity.EXTRA_STATUS_TEXT,
                    "TARO positive-evidence shadow: $exactSceneId\n" +
                        "Keep this view, gently move the phone, and do not rely on this test for guidance."
                )
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
        val ownedHistory = TaroOwnedRgbPayloadHistory(
            maximumRetainedAgeNs = MAXIMUM_HISTORY_AGE_NS,
            maximumRetainedBytes = MAXIMUM_HISTORY_BYTES
        )
        val payloads = linkedMapOf<UstrfFrameStamp, TaroOwnedRgbPayload>()
        val evidenceCache = linkedMapOf<UstrfFrameStamp, TaroPositiveVisualReceipt>()
        val adapter = TaroOwnedRgbYoloEvidenceAdapter(detector)
        var frameAttemptsUsed = 0
        var cameraImageNotYetAvailableCount = 0
        var sourceBoundPayloadCopyCount = 0
        var evaluableReferenceCount = 0
        var positiveSupportReferenceCount = 0
        var opportunityReferenceCount = 0
        var poseStrictWinReferenceCount = 0
        var tieReferenceCount = 0
        var poseLossReferenceCount = 0
        var currentFocusedTokenSum = 0
        var passiveNewFocusedTokenSum = 0
        var poseNewFocusedTokenSum = 0
        var currentAllTokenSum = 0
        var passiveNewAllTokenSum = 0
        var poseNewAllTokenSum = 0
        var exactPassivePayloadLookupCount = 0
        var exactPosePayloadLookupCount = 0
        var selectedPayloadLookupMissCount = 0
        var sourceIdentityMismatchCount = 0
        var continuityResetCount = 0
        var continuityResetEvictionCount = 0
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        var detectorRotationDegrees: Int? = null
        val decodeLatencyMs = mutableListOf<Double>()
        val detectorPreprocessLatencyMs = mutableListOf<Double>()
        val detectorInferenceLatencyMs = mutableListOf<Double>()
        val detectorPostprocessLatencyMs = mutableListOf<Double>()
        val detectorTotalLatencyMs = mutableListOf<Double>()
        val poseAdmissionFailureCounts = linkedMapOf<String, Int>()
        val selectionFailureCounts = linkedMapOf<String, Int>()
        val modelFailureCounts = linkedMapOf<String, Int>()
        val resourceErrorCounts = linkedMapOf<String, Int>()
        val recordLatency: (TaroPositiveVisualReceipt) -> Unit = { receipt ->
            decodeLatencyMs += receipt.decodeLatencyMs
            detectorPreprocessLatencyMs += receipt.detectorPreprocessLatencyMs.toDouble()
            detectorInferenceLatencyMs += receipt.detectorInferenceLatencyMs.toDouble()
            detectorPostprocessLatencyMs += receipt.detectorPostprocessLatencyMs.toDouble()
            detectorTotalLatencyMs += receipt.detectorTotalLatencyMs.toDouble()
        }
        var session: Session? = null
        try {
            if (detectorReadyAtStart) {
                activity.runOnGlThreadAndWait(timeoutSeconds = DEVICE_TEST_TIMEOUT_SECONDS) {
                    session = Session(activity).also { created ->
                        created.configure(Config(created).apply { updateMode = Config.UpdateMode.BLOCKING })
                        created.setDisplayGeometry(
                            activity.display?.rotation ?: Surface.ROTATION_0,
                            activity.window.decorView.width.coerceAtLeast(1),
                            activity.window.decorView.height.coerceAtLeast(1)
                        )
                        val characteristics = (
                            activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                            ).getCameraCharacteristics(created.cameraConfig.cameraId)
                        detectorRotationDegrees = detectorRotationDegrees(
                            characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION),
                            DETECTOR_ANALYSIS_TARGET_ROTATION,
                            characteristics.get(CameraCharacteristics.LENS_FACING)
                        )
                        created.setCameraTextureName(activity.cameraTextureName())
                        created.resume()
                    }
                    val poseAdapter = TaroArCoreAnchorPoseAdmissionAdapter(
                        session = requireNotNull(session),
                        sessionToken = "$SESSION_TOKEN_PREFIX:$exactSceneId"
                    )
                    try {
                        for (attempt in 0 until frameAttempts) {
                            if (evaluableReferenceCount >= targetReferences) break
                            frameAttemptsUsed++
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
                                val reset = ownedHistory.reset()
                                if (reset.evictedEntryCount > 0) {
                                    continuityResetCount++
                                    continuityResetEvictionCount += reset.evictedEntryCount
                                }
                                payloads.clear()
                                evidenceCache.clear()
                                Thread.sleep(FRAME_SETTLE_MS)
                                continue
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
                                copyOwnedPayload(cameraImage, sourceFrame, poseAdmission)
                            }
                            if (referencePayload == null) {
                                Thread.sleep(FRAME_SETTLE_MS)
                                continue
                            }
                            sourceBoundPayloadCopyCount++
                            ownedHistory.advanceTo(sourceFrame.capturedAtNs)
                            removePayloadsNotRetained(payloads, ownedHistory)
                            val passivePayload = TaroPositiveVisualEvidence.selectPassive(
                                sourceFrame,
                                payloads.values.toList()
                            )
                            val poseSelection = selector.select(
                                sourceFrame,
                                poseAdmission.cameraPose,
                                ownedHistory.bufferedPoseFrames()
                            )
                            val posePayload = when (poseSelection) {
                                is TaroPoseDiverseSelection.Available -> {
                                    maximumSelectedTranslationM = maxOf(
                                        maximumSelectedTranslationM,
                                        poseSelection.translationM
                                    )
                                    maximumSelectedYawDeltaRad = maxOf(
                                        maximumSelectedYawDeltaRad,
                                        poseSelection.yawDeltaRad
                                    )
                                    minimumSelectedGapNs = minOf(minimumSelectedGapNs, poseSelection.gapNs)
                                    maximumSelectedGapNs = maxOf(maximumSelectedGapNs, poseSelection.gapNs)
                                    ownedHistory.lookupExact(poseSelection.selectedFrame).also {
                                        if (it == null) selectedPayloadLookupMissCount++ else exactPosePayloadLookupCount++
                                    }
                                }
                                is TaroPoseDiverseSelection.Unavailable -> {
                                    increment(selectionFailureCounts, poseSelection.failure.name)
                                    null
                                }
                            }
                            if (passivePayload != null) exactPassivePayloadLookupCount++
                            val rotation = detectorRotationDegrees
                            if (passivePayload != null && posePayload != null && rotation != null) {
                                // All three identities are frozen above; only now may model receipts be read.
                                val currentEvidence = evidenceFor(
                                    referencePayload,
                                    rotation,
                                    adapter,
                                    evidenceCache,
                                    modelFailureCounts,
                                    recordLatency
                                )
                                val passiveEvidence = evidenceFor(
                                    passivePayload,
                                    rotation,
                                    adapter,
                                    evidenceCache,
                                    modelFailureCounts,
                                    recordLatency
                                )
                                val poseEvidence = evidenceFor(
                                    posePayload,
                                    rotation,
                                    adapter,
                                    evidenceCache,
                                    modelFailureCounts,
                                    recordLatency
                                )
                                if (currentEvidence != null && passiveEvidence != null && poseEvidence != null) {
                                    if (
                                        currentEvidence.sourceFrame != sourceFrame ||
                                        passiveEvidence.sourceFrame != passivePayload.sourceFrame ||
                                        poseEvidence.sourceFrame != posePayload.sourceFrame
                                    ) {
                                        sourceIdentityMismatchCount++
                                    } else {
                                        val comparison = TaroPositiveVisualEvidence.compare(
                                            currentEvidence,
                                            passiveEvidence,
                                            poseEvidence
                                        )
                                        evaluableReferenceCount++
                                        currentFocusedTokenSum += comparison.currentFocusedTokenCount
                                        passiveNewFocusedTokenSum += comparison.passiveNewFocusedTokenCount
                                        poseNewFocusedTokenSum += comparison.poseDiverseNewFocusedTokenCount
                                        currentAllTokenSum += comparison.currentAllTokenCount
                                        passiveNewAllTokenSum += comparison.passiveNewAllTokenCount
                                        poseNewAllTokenSum += comparison.poseDiverseNewAllTokenCount
                                        if (
                                            currentEvidence.tokens.isNotEmpty() ||
                                            passiveEvidence.tokens.isNotEmpty() ||
                                            poseEvidence.tokens.isNotEmpty()
                                        ) {
                                            positiveSupportReferenceCount++
                                        }
                                        when {
                                            comparison.poseDiverseNewFocusedTokenCount >
                                                comparison.passiveNewFocusedTokenCount -> {
                                                opportunityReferenceCount++
                                                poseStrictWinReferenceCount++
                                            }
                                            comparison.poseDiverseNewFocusedTokenCount <
                                                comparison.passiveNewFocusedTokenCount -> {
                                                opportunityReferenceCount++
                                                poseLossReferenceCount++
                                            }
                                            else -> tieReferenceCount++
                                        }
                                    }
                                }
                            }
                            val append = ownedHistory.append(referencePayload)
                            payloads[sourceFrame] = referencePayload
                            if (append.byteCapEvictionCount > 0 || append.ageEvictionCount > 0) {
                                removePayloadsNotRetained(payloads, ownedHistory)
                            }
                            Thread.sleep(FRAME_SETTLE_MS)
                        }
                    } finally {
                        poseAdapter.close()
                        session?.pause()
                        session?.close()
                    }
                }
            }
        } finally {
            ownedHistory.close()
            detector.close()
            activity.runOnUiThread { activity.finish() }
        }

        fun mean(sum: Int): Any = if (evaluableReferenceCount == 0) JSONObject.NULL
            else sum.toDouble() / evaluableReferenceCount
        val detectorP95 = percentileOrNull(detectorTotalLatencyMs, .95)
        val structuralGatePass = detectorReadyAtStart &&
            modelSha256 == FROZEN_MODEL_SHA256 &&
            labelsSha256 == FROZEN_LABELS_SHA256 &&
            detectorRotationDegrees != null &&
            evaluableReferenceCount in MINIMUM_EVALUABLE_REFERENCES..MAXIMUM_EVALUABLE_REFERENCES &&
            sourceIdentityMismatchCount == 0 &&
            selectedPayloadLookupMissCount == 0 &&
            modelFailureCounts.values.sum() == 0 &&
            resourceErrorCounts.values.sum() == 0 &&
            detectorP95 != null && detectorP95 <= MAXIMUM_DETECTOR_TOTAL_P95_MS &&
            minimumSelectedGapNs >= TaroPositiveVisualEvidence.MINIMUM_GAP_NS &&
            maximumSelectedGapNs <= TaroPositiveVisualEvidence.MAXIMUM_GAP_NS &&
            (maximumSelectedTranslationM >= MINIMUM_OBSERVED_TRANSLATION_M ||
                maximumSelectedYawDeltaRad >= MINIMUM_OBSERVED_YAW_DELTA_RAD)
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_yolo_positive_evidence_scene_v1")
            .put("protocol_id", "TARO_RGB_PAIR_YOLO_POSITIVE_EVIDENCE_SHADOW_R0")
            .put("scene_id", exactSceneId)
            .put("availability", availability.name)
            .put("detector_ready_at_start", detectorReadyAtStart)
            .put("detector_status_at_start", detectorStatusAtStart)
            .put("execution_backend", DetectorExecutionBackend.CPU_XNNPACK.wireName)
            .put("model_sha256", modelSha256)
            .put("labels_sha256", labelsSha256)
            .put("frame_attempts_limit", frameAttempts)
            .put("frame_attempts_used", frameAttemptsUsed)
            .put("target_evaluable_references", targetReferences)
            .put("camera_image_not_yet_available_count", cameraImageNotYetAvailableCount)
            .put("source_bound_payload_copy_count", sourceBoundPayloadCopyCount)
            .put("unique_inference_count", evidenceCache.size)
            .put("evaluable_reference_count", evaluableReferenceCount)
            .put("positive_support_reference_count", positiveSupportReferenceCount)
            .put("opportunity_reference_count", opportunityReferenceCount)
            .put("pose_strict_win_reference_count", poseStrictWinReferenceCount)
            .put("tie_reference_count", tieReferenceCount)
            .put("pose_loss_reference_count", poseLossReferenceCount)
            .put("current_focused_token_sum", currentFocusedTokenSum)
            .put("passive_new_focused_token_sum", passiveNewFocusedTokenSum)
            .put("pose_new_focused_token_sum", poseNewFocusedTokenSum)
            .put("current_all_token_sum", currentAllTokenSum)
            .put("passive_new_all_token_sum", passiveNewAllTokenSum)
            .put("pose_new_all_token_sum", poseNewAllTokenSum)
            .put("passive_new_focused_token_mean", mean(passiveNewFocusedTokenSum))
            .put("pose_new_focused_token_mean", mean(poseNewFocusedTokenSum))
            .put("passive_new_all_token_mean", mean(passiveNewAllTokenSum))
            .put("pose_new_all_token_mean", mean(poseNewAllTokenSum))
            .put("exact_passive_payload_lookup_count", exactPassivePayloadLookupCount)
            .put("exact_pose_payload_lookup_count", exactPosePayloadLookupCount)
            .put("selected_payload_lookup_miss_count", selectedPayloadLookupMissCount)
            .put("source_identity_mismatch_count", sourceIdentityMismatchCount)
            .put("continuity_reset_count", continuityResetCount)
            .put("continuity_reset_eviction_count", continuityResetEvictionCount)
            .put("minimum_selected_gap_ns", minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE } ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", maximumSelectedGapNs.takeIf { evaluableReferenceCount > 0 } ?: JSONObject.NULL)
            .put("maximum_selected_translation_m", maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", maximumSelectedYawDeltaRad)
            .put("decode_latency_ms", latencySummary(decodeLatencyMs))
            .put("detector_preprocess_latency_ms", latencySummary(detectorPreprocessLatencyMs))
            .put("detector_inference_latency_ms", latencySummary(detectorInferenceLatencyMs))
            .put("detector_postprocess_latency_ms", latencySummary(detectorPostprocessLatencyMs))
            .put("detector_total_latency_ms", latencySummary(detectorTotalLatencyMs))
            .put("pose_admission_failure_counts", JSONObject(poseAdmissionFailureCounts as Map<*, *>))
            .put("selection_failure_counts", JSONObject(selectionFailureCounts as Map<*, *>))
            .put("model_failure_counts", JSONObject(modelFailureCounts as Map<*, *>))
            .put("resource_error_counts", JSONObject(resourceErrorCounts as Map<*, *>))
            .put("positive_support_gate_pass", positiveSupportReferenceCount > 0)
            .put("structural_gate_pass", structuralGatePass)
            .put("privacy", JSONObject()
                .put("raw_images_persisted", false)
                .put("detections_or_boxes_persisted", false)
                .put("scene_address_persisted", false)
                .put("person_identity_persisted", false))
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("screen_space_positive_evidence_only", true)
                .put("absence_is_safe", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_YOLO_POSITIVE_EVIDENCE_SCENE_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("Frozen-YOLO scene shard failed a structural or runtime gate", structuralGatePass)
    }

    private fun evidenceFor(
        payload: TaroOwnedRgbPayload,
        rotationDegrees: Int,
        adapter: TaroOwnedRgbYoloEvidenceAdapter,
        cache: MutableMap<UstrfFrameStamp, TaroPositiveVisualReceipt>,
        failures: MutableMap<String, Int>,
        recordLatency: (TaroPositiveVisualReceipt) -> Unit
    ): TaroPositiveVisualReceipt? {
        cache[payload.sourceFrame]?.let { return it }
        return when (val result = adapter.observe(payload, rotationDegrees)) {
            is TaroOwnedRgbYoloEvidenceResult.Available -> result.receipt.also {
                cache[payload.sourceFrame] = it
                recordLatency(it)
            }
            is TaroOwnedRgbYoloEvidenceResult.Unavailable -> {
                increment(failures, result.failure.name)
                null
            }
        }
    }

    private fun removePayloadsNotRetained(
        payloads: MutableMap<UstrfFrameStamp, TaroOwnedRgbPayload>,
        history: TaroOwnedRgbPayloadHistory
    ) {
        payloads.keys.removeAll { history.lookupExact(it) == null }
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

    private fun detectorRotationDegrees(
        sensorOrientationDegrees: Int?,
        targetRotation: Int,
        lensFacing: Int?
    ): Int? {
        val sensor = sensorOrientationDegrees ?: return null
        val targetDegrees = when (targetRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return null
        }
        return if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensor + targetDegrees) % 360
        } else {
            (sensor - targetDegrees + 360) % 360
        }.takeIf { it in setOf(0, 90, 180, 270) }
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

    private fun assetSha256(context: Context, assetName: String): String =
        context.assets.open(assetName).use(::sha256)

    private fun sha256(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(HASH_BUFFER_BYTES)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }.uppercase()
    }

    private fun increment(counts: MutableMap<String, Int>, key: String) {
        counts[key] = (counts[key] ?: 0) + 1
    }

    private fun latencySummary(values: List<Double>) = JSONObject()
        .put("sample_count", values.size)
        .put("p50", percentileOrNull(values, .50) ?: JSONObject.NULL)
        .put("p95", percentileOrNull(values, .95) ?: JSONObject.NULL)

    private fun percentileOrNull(values: List<Double>, percentile: Double): Double? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * percentile).roundToInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private companion object {
        const val ARG_SCENE_ID = "taroSceneId"
        const val ARG_TARGET_REFERENCES = "taroYoloTargetReferences"
        const val ARG_FRAME_ATTEMPTS = "taroYoloFrameAttempts"
        val SCENE_ID_REGEX = Regex("[A-Za-z0-9_-]{1,32}")
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val MINIMUM_FRAME_ATTEMPTS = 300
        const val DEFAULT_FRAME_ATTEMPTS = 600
        const val MAXIMUM_FRAME_ATTEMPTS = 1_800
        const val MINIMUM_EVALUABLE_REFERENCES = 20
        const val DEFAULT_TARGET_REFERENCES = 30
        const val MAXIMUM_EVALUABLE_REFERENCES = 40
        const val DEVICE_TEST_TIMEOUT_SECONDS = 180L
        const val FRAME_SETTLE_MS = 10L
        const val MAXIMUM_HISTORY_AGE_NS = 1_000_000_000L
        const val MAXIMUM_HISTORY_BYTES = 32L * 1024L * 1024L
        const val MAXIMUM_DETECTOR_TOTAL_P95_MS = 100.0
        const val MINIMUM_OBSERVED_TRANSLATION_M = .02f
        const val MINIMUM_OBSERVED_YAW_DELTA_RAD = .0349066f
        const val DETECTOR_ANALYSIS_TARGET_ROTATION = Surface.ROTATION_0
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN_PREFIX = "yolo-positive-evidence-shadow-r0"
        const val FROZEN_MODEL_SHA256 = "00EDB41A528B0A7E709C4AF8CE3E685491492C4539274804E5CFC17A1A867CD2"
        const val FROZEN_LABELS_SHA256 = "BD17F1EE35D5F3C862A4894605855ABBB9DDA4B0621FDB0AC4C2C8C7BB7E730A"
        const val HASH_BUFFER_BYTES = 1024 * 1024
        const val REPORT_KEY = "taro_arcore_yolo_positive_evidence_scene"
        const val TAG = "UstrfShadowBenchmark"
    }
}
