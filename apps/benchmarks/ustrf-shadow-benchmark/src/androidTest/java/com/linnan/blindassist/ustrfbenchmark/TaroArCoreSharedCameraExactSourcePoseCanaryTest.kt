package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Context
import android.content.Intent
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
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
import java.util.EnumSet
import java.util.TreeMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Isolated device canary for the source seam missing from the default CameraX app path.
 *
 * ARCore owns tracking while Camera2 feeds one registered 640x480 YUV app surface. Images are
 * copied and closed on a bounded handler thread. Only exact Image.timestamp == Frame.timestamp
 * joins may enter the existing anchor-relative selector. No model, app UI, risk, or guidance path
 * is reachable from this test.
 */
@RunWith(AndroidJUnit4::class)
class TaroArCoreSharedCameraExactSourcePoseCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun sharedCameraAppSurface_producesExactSourcePosePairsAndSelections() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val maximumFrameAttempts = arguments.getString("taroSharedCameraMaximumFrameAttempts")
            ?.toIntOrNull()
            ?.coerceIn(MINIMUM_FRAME_ATTEMPTS, MAXIMUM_FRAME_ATTEMPTS)
            ?: DEFAULT_MAXIMUM_FRAME_ATTEMPTS
        val timeoutSeconds = arguments.getString("taroSharedCameraTimeoutSeconds")
            ?.toLongOrNull()
            ?.coerceIn(MINIMUM_TIMEOUT_SECONDS, MAXIMUM_TIMEOUT_SECONDS)
            ?: DEFAULT_TIMEOUT_SECONDS
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .putExtra(
                    UstrfArCoreBenchmarkActivity.EXTRA_STATUS_TEXT,
                    "TARO SharedCamera source canary active\n" +
                        "Autonomous stationary capture; no user action is requested.\n" +
                        "No model, risk, or navigation command will run."
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

        val state = CanaryState()
        try {
            activity.runOnGlThreadAndWait(timeoutSeconds = timeoutSeconds + GL_TIMEOUT_MARGIN_SECONDS) {
                runSharedCameraCanary(
                    activity = activity,
                    maximumFrameAttempts = maximumFrameAttempts,
                    timeoutSeconds = timeoutSeconds,
                    state = state
                )
            }
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val pairerBeforeClose = state.pairerSnapshotBeforeClose
        val pairerAfterClose = state.pairerSnapshotAfterClose
        val resourceErrorCount = state.resourceErrorCount()
        val allResourcesClosed = state.imageReaderClosed &&
            state.captureSessionClosed &&
            state.cameraDeviceClosed &&
            state.arcoreSessionClosed &&
            state.handlerThreadTerminated &&
            state.imagesAcquired.get() == state.imagesClosed.get() &&
            state.maximumImagesAcquiredConcurrently.get() <= MAXIMUM_IMAGES
        val resourceGatePass = state.runtimeFailure.get() == null &&
            resourceErrorCount == 0 &&
            allResourcesClosed
        val denominatorGatePass = state.exactSourcePosePairCount >= TARGET_EXACT_SOURCE_POSE_PAIRS &&
            state.availableSelectionCount >= MINIMUM_POSE_DIVERSE_SELECTIONS
        val exactIdentityGatePass = state.nearestTimestampFallbackCount == 0 &&
            state.crossAnchorEpochHistoryCount == 0 &&
            state.selectionReferenceIdentityMismatchCount == 0 &&
            state.selectedPayloadLookupMissCount == 0 &&
            state.selectedReceiptIdentityMismatchCount == 0
        val terminal = when {
            !resourceGatePass -> TERMINAL_RUNTIME_FAIL
            !denominatorGatePass || !exactIdentityGatePass -> TERMINAL_NOT_EVALUABLE
            else -> TERMINAL_PASS
        }
        val report = JSONObject()
            .put("schema", "blindassist_taro_arcore_shared_camera_exact_source_pose_canary_v1")
            .put("protocol_id", "TARO_ARCORE_SHARED_CAMERA_SOURCE_CANARY_R0")
            .put("package", instrumentation.targetContext.packageName)
            .put("availability", availability.name)
            .put("session_feature", "SHARED_CAMERA")
            .put("capture_template", "TEMPLATE_RECORD")
            .put("app_surface", "640x480 YUV_420_888 ImageReader maxImages=2")
            .put("frame_attempts", state.frameAttempts)
            .put("timeout_seconds", timeoutSeconds)
            .put("timed_out", state.timedOut)
            .put("shared_camera_session_created", state.sharedCameraSessionCreated)
            .put("camera2_session_configured", state.camera2SessionConfigured)
            .put("arcore_session_resumed", state.arcoreSessionResumed)
            .put("image_available_callback_count", state.imageAvailableCallbackCount.get())
            .put("image_acquired_count", state.imagesAcquired.get())
            .put("image_closed_count", state.imagesClosed.get())
            .put("maximum_images_acquired_concurrently", state.maximumImagesAcquiredConcurrently.get())
            .put("null_image_acquire_count", state.nullImageAcquireCount.get())
            .put("exact_source_pose_pair_count", state.exactSourcePosePairCount)
            .put("available_pose_diverse_selection_count", state.availableSelectionCount)
            .put("exact_selected_payload_lookup_count", state.exactSelectedPayloadLookupCount)
            .put("selected_payload_lookup_miss_count", state.selectedPayloadLookupMissCount)
            .put("selection_reference_identity_mismatch_count", state.selectionReferenceIdentityMismatchCount)
            .put("selected_receipt_identity_mismatch_count", state.selectedReceiptIdentityMismatchCount)
            .put("nearest_timestamp_fallback_count", state.nearestTimestampFallbackCount)
            .put("cross_anchor_epoch_history_count", state.crossAnchorEpochHistoryCount)
            .put("continuity_reset_count", state.continuityResetCount)
            .put("distinct_content_hash_count", state.distinctContentHashes.size)
            .put("image_width_px", state.imageWidthPx ?: JSONObject.NULL)
            .put("image_height_px", state.imageHeightPx ?: JSONObject.NULL)
            .put("image_format", state.imageFormat ?: JSONObject.NULL)
            .put("peak_owned_bytes", state.peakOwnedBytes)
            .put("peak_history_bytes", state.peakHistoryBytes)
            .put("peak_history_entry_count", state.peakHistoryEntryCount)
            .put("maximum_history_age_observed_ns", state.maximumHistoryAgeObservedNs)
            .put("minimum_selected_gap_ns", state.minimumSelectedGapNs.takeIf { it != Long.MAX_VALUE }
                ?: JSONObject.NULL)
            .put("maximum_selected_gap_ns", state.maximumSelectedGapNs.takeIf {
                state.availableSelectionCount > 0
            } ?: JSONObject.NULL)
            .put("maximum_selected_translation_m", state.maximumSelectedTranslationM)
            .put("maximum_selected_yaw_delta_rad", state.maximumSelectedYawDeltaRad)
            .put("pairing_snapshot_before_close", pairerSnapshotJson(pairerBeforeClose))
            .put("pairing_snapshot_after_close", pairerSnapshotJson(pairerAfterClose))
            .put("pose_admission_failure_counts", state.counterJson(state.poseAdmissionFailureCounts))
            .put("selection_failure_counts", state.counterJson(state.selectionFailureCounts))
            .put("resource_error_counts", state.counterJson(state.resourceErrorCounts))
            .put("runtime_failure", state.runtimeFailure.get()?.let {
                "${it.javaClass.simpleName}:${it.message}"
            } ?: JSONObject.NULL)
            .put("resource_receipt", JSONObject()
                .put("image_reader_closed", state.imageReaderClosed)
                .put("capture_session_closed", state.captureSessionClosed)
                .put("camera_device_closed", state.cameraDeviceClosed)
                .put("arcore_session_closed", state.arcoreSessionClosed)
                .put("handler_thread_terminated", state.handlerThreadTerminated)
                .put("all_images_and_camera_resources_closed", allResourcesClosed))
            .put("gates", JSONObject()
                .put("exact_source_pose_pair_target_reached",
                    state.exactSourcePosePairCount >= TARGET_EXACT_SOURCE_POSE_PAIRS)
                .put("minimum_pose_diverse_selections_reached",
                    state.availableSelectionCount >= MINIMUM_POSE_DIVERSE_SELECTIONS)
                .put("zero_nearest_timestamp_fallbacks", state.nearestTimestampFallbackCount == 0)
                .put("zero_cross_anchor_epoch_history", state.crossAnchorEpochHistoryCount == 0)
                .put("zero_image_reader_or_resource_errors", resourceErrorCount == 0)
                .put("all_images_and_camera_resources_closed", allResourcesClosed)
                .put("denominator_gate_pass", denominatorGatePass)
                .put("exact_identity_gate_pass", exactIdentityGatePass)
                .put("resource_gate_pass", resourceGatePass))
            .put("terminal", terminal)
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("raw_images_persisted", false)
                .put("model_run", false)
                .put("app_ui_connected", false)
                .put("risk_field_fusion_authorized", false)
                .put("guidance_authorized", false)
                .put("default_app_changed", false)
                .put("production_authorized", false))
        Log.i(TAG, "TARO_ARCORE_SHARED_CAMERA_SOURCE_CANARY_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertEquals("SharedCamera source canary did not reach its frozen PASS terminal", TERMINAL_PASS, terminal)
    }

    @Suppress("DEPRECATION")
    private fun runSharedCameraCanary(
        activity: UstrfArCoreBenchmarkActivity,
        maximumFrameAttempts: Int,
        timeoutSeconds: Long,
        state: CanaryState
    ) {
        var session: Session? = null
        var imageReader: ImageReader? = null
        var handlerThread: HandlerThread? = null
        var arcoreResumed = false
        val cameraDeviceRef = AtomicReference<CameraDevice?>()
        val captureSessionRef = AtomicReference<CameraCaptureSession?>()
        val cameraClosed = CountDownLatch(1)
        val captureSessionClosed = CountDownLatch(1)
        val cameraConfigured = CountDownLatch(1)
        val cameraActive = CountDownLatch(1)
        try {
            handlerThread = HandlerThread("taro-shared-camera-source").also { it.start() }
            val backgroundHandler = Handler(handlerThread.looper)
            val createdSession = Session(activity, EnumSet.of(Session.Feature.SHARED_CAMERA))
            session = createdSession
            state.sharedCameraSessionCreated = true
            createdSession.configure(Config(createdSession).apply {
                updateMode = Config.UpdateMode.BLOCKING
                focusMode = Config.FocusMode.AUTO
            })
            createdSession.setCameraTextureName(activity.cameraTextureName())
            val sharedCamera = createdSession.sharedCamera
            val cameraId = createdSession.cameraConfig.cameraId
            imageReader = ImageReader.newInstance(
                APP_IMAGE_WIDTH_PX,
                APP_IMAGE_HEIGHT_PX,
                ImageFormat.YUV_420_888,
                MAXIMUM_IMAGES
            )
            imageReader.setOnImageAvailableListener(
                { reader -> state.onImageAvailable(reader) },
                backgroundHandler
            )
            sharedCamera.setAppSurfaces(cameraId, listOf(imageReader.surface))
            val surfaces = sharedCamera.arCoreSurfaces.toMutableList().apply {
                add(imageReader.surface)
            }
            val captureRequestRef = AtomicReference<CaptureRequest?>()
            val captureCallback = object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    cameraSession: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) = Unit

                override fun onCaptureFailed(
                    cameraSession: CameraCaptureSession,
                    request: CaptureRequest,
                    failure: CaptureFailure
                ) {
                    state.fail("CAMERA_CAPTURE_FAILED", IllegalStateException(
                        "frame=${failure.frameNumber},reason=${failure.reason}"
                    ))
                }

                override fun onCaptureBufferLost(
                    cameraSession: CameraCaptureSession,
                    request: CaptureRequest,
                    target: android.view.Surface,
                    frameNumber: Long
                ) {
                    state.fail("CAMERA_CAPTURE_BUFFER_LOST", IllegalStateException("frame=$frameNumber"))
                }
            }
            val cameraSessionStateCallback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(configuredSession: CameraCaptureSession) {
                    captureSessionRef.set(configuredSession)
                    try {
                        configuredSession.setRepeatingRequest(
                            requireNotNull(captureRequestRef.get()),
                            captureCallback,
                            backgroundHandler
                        )
                        state.camera2SessionConfigured = true
                    } catch (throwable: Throwable) {
                        state.fail("SET_REPEATING_REQUEST_FAILED", throwable)
                    } finally {
                        cameraConfigured.countDown()
                    }
                }

                override fun onConfigureFailed(failedSession: CameraCaptureSession) {
                    captureSessionRef.set(failedSession)
                    state.fail("CAMERA_SESSION_CONFIGURE_FAILED", IllegalStateException("onConfigureFailed"))
                    cameraConfigured.countDown()
                    cameraActive.countDown()
                }

                override fun onActive(activeSession: CameraCaptureSession) {
                    cameraActive.countDown()
                }

                override fun onClosed(closedSession: CameraCaptureSession) {
                    state.captureSessionClosed = true
                    captureSessionClosed.countDown()
                }
            }
            val wrappedSessionCallback = sharedCamera.createARSessionStateCallback(
                cameraSessionStateCallback,
                backgroundHandler
            )
            val cameraDeviceCallback = object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    cameraDeviceRef.set(device)
                    try {
                        val requestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
                        surfaces.forEach(requestBuilder::addTarget)
                        captureRequestRef.set(requestBuilder.build())
                        device.createCaptureSession(surfaces, wrappedSessionCallback, backgroundHandler)
                    } catch (throwable: Throwable) {
                        state.fail("CREATE_CAPTURE_SESSION_FAILED", throwable)
                        cameraConfigured.countDown()
                        cameraActive.countDown()
                    }
                }

                override fun onDisconnected(device: CameraDevice) {
                    state.fail("CAMERA_DEVICE_DISCONNECTED", IllegalStateException(device.id))
                    device.close()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    state.fail("CAMERA_DEVICE_ERROR_$error", IllegalStateException(device.id))
                    device.close()
                    cameraConfigured.countDown()
                    cameraActive.countDown()
                }

                override fun onClosed(device: CameraDevice) {
                    state.cameraDeviceClosed = true
                    cameraClosed.countDown()
                }
            }
            val wrappedDeviceCallback = sharedCamera.createARDeviceStateCallback(
                cameraDeviceCallback,
                backgroundHandler
            )
            val cameraManager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            cameraManager.openCamera(cameraId, wrappedDeviceCallback, backgroundHandler)
            check(cameraConfigured.await(CAMERA_LIFECYCLE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                "Camera2 capture session configuration timed out"
            }
            state.runtimeFailure.get()?.let { throw it }
            check(cameraActive.await(CAMERA_LIFECYCLE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                "Camera2 capture session did not become active"
            }
            createdSession.resume()
            arcoreResumed = true
            state.arcoreSessionResumed = true
            sharedCamera.setCaptureCallback(captureCallback, backgroundHandler)

            val poseAdapter = TaroArCoreAnchorPoseAdmissionAdapter(
                session = createdSession,
                sessionToken = SESSION_TOKEN
            )
            val deadlineMs = SystemClock.elapsedRealtime() + TimeUnit.SECONDS.toMillis(timeoutSeconds)
            try {
                for (attempt in 0 until maximumFrameAttempts) {
                    if (SystemClock.elapsedRealtime() >= deadlineMs) {
                        state.timedOut = true
                        break
                    }
                    if (state.runtimeFailure.get() != null) break
                    val frame = createdSession.update()
                    state.frameAttempts++
                    val sourceFrame = UstrfFrameStamp(
                        frameId = attempt.toLong(),
                        capturedAtNs = frame.timestamp,
                        coordinateFrame = ARCORE_CAMERA_FRAME
                    )
                    when (val admission = poseAdapter.observe(frame, sourceFrame)) {
                        is TaroArCoreAnchorPoseAdmission.Available -> state.observePose(sourceFrame, admission)
                        is TaroArCoreAnchorPoseAdmission.Unavailable -> {
                            state.increment(state.poseAdmissionFailureCounts, admission.failure.name)
                            state.resetContinuity()
                        }
                    }
                    state.drainAndProcessMatchedPayloads()
                    if (state.exactSourcePosePairCount >= TARGET_EXACT_SOURCE_POSE_PAIRS &&
                        state.availableSelectionCount >= MINIMUM_POSE_DIVERSE_SELECTIONS
                    ) {
                        break
                    }
                }
                state.stopAcceptingImagesAndDrain(imageReader, backgroundHandler)
                state.drainAndProcessMatchedPayloads()
            } finally {
                poseAdapter.close()
            }
        } catch (throwable: Throwable) {
            state.fail("SHARED_CAMERA_RUNTIME", throwable)
        } finally {
            state.acceptImages.set(false)
            try {
                imageReader?.setOnImageAvailableListener(null, null)
            } catch (throwable: Throwable) {
                state.fail("IMAGE_READER_LISTENER_CLEAR_FAILED", throwable)
            }
            if (arcoreResumed) {
                try {
                    session?.pause()
                } catch (throwable: Throwable) {
                    state.fail("ARCORE_PAUSE_FAILED", throwable)
                }
            }
            val captureSession = captureSessionRef.get()
            if (captureSession != null) {
                try {
                    captureSession.stopRepeating()
                    captureSession.abortCaptures()
                } catch (throwable: Throwable) {
                    state.fail("CAMERA_CAPTURE_STOP_FAILED", throwable)
                } finally {
                    captureSession.close()
                }
                if (!awaitCleanup(captureSessionClosed)) {
                    state.fail("CAMERA_SESSION_CLOSE_TIMEOUT", IllegalStateException("onClosed not received"))
                }
            } else {
                state.captureSessionClosed = !state.camera2SessionConfigured
            }
            val cameraDevice = cameraDeviceRef.get()
            if (cameraDevice != null) {
                cameraDevice.close()
                if (!awaitCleanup(cameraClosed)) {
                    state.fail("CAMERA_DEVICE_CLOSE_TIMEOUT", IllegalStateException("onClosed not received"))
                }
            } else {
                state.cameraDeviceClosed = !state.camera2SessionConfigured
            }
            try {
                imageReader?.close()
                state.imageReaderClosed = imageReader != null
            } catch (throwable: Throwable) {
                state.fail("IMAGE_READER_CLOSE_FAILED", throwable)
            }
            try {
                session?.close()
                state.arcoreSessionClosed = session != null
            } catch (throwable: Throwable) {
                state.fail("ARCORE_CLOSE_FAILED", throwable)
            }
            state.closeOwnedState()
            handlerThread?.quitSafely()
            try {
                handlerThread?.join(TimeUnit.SECONDS.toMillis(CAMERA_LIFECYCLE_TIMEOUT_SECONDS))
                state.handlerThreadTerminated = handlerThread?.isAlive == false
                if (!state.handlerThreadTerminated) {
                    state.fail("CAMERA_HANDLER_THREAD_CLOSE_TIMEOUT", IllegalStateException("thread still alive"))
                }
            } catch (interrupted: InterruptedException) {
                Thread.currentThread().interrupt()
                state.fail("CAMERA_HANDLER_THREAD_INTERRUPTED", interrupted)
            }
        }
    }

    private class CanaryState {
        val pairer = TaroSharedCameraExactSourcePosePairer(
            maximumPendingAgeNs = MAXIMUM_RETAINED_AGE_NS,
            maximumPendingImageBytes = MAXIMUM_OWNED_BYTES
        )
        private val history = TaroOwnedRgbPayloadHistory(
            maximumRetainedAgeNs = MAXIMUM_RETAINED_AGE_NS,
            maximumRetainedBytes = MAXIMUM_OWNED_BYTES
        )
        private val selector = TaroPoseDiverseFrameSelector(enabled = true)
        private val pairingGate = Any()
        private val matchedPayloads = ConcurrentLinkedQueue<TaroOwnedRgbPayload>()
        private val readyByTimestamp = TreeMap<Long, TaroOwnedRgbPayload>()
        val acceptImages = AtomicBoolean(true)
        val runtimeFailure = AtomicReference<Throwable?>()
        val resourceErrorCounts = ConcurrentHashMap<String, AtomicInteger>()
        val poseAdmissionFailureCounts = ConcurrentHashMap<String, AtomicInteger>()
        val selectionFailureCounts = ConcurrentHashMap<String, AtomicInteger>()
        val imageAvailableCallbackCount = AtomicInteger()
        val imagesAcquired = AtomicInteger()
        val imagesClosed = AtomicInteger()
        val nullImageAcquireCount = AtomicInteger()
        private val activeImages = AtomicInteger()
        val maximumImagesAcquiredConcurrently = AtomicInteger()
        var frameAttempts = 0
        var exactSourcePosePairCount = 0
        var availableSelectionCount = 0
        var exactSelectedPayloadLookupCount = 0
        var selectedPayloadLookupMissCount = 0
        var selectionReferenceIdentityMismatchCount = 0
        var selectedReceiptIdentityMismatchCount = 0
        val nearestTimestampFallbackCount = 0
        var crossAnchorEpochHistoryCount = 0
        var continuityResetCount = 0
        var imageWidthPx: Int? = null
        var imageHeightPx: Int? = null
        var imageFormat: Int? = null
        var peakOwnedBytes = 0L
        var peakHistoryBytes = 0L
        var peakHistoryEntryCount = 0
        var maximumHistoryAgeObservedNs = 0L
        var minimumSelectedGapNs = Long.MAX_VALUE
        var maximumSelectedGapNs = 0L
        var maximumSelectedTranslationM = 0f
        var maximumSelectedYawDeltaRad = 0f
        val distinctContentHashes = linkedSetOf<String>()
        var timedOut = false
        var sharedCameraSessionCreated = false
        var camera2SessionConfigured = false
        var arcoreSessionResumed = false
        var imageReaderClosed = false
        var captureSessionClosed = false
        var cameraDeviceClosed = false
        var arcoreSessionClosed = false
        var handlerThreadTerminated = false
        var pairerSnapshotBeforeClose = pairer.snapshot()
        var pairerSnapshotAfterClose = pairer.snapshot()
        private var lastProcessedTimestampNs = -1L
        private var activeAnchorWorldFrame: String? = null

        fun onImageAvailable(reader: ImageReader) {
            imageAvailableCallbackCount.incrementAndGet()
            val image = try {
                reader.acquireLatestImage()
            } catch (throwable: Throwable) {
                fail("IMAGE_READER_ACQUIRE_FAILED", throwable)
                null
            }
            if (image == null) {
                nullImageAcquireCount.incrementAndGet()
                return
            }
            imagesAcquired.incrementAndGet()
            val concurrent = activeImages.incrementAndGet()
            maximumImagesAcquiredConcurrently.accumulateAndGet(concurrent, ::maxOf)
            try {
                if (acceptImages.get()) {
                    val ownedImage = copyOwnedImage(image)
                    synchronized(pairingGate) {
                        enqueue(pairer.observeImage(ownedImage))
                    }
                }
            } catch (throwable: Throwable) {
                fail("IMAGE_READER_COPY_FAILED", throwable)
            } finally {
                try {
                    image.close()
                } finally {
                    activeImages.decrementAndGet()
                    imagesClosed.incrementAndGet()
                }
            }
        }

        fun observePose(
            sourceFrame: UstrfFrameStamp,
            admission: TaroArCoreAnchorPoseAdmission.Available
        ) {
            synchronized(pairingGate) {
                enqueue(pairer.observePose(sourceFrame, admission))
            }
        }

        fun resetContinuity() {
            synchronized(pairingGate) {
                val pairerReset = pairer.reset()
                matchedPayloads.clear()
                readyByTimestamp.clear()
                val historyReset = history.reset()
                if (pairerReset.evictedImageCount > 0 || pairerReset.evictedPoseCount > 0 ||
                    historyReset.evictedEntryCount > 0
                ) {
                    continuityResetCount++
                }
                lastProcessedTimestampNs = -1L
                activeAnchorWorldFrame = null
            }
        }

        fun drainAndProcessMatchedPayloads() {
            synchronized(pairingGate) {
                while (true) {
                    val payload = matchedPayloads.poll() ?: break
                    if (readyByTimestamp.put(payload.sourceFrame.capturedAtNs, payload) != null) {
                        fail("DUPLICATE_MATCHED_PAYLOAD", IllegalStateException(
                            payload.sourceFrame.capturedAtNs.toString()
                        ))
                    }
                }
            }
            while (readyByTimestamp.isNotEmpty()) {
                val payload = requireNotNull(readyByTimestamp.pollFirstEntry()).value
                processPayload(payload)
            }
        }

        fun stopAcceptingImagesAndDrain(reader: ImageReader?, handler: Handler) {
            acceptImages.set(false)
            reader?.setOnImageAvailableListener(null, null)
            val drained = CountDownLatch(1)
            check(handler.post { drained.countDown() }) { "Camera handler rejected drain barrier" }
            check(drained.await(CAMERA_LIFECYCLE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                "Image callback drain timed out"
            }
        }

        fun closeOwnedState() {
            pairerSnapshotBeforeClose = pairer.snapshot()
            synchronized(pairingGate) {
                pairer.reset()
                matchedPayloads.clear()
                readyByTimestamp.clear()
            }
            history.close()
            pairerSnapshotAfterClose = pairer.snapshot()
        }

        fun fail(key: String, throwable: Throwable) {
            increment(resourceErrorCounts, key)
            runtimeFailure.compareAndSet(null, throwable)
        }

        fun increment(counts: ConcurrentHashMap<String, AtomicInteger>, key: String) {
            counts.computeIfAbsent(key) { AtomicInteger() }.incrementAndGet()
        }

        fun resourceErrorCount(): Int = resourceErrorCounts.values.sumOf(AtomicInteger::get)

        fun counterJson(counts: ConcurrentHashMap<String, AtomicInteger>): JSONObject =
            JSONObject(counts.mapValues { it.value.get() } as Map<*, *>)

        private fun enqueue(receipt: TaroSharedCameraPairingReceipt) {
            receipt.matchedPayload?.let(matchedPayloads::add)
        }

        private fun processPayload(payload: TaroOwnedRgbPayload) {
            val timestampNs = payload.sourceFrame.capturedAtNs
            if (timestampNs <= lastProcessedTimestampNs) {
                fail("NON_MONOTONIC_EXACT_PAIR", IllegalStateException(
                    "last=$lastProcessedTimestampNs,current=$timestampNs"
                ))
                return
            }
            lastProcessedTimestampNs = timestampNs
            if (payload.imageWidthPx != APP_IMAGE_WIDTH_PX ||
                payload.imageHeightPx != APP_IMAGE_HEIGHT_PX ||
                payload.imageFormat != ImageFormat.YUV_420_888
            ) {
                fail("APP_SURFACE_FORMAT_MISMATCH", IllegalStateException(
                    "${payload.imageWidthPx}x${payload.imageHeightPx}:${payload.imageFormat}"
                ))
                return
            }
            val worldFrame = payload.anchorPose.worldFrame
            if (activeAnchorWorldFrame != null && activeAnchorWorldFrame != worldFrame &&
                history.retainedEntryCount > 0
            ) {
                crossAnchorEpochHistoryCount++
                history.reset()
            }
            activeAnchorWorldFrame = worldFrame
            exactSourcePosePairCount++
            imageWidthPx = payload.imageWidthPx
            imageHeightPx = payload.imageHeightPx
            imageFormat = payload.imageFormat
            distinctContentHashes += payload.contentSha256
            val advance = history.advanceTo(timestampNs)
            if (advance.retainedBytes > MAXIMUM_OWNED_BYTES) {
                fail("HISTORY_ADVANCE_BYTE_BOUND", IllegalStateException(advance.retainedBytes.toString()))
            }
            when (val selection = selector.select(
                referenceFrame = payload.sourceFrame,
                referencePose = payload.anchorPose,
                bufferedFrames = history.bufferedPoseFrames()
            )) {
                is TaroPoseDiverseSelection.Available -> {
                    availableSelectionCount++
                    if (selection.referenceFrame != payload.sourceFrame) {
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
                    }
                    minimumSelectedGapNs = minOf(minimumSelectedGapNs, selection.gapNs)
                    maximumSelectedGapNs = maxOf(maximumSelectedGapNs, selection.gapNs)
                    maximumSelectedTranslationM = maxOf(maximumSelectedTranslationM, selection.translationM)
                    maximumSelectedYawDeltaRad = maxOf(maximumSelectedYawDeltaRad, selection.yawDeltaRad)
                }
                is TaroPoseDiverseSelection.Unavailable ->
                    increment(selectionFailureCounts, selection.failure.name)
            }
            val append = history.append(payload)
            peakHistoryBytes = maxOf(peakHistoryBytes, append.retainedBytes)
            peakHistoryEntryCount = maxOf(peakHistoryEntryCount, append.retainedEntryCount)
            history.oldestRetainedTimestampNs?.let { oldest ->
                maximumHistoryAgeObservedNs = maxOf(maximumHistoryAgeObservedNs, timestampNs - oldest)
            }
            val pendingBytes = pairer.snapshot().pendingImageBytes
            peakOwnedBytes = maxOf(peakOwnedBytes, append.retainedBytes + pendingBytes)
            if (peakOwnedBytes > MAXIMUM_OWNED_BYTES) {
                fail("COMBINED_OWNED_BYTE_BOUND", IllegalStateException(peakOwnedBytes.toString()))
            }
        }

        private fun copyOwnedImage(image: Image): TaroSharedCameraOwnedYuvFrame {
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
            return TaroSharedCameraOwnedYuvFrame(
                timestampNs = image.timestamp,
                imageWidthPx = image.width,
                imageHeightPx = image.height,
                imageFormat = image.format,
                planes = planes,
                contentSha256 = digest.digest().joinToString("") { "%02x".format(it) }
            )
        }
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

    private fun awaitCleanup(latch: CountDownLatch): Boolean =
        try {
            latch.await(CAMERA_LIFECYCLE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
            false
        }

    private fun pairerSnapshotJson(snapshot: TaroSharedCameraPairerSnapshot) = JSONObject()
        .put("exact_match_count", snapshot.exactMatchCount)
        .put("pending_image_count", snapshot.pendingImageCount)
        .put("pending_pose_count", snapshot.pendingPoseCount)
        .put("pending_image_bytes", snapshot.pendingImageBytes)
        .put("age_evicted_image_count", snapshot.ageEvictedImageCount)
        .put("age_evicted_pose_count", snapshot.ageEvictedPoseCount)
        .put("byte_cap_evicted_image_count", snapshot.byteCapEvictedImageCount)
        .put("stale_input_rejected_count", snapshot.staleInputRejectedCount)
        .put("duplicate_input_rejected_count", snapshot.duplicateInputRejectedCount)

    private companion object {
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val APP_IMAGE_WIDTH_PX = 640
        const val APP_IMAGE_HEIGHT_PX = 480
        const val MAXIMUM_IMAGES = 2
        const val MINIMUM_FRAME_ATTEMPTS = 120
        const val DEFAULT_MAXIMUM_FRAME_ATTEMPTS = 900
        const val MAXIMUM_FRAME_ATTEMPTS = 900
        const val MINIMUM_TIMEOUT_SECONDS = 30L
        const val DEFAULT_TIMEOUT_SECONDS = 180L
        const val MAXIMUM_TIMEOUT_SECONDS = 180L
        const val GL_TIMEOUT_MARGIN_SECONDS = 20L
        const val CAMERA_LIFECYCLE_TIMEOUT_SECONDS = 5L
        const val TARGET_EXACT_SOURCE_POSE_PAIRS = 120
        const val MINIMUM_POSE_DIVERSE_SELECTIONS = 100
        const val MAXIMUM_RETAINED_AGE_NS = 1_000_000_000L
        const val MAXIMUM_OWNED_BYTES = 32L * 1024L * 1024L
        const val ARCORE_CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "shared-camera-source-canary-r0"
        const val TERMINAL_PASS = "SHARED_CAMERA_EXACT_SOURCE_POSE_PASS"
        const val TERMINAL_NOT_EVALUABLE = "SHARED_CAMERA_SOURCE_POSE_NOT_EVALUABLE"
        const val TERMINAL_RUNTIME_FAIL = "SHARED_CAMERA_SOURCE_RUNTIME_FAIL_STOP"
        const val REPORT_KEY = "taro_arcore_shared_camera_source_canary"
        const val TAG = "UstrfShadowBenchmark"
    }
}
