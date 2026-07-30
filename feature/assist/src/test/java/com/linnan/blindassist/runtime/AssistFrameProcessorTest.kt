package com.linnan.blindassist.runtime

import android.graphics.Bitmap
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.preferences.PreferenceStore
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.session.DualLoopShadowDisposition
import com.linnan.blindassist.session.DualLoopShadowObservation
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.VisionFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class AssistFrameProcessorTest {
    @Test
    fun inactiveFrameIsDroppedAndClosedOnce() {
        val detector = FakeDetector()
        val frame = FakeVisionFrame()
        val stats = FramePipelineStats()
        val processor = processor(
            detector = detector,
            stats = stats,
            active = false
        )

        processor.process(frame)

        assertEquals(0, detector.detectCalls)
        assertEquals(1, frame.closeCalls)
        assertEquals(1L, stats.snapshot().droppedInactive)
    }

    @Test
    fun detectorUnavailableFrameIsDroppedAndClosedOnce() {
        val detector = FakeDetector(ready = false)
        val frame = FakeVisionFrame()
        val stats = FramePipelineStats()
        val processor = processor(detector = detector, stats = stats)

        processor.process(frame)

        assertEquals(0, detector.detectCalls)
        assertEquals(1, frame.closeCalls)
        assertEquals(1L, stats.snapshot().droppedDetectorUnavailable)
    }

    @Test
    fun disabledDetectionFrameIsDroppedAndClosedOnce() {
        val detector = FakeDetector()
        val frame = FakeVisionFrame()
        val stats = FramePipelineStats()
        val processor = processor(
            detector = detector,
            stats = stats,
            config = runtimeConfig().withDetectionEnabled(false)
        )

        processor.process(frame)

        assertEquals(0, detector.detectCalls)
        assertEquals(1, frame.closeCalls)
        assertEquals(1L, stats.snapshot().droppedInactive)
    }

    @Test
    fun stoppedLifecycleDropsFrameBeforeDetectorRuns() {
        val detector = FakeDetector()
        val frame = FakeVisionFrame()
        val stats = FramePipelineStats()
        val lifecycleGate = AssistRuntimeLifecycleGate()
        lifecycleGate.startSession {}
        lifecycleGate.stopSession {}
        val processor = processor(
            detector = detector,
            stats = stats,
            lifecycleGate = lifecycleGate,
            startLifecycleGate = false
        )

        processor.process(frame)

        assertEquals(0, detector.detectCalls)
        assertEquals(1, frame.closeCalls)
        assertEquals(0L, stats.snapshot().droppedInactive)
    }

    @Test
    fun busyFrameIsDroppedAndClosedOnce() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val detector = FakeDetector(onDetect = {
            started.countDown()
            release.await(2, TimeUnit.SECONDS)
        })
        val stats = FramePipelineStats()
        val processor = processor(detector = detector, stats = stats)
        val firstFrame = FakeVisionFrame()
        val secondFrame = FakeVisionFrame()

        val worker = Thread { processor.process(firstFrame) }
        worker.start()
        assertTrue(started.await(2, TimeUnit.SECONDS))

        processor.process(secondFrame)
        release.countDown()
        worker.join(2000)

        assertEquals(1, secondFrame.closeCalls)
        assertEquals(1L, stats.snapshot().droppedBusy)
    }

    @Test
    fun detectorExceptionReportsCameraFailureAndClosesFrameOnce() {
        val failures = mutableListOf<String>()
        val frame = FakeVisionFrame()
        val processor = processor(
            detector = FakeDetector(error = IllegalStateException("boom")),
            onCameraFailure = { failures += it }
        )

        processor.process(frame)

        assertEquals(1, frame.closeCalls)
        assertEquals(1, failures.size)
        assertTrue(failures.single().contains("boom"))
    }

    @Test
    fun successfulFrameIsProcessedAndClosedOnce() {
        val detector = FakeDetector(inferenceMs = 33L)
        val frame = FakeVisionFrame(width = 4, height = 3)
        val stats = FramePipelineStats()
        val processor = processor(detector = detector, stats = stats)

        processor.process(frame)

        val snapshot = stats.snapshot()
        assertEquals(1, detector.detectCalls)
        assertEquals(1, frame.closeCalls)
        assertEquals(1L, snapshot.processed)
        assertEquals(33L, snapshot.inferenceP50Ms)
        assertEquals(33L, snapshot.inferenceP95Ms)
    }

    @Test
    fun staleDetectorResultCannotReachNewSessionFeedbackOrStats() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val detector = FakeDetector(onDetect = {
            started.countDown()
            release.await(2, TimeUnit.SECONDS)
        })
        val gate = AssistRuntimeLifecycleGate()
        val stats = FramePipelineStats()
        val gateway = FakeFeedbackGateway()
        val frame = FakeVisionFrame()
        val processor = processor(
            detector = detector,
            stats = stats,
            lifecycleGate = gate,
            feedbackGateway = gateway
        )

        val worker = Thread { processor.process(frame) }
        worker.start()
        assertTrue(started.await(1, TimeUnit.SECONDS))
        gate.stopSession { processor.resetSessionStats() }
        gate.startSession { processor.resetSessionStats() }
        release.countDown()
        worker.join(2_000L)

        assertEquals(0, gateway.notifyCalls)
        assertEquals(0L, stats.snapshot().processed)
        assertEquals(1, frame.closeCalls)
    }

    @Test
    fun queuedUiRenderIsIgnoredAfterSessionChanges() {
        val queued = mutableListOf<() -> Unit>()
        val appViewModel = BlindAssistViewModel(UserPreferences(InMemoryPreferenceStore()))
        val initialState = appViewModel.uiState.value
        val gate = AssistRuntimeLifecycleGate()
        val processor = processor(
            lifecycleGate = gate,
            appViewModel = appViewModel,
            runOnUiThread = { queued += it }
        )

        processor.process(FakeVisionFrame())
        assertEquals(1, queued.size)
        gate.stopSession { processor.resetSessionStats() }
        gate.startSession { processor.resetSessionStats() }
        queued.single().invoke()

        assertEquals(initialState, appViewModel.uiState.value)
    }

    @Test
    fun mismatchedDetectorStampFailsClosedAndReportsFailure() {
        val failures = mutableListOf<String>()
        val inputStamp = stamp(1L, 1_000L)
        val frame = FakeVisionFrame(frameStamp = inputStamp)
        val processor = processor(
            detector = FakeDetector(resultStamp = stamp(2L, 1_000L)),
            onCameraFailure = { failures += it }
        )

        processor.process(frame)

        assertEquals(1, frame.closeCalls)
        assertEquals(1, failures.size)
        assertTrue(failures.single().contains("source frame"))
    }

    @Test
    fun baselineAndDualLoopShadowConstructNoUstrfAdaptersWhileExperimentConstructsItsAdapter() {
        val coordinator = AssistSessionCoordinator(feedbackGateway = FakeFeedbackGateway())

        val baseline = UstrfRuntimeAdapters.forMode(AssistRuntimeMode.BASELINE, coordinator)
        val dualLoopShadow = UstrfRuntimeAdapters.forMode(
            AssistRuntimeMode.DUAL_LOOP_SHADOW,
            coordinator
        )
        val experiment = UstrfRuntimeAdapters.forMode(AssistRuntimeMode.USTRF_EXPERIMENT, coordinator)

        assertNull(baseline.experimental)
        assertNull(dualLoopShadow.experimental)
        assertNotNull(experiment.experimental)
    }

    @Test
    fun dualLoopShadowWithoutAdmittedSourceKeepsBaselineFeedbackFrameExact() {
        val baselineGateway = FakeFeedbackGateway()
        val shadowGateway = FakeFeedbackGateway()
        var observed: DualLoopShadowObservation? = null
        val baseline = processor(feedbackGateway = baselineGateway)
        val shadow = processor(
            feedbackGateway = shadowGateway,
            mode = AssistRuntimeMode.DUAL_LOOP_SHADOW,
            onDualLoopShadowObservation = { observed = it }
        )

        baseline.process(FakeVisionFrame())
        shadow.process(FakeVisionFrame())

        assertEquals(baselineGateway.lastRisk, shadowGateway.lastRisk)
        assertEquals(baselineGateway.notifyCalls, shadowGateway.notifyCalls)
        assertEquals(DualLoopShadowDisposition.EVIDENCE_ABSENT, observed?.disposition)
    }

    @Test
    fun dualLoopShadowObserverFailureCannotBreakBaselineFrame() {
        val gateway = FakeFeedbackGateway()
        val frame = FakeVisionFrame()
        val shadow = processor(
            feedbackGateway = gateway,
            mode = AssistRuntimeMode.DUAL_LOOP_SHADOW,
            onDualLoopShadowObservation = { error("observer fixture failure") }
        )

        shadow.process(frame)

        assertEquals(1, frame.closeCalls)
        assertNotNull(gateway.lastRisk)
    }

    @Test
    fun ustrfExperimentUsesFailClosedEvidencePathWhileBaselineKeepsLegacyEmptyFrame() {
        val baselineGateway = FakeFeedbackGateway()
        val experimentGateway = FakeFeedbackGateway()
        val experimentViewModel = BlindAssistViewModel(UserPreferences(InMemoryPreferenceStore()))
        val baseline = processor(feedbackGateway = baselineGateway)
        val experiment = processor(
            feedbackGateway = experimentGateway,
            appViewModel = experimentViewModel,
            mode = AssistRuntimeMode.USTRF_EXPERIMENT
        )

        baseline.process(FakeVisionFrame())
        experiment.process(FakeVisionFrame())

        assertEquals(com.linnan.blindassist.risk.RiskLevel.NONE, baselineGateway.lastRisk?.level)
        assertEquals(com.linnan.blindassist.risk.RiskLevel.HIGH, experimentGateway.lastRisk?.level)
        assertEquals("USTRF 实验代理判断", experimentViewModel.uiState.value.cameraGuidance.explanationHeadline)
        assertTrue(experimentViewModel.uiState.value.cameraGuidance.detail.contains("USTRF实验输入不完整"))
    }

    private fun processor(
        detector: FakeDetector = FakeDetector(),
        stats: FramePipelineStats = FramePipelineStats(),
        lifecycleGate: AssistRuntimeLifecycleGate = AssistRuntimeLifecycleGate(),
        startLifecycleGate: Boolean = true,
        active: Boolean = true,
        config: AssistRuntimeConfig = runtimeConfig(),
        onCameraFailure: (String) -> Unit = {},
        feedbackGateway: FakeFeedbackGateway = FakeFeedbackGateway(),
        appViewModel: BlindAssistViewModel = BlindAssistViewModel(UserPreferences(InMemoryPreferenceStore())),
        runOnUiThread: ((() -> Unit) -> Unit) = { it() },
        decisionClockNs: () -> Long = { 2_000_000_000L },
        mode: AssistRuntimeMode = AssistRuntimeMode.BASELINE,
        onDualLoopShadowObservation: (DualLoopShadowObservation) -> Unit = {}
    ): AssistFrameProcessor {
        val coordinator = AssistSessionCoordinator(feedbackGateway = feedbackGateway)
        if (startLifecycleGate) {
            lifecycleGate.startSession { coordinator.startSession() }
        }
        val configSnapshot = AssistRuntimeConfigSnapshot(config)
        val guidanceFactory = AssistRuntimeGuidanceFactory(detector) { configSnapshot.get() }
        val renderer = AssistRuntimeRenderer(
            appViewModel = appViewModel,
            detector = detector,
            guidanceFactory = guidanceFactory,
            fieldTestSummaryProvider = FieldTestSummaryProvider(coordinator),
            mode = mode,
            performanceLogger = AssistRuntimePerformanceLogger(clockMs = { 0L })
        )
        return AssistFrameProcessor(
            detector = detector,
            coordinator = coordinator,
            configSnapshot = configSnapshot,
            renderer = renderer,
            stats = stats,
            lifecycleGate = lifecycleGate,
            isCameraActive = { active },
            runOnUiThread = runOnUiThread,
            onCameraFailure = onCameraFailure,
            decisionClockNs = decisionClockNs,
            onDualLoopShadowObservation = onDualLoopShadowObservation,
            mode = mode
        )
    }

    private class FakeVisionFrame(
        override val width: Int = 2,
        override val height: Int = 2,
        override val rotationDegrees: Int = 0,
        override val frameStamp: FrameStamp? = null
    ) : VisionFrame {
        var closeCalls = 0
            private set

        override fun close() {
            closeCalls += 1
        }
    }

    private class FakeDetector(
        private val ready: Boolean = true,
        private val error: Exception? = null,
        private val inferenceMs: Long = 22L,
        private val onDetect: () -> Unit = {},
        private val resultStamp: FrameStamp? = null
    ) : ObjectDetector {
        var detectCalls = 0
            private set

        override val isReady: Boolean get() = ready
        override val statusMessage: String = "ready"

        override fun detect(frame: VisionFrame): DetectorFrameResult {
            detectCalls += 1
            onDetect()
            error?.let { throw it }
            return DetectorFrameResult(
                detections = emptyList(),
                frameSize = FrameSize(frame.width, frame.height),
                metrics = DetectorMetrics(
                    totalMs = inferenceMs + 10L,
                    preprocessMs = 4L,
                    inferenceMs = inferenceMs,
                    postprocessMs = 6L,
                    fps = 0f,
                    modelStatus = statusMessage
                ),
                sourceFrame = resultStamp
            )
        }

        override fun detect(bitmap: Bitmap): DetectorFrameResult {
            throw UnsupportedOperationException("Bitmap path is not used in this test")
        }

        override fun close() = Unit
    }

    private class FakeFeedbackGateway : FeedbackGateway {
        var resetCalls = 0
            private set
        var notifyCalls = 0
            private set
        var lastRisk: com.linnan.blindassist.risk.RiskResult? = null
            private set

        override fun resetSession() {
            resetCalls += 1
        }

        override fun notify(
            risk: com.linnan.blindassist.risk.RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifyCalls += 1
            lastRisk = risk
            return FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
        }
    }

    private class InMemoryPreferenceStore : PreferenceStore {
        private val booleans = mutableMapOf<String, Boolean>()
        private val strings = mutableMapOf<String, String>()

        override fun getBoolean(key: String, defaultValue: Boolean): Boolean = booleans[key] ?: defaultValue
        override fun putBoolean(key: String, value: Boolean) {
            booleans[key] = value
        }

        override fun getString(key: String, defaultValue: String): String = strings[key] ?: defaultValue
        override fun putString(key: String, value: String) {
            strings[key] = value
        }
    }

    private companion object {
        fun stamp(frameId: Long, capturedAtMs: Long): FrameStamp = FrameStamp(
            frameId = frameId,
            capturedAtNs = capturedAtMs * 1_000_000L,
            receivedAtNs = (capturedAtMs + 10L) * 1_000_000L,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        )

        fun runtimeConfig(): AssistRuntimeConfig {
            return AssistRuntimeConfig(
                detectionEnabled = true,
                speechEnabled = true,
                vibrationEnabled = true,
                careModeEnabled = false,
                alertProfile = AlertProfile.STANDARD,
                assistScenario = AssistScenario.GENERAL,
                speechStyle = SpeechStyle.STANDARD,
                vibrationStrength = VibrationStrength.STANDARD,
                appLanguage = AppLanguage.ZH,
                dailyUsageMode = DailyUsageMode.GENERAL_DAILY
            )
        }
    }
}
