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
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.VisionFrame
import org.junit.Assert.assertEquals
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

    private fun processor(
        detector: FakeDetector = FakeDetector(),
        stats: FramePipelineStats = FramePipelineStats(),
        active: Boolean = true,
        config: AssistRuntimeConfig = runtimeConfig(),
        onCameraFailure: (String) -> Unit = {}
    ): AssistFrameProcessor {
        val coordinator = AssistSessionCoordinator(feedbackGateway = FakeFeedbackGateway())
        val appViewModel = BlindAssistViewModel(UserPreferences(InMemoryPreferenceStore()))
        val configSnapshot = AssistRuntimeConfigSnapshot(config)
        val guidanceFactory = AssistRuntimeGuidanceFactory(detector) { configSnapshot.get() }
        val renderer = AssistRuntimeRenderer(
            appViewModel = appViewModel,
            detector = detector,
            guidanceFactory = guidanceFactory,
            fieldTestSummaryProvider = FieldTestSummaryProvider(coordinator),
            performanceLogger = AssistRuntimePerformanceLogger(clockMs = { 0L })
        )
        return AssistFrameProcessor(
            detector = detector,
            coordinator = coordinator,
            configSnapshot = configSnapshot,
            renderer = renderer,
            stats = stats,
            isCameraActive = { active },
            runOnUiThread = { it() },
            onCameraFailure = onCameraFailure
        )
    }

    private class FakeVisionFrame(
        override val width: Int = 2,
        override val height: Int = 2,
        override val rotationDegrees: Int = 0
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
        private val onDetect: () -> Unit = {}
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
                )
            )
        }

        override fun detect(bitmap: Bitmap): DetectorFrameResult {
            throw UnsupportedOperationException("Bitmap path is not used in this test")
        }

        override fun close() = Unit
    }

    private class FakeFeedbackGateway : FeedbackGateway {
        override fun notify(
            risk: com.linnan.blindassist.risk.RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
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
