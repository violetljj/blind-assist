package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfSlowLoopEventGate
import com.linnan.blindassist.ustrf.UstrfSlowLoopEventGateResult
import com.linnan.blindassist.ustrf.UstrfSlowLoopEventSuppression
import com.linnan.blindassist.ustrf.UstrfSlowLoopTrigger
import com.linnan.blindassist.ustrf.UstrfSlowLoopTriggerRequest
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Phone-only, reference-free evidence. Camera frames may trigger a bounded slow-loop event,
 * but this test has no semantic model, no navigation output, and no access to the fast loop.
 */
@RunWith(AndroidJUnit4::class)
class UstrfReferenceFreeEventShadowTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun periodicCameraEvents_areFrameBoundThrottled_andNeverSafetyCommands() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val provider = ProcessCameraProvider.getInstance(context).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val frameIds = AtomicLong()
        val frames = AtomicInteger()
        val acceptedEvents = AtomicInteger()
        val sourceBindingsValid = AtomicInteger()
        val suppressions = linkedMapOf<UstrfSlowLoopEventSuppression, Int>()
        val lock = Any()
        val captured = CountDownLatch(REQUIRED_FRAMES)
        val gate = UstrfSlowLoopEventGate(
            minimumInterEventNs = 100_000_000L,
            periodicKeyframeIntervalNs = 500_000_000L,
            eventTtlNs = 1_000_000_000L
        )
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            try {
                val source = UstrfFrameStamp(
                    frameId = frameIds.incrementAndGet(),
                    capturedAtNs = image.imageInfo.timestamp,
                    coordinateFrame = CAMERA_FRAME
                )
                when (val result = gate.request(
                    UstrfSlowLoopTriggerRequest(
                        trigger = UstrfSlowLoopTrigger.PERIODIC_KEYFRAME,
                        sourceFrame = source,
                        observedAtNs = source.capturedAtNs
                    )
                )) {
                    is UstrfSlowLoopEventGateResult.Accepted -> {
                        acceptedEvents.incrementAndGet()
                        if (result.event.queryFrame == source && result.event.validUntilNs > source.capturedAtNs) {
                            sourceBindingsValid.incrementAndGet()
                        }
                    }
                    is UstrfSlowLoopEventGateResult.Suppressed -> synchronized(lock) {
                        suppressions[result.reason] = (suppressions[result.reason] ?: 0) + 1
                    }
                }
                frames.incrementAndGet()
                captured.countDown()
            } finally {
                image.close()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            assertTrue("insufficient CameraX frames", captured.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync { provider.unbindAll(); owner.destroy() }
            analysis.clearAnalyzer()
            analyserExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
        }

        val suppressionJson = synchronized(lock) {
            JSONObject(suppressions.mapKeys { it.key.name })
        }
        val report = JSONObject()
            .put("schema", "blindassist_ustrf_reference_free_event_shadow_v1")
            .put("package", context.packageName)
            .put("capture_timestamp_source", "ImageProxy.imageInfo.timestamp")
            .put("frame_count", frames.get())
            .put("accepted_event_count", acceptedEvents.get())
            .put("frame_bound_event_count", sourceBindingsValid.get())
            .put("suppression_counts", suppressionJson)
            .put("manual_motion_required", false)
            .put("semantic_model_present", false)
            .put("fast_loop_action_present", false)
            .put("authorization", JSONObject()
                .put("reference_free_shadow_only", true)
                .put("metric_geometry_authorized", false)
                .put("production_authorized", false))
        Log.i(TAG, "USTRF_REFERENCE_FREE_EVENT_SHADOW_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("incorrect frame count", frames.get() == REQUIRED_FRAMES)
        assertTrue("no frame-bound events", acceptedEvents.get() > 0 && sourceBindingsValid.get() == acceptedEvents.get())
        assertTrue(
            "periodic throttling did not occur",
            synchronized(lock) { (suppressions[UstrfSlowLoopEventSuppression.PERIODIC_KEYFRAME_INTERVAL_NOT_REACHED] ?: 0) > 0 }
        )
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private fun productionResolutionSelector(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
        .build()

    private companion object {
        const val REQUIRED_FRAMES = 30
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val CAMERA_FRAME = "camera-v1"
        const val REPORT_KEY = "ustrf_reference_free_event_shadow_report"
        const val TAG = "UstrfShadowBenchmark"
    }
}
