package com.linnan.blindassist.hftf

import android.os.Build
import android.os.SystemClock
import android.util.AtomicFile
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.BuildConfig
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.CausalTrackTristateGeometryProducer
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.session.DualLoopCorrectionDecision
import com.linnan.blindassist.session.DualLoopRuntimeMode
import com.linnan.blindassist.session.DualLoopShadowDisposition
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.io.File
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.math.exp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HftfD35DeviceShadowCanaryTest {
    @Test
    fun deviceParityRuntimeAndNonInterference() {
        assertTrue(BuildConfig.DUAL_LOOP_SHADOW)
        assertFalse(BuildConfig.DUAL_LOOP_ACTIVE)
        assertEquals(
            "com.linnan.blindassist.dualloop.shadow",
            BuildConfig.APPLICATION_ID
        )
        val corpus = loadCorpus()
        assertEquals(EXPECTED_ROWS, corpus.rows.size)
        assertEquals(EXPECTED_INPUT_SHA256, corpus.sha256)
        executeParity(corpus.rows, measure = false)
        val parity = executeParity(corpus.rows, measure = true)
        val nonInterference = verifyNonInterference()
        val p50Ms = percentile(parity.latenciesNs, 0.50) / 1_000_000.0
        val p95Ms = percentile(parity.latenciesNs, 0.95) / 1_000_000.0
        val p99Ms = percentile(parity.latenciesNs, 0.99) / 1_000_000.0
        val supported =
            parity.decisionMismatches == 0 &&
                parity.slopePresenceMismatches == 0 &&
                parity.maximumSlopeError <= MAXIMUM_SLOPE_ERROR &&
                p95Ms <= MAXIMUM_P95_MS &&
                nonInterference
        val status =
            if (supported) {
                "D35_ANDROID_DEVICE_SHADOW_PARITY_RUNTIME_" +
                    "NONINTERFERENCE_SUPPORTED"
            } else {
                "D35_ANDROID_DEVICE_SHADOW_PARITY_RUNTIME_" +
                    "NONINTERFERENCE_NOT_SUPPORTED"
            }
        val report = """
            {
              "schema": "blindassist_hftf_stage_c_d35_android_device_shadow_canary_v0",
              "status": "$status",
              "supported": $supported,
              "application_id": "${BuildConfig.APPLICATION_ID}",
              "dual_loop_shadow": ${BuildConfig.DUAL_LOOP_SHADOW},
              "dual_loop_active": ${BuildConfig.DUAL_LOOP_ACTIVE},
              "device_manufacturer": "${jsonEscape(Build.MANUFACTURER)}",
              "device_model": "${jsonEscape(Build.MODEL)}",
              "android_sdk": ${Build.VERSION.SDK_INT},
              "input_sha256": "${corpus.sha256}",
              "row_count": ${corpus.rows.size},
              "distinct_tracks": ${corpus.rows.map { it.sequence to it.trackId }.distinct().size},
              "decision_mismatches": ${parity.decisionMismatches},
              "slope_presence_mismatches": ${parity.slopePresenceMismatches},
              "maximum_absolute_slope_error_per_s": ${parity.maximumSlopeError},
              "producer_call_p50_ms": $p50Ms,
              "producer_call_p95_ms": $p95Ms,
              "producer_call_p99_ms": $p99Ms,
              "maximum_p95_gate_ms": $MAXIMUM_P95_MS,
              "non_interference_passed": $nonInterference,
              "non_actuating": true,
              "future_truth_consumed": false
            }
        """.trimIndent() + "\n"
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val outputDirectory = requireNotNull(
            context.getExternalFilesDir("hftf-d35")
        )
        val output = File(outputDirectory, "report.json")
        val atomicOutput = AtomicFile(output)
        val stream = atomicOutput.startWrite()
        try {
            stream.write(report.toByteArray(Charsets.UTF_8))
            atomicOutput.finishWrite(stream)
        } catch (error: Throwable) {
            atomicOutput.failWrite(stream)
            throw error
        }
        println("D35_REPORT_PATH=${output.absolutePath}")
        println(report.trim())
        assertTrue(status, supported)
    }

    private fun loadCorpus(): Corpus {
        val context = InstrumentationRegistry.getInstrumentation().context
        val uncompressed = context.assets.open(ASSET_NAME).use { raw ->
            GZIPInputStream(raw).use { gzip -> gzip.readBytes() }
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(uncompressed)
            .joinToString("") { "%02x".format(it) }
        val lines = uncompressed.toString(Charsets.UTF_8).lineSequence()
            .filter { it.isNotBlank() }
            .toList()
        require(lines.isNotEmpty())
        val header = lines.first().split('\t')
        val index = header.withIndex().associate { it.value to it.index }
        fun value(parts: List<String>, name: String): String =
            parts[requireNotNull(index[name]) { "D35 missing column $name" }]
        val rows = lines.drop(1).map { line ->
            val parts = line.split('\t')
            Row(
                sequence = value(parts, "sequence"),
                trackId = value(parts, "track_id").toInt(),
                frameIndex = value(parts, "frame_index").toInt(),
                timestampNs = value(parts, "timestamp_ns").toLong(),
                left = value(parts, "left").toFloat(),
                top = value(parts, "top").toFloat(),
                right = value(parts, "right").toFloat(),
                bottom = value(parts, "bottom").toFloat(),
                expectedDecision = value(parts, "expected_decision"),
                expectedSlope = value(parts, "expected_slope_per_s")
                    .takeIf { it.isNotEmpty() }
                    ?.toDouble()
            )
        }
        return Corpus(digest, rows)
    }

    private fun executeParity(rows: List<Row>, measure: Boolean): ParityResult {
        var producer = CausalTrackTristateGeometryProducer()
        var key: Pair<String, Int>? = null
        var previousFrame: Int? = null
        var decisionMismatches = 0
        var slopePresenceMismatches = 0
        var maximumSlopeError = 0.0
        val latencies = ArrayList<Long>(rows.size)
        rows.forEach { row ->
            val currentKey = row.sequence to row.trackId
            if (currentKey != key) {
                producer = CausalTrackTristateGeometryProducer()
                key = currentKey
                previousFrame = null
            }
            if (previousFrame != null && row.frameIndex != previousFrame!! + 1) {
                producer.reset()
            }
            previousFrame = row.frameIndex
            val stamp = stamp(
                sourceId = "d35:${row.sequence}:${row.trackId}",
                frameId = row.frameIndex.toLong(),
                capturedAtNs = row.timestampNs
            )
            val detection = Detection(
                classId = 0,
                label = "person",
                confidence = 1f,
                boundingBox = BoundingBox(
                    row.left,
                    row.top,
                    row.right,
                    row.bottom
                ),
                frameSize = JRDB_FRAME_SIZE
            )
            val started = if (measure) SystemClock.elapsedRealtimeNanos() else 0L
            val actual = requireNotNull(
                producer.produce(
                    stamp,
                    detection,
                    row.timestampNs + 2_000_000L
                )
            )
            if (measure) {
                latencies += SystemClock.elapsedRealtimeNanos() - started
            }
            if (actual.correctionDecision.name != row.expectedDecision) {
                decisionMismatches += 1
            }
            val actualSlope = actual.signedApproachRatePerS?.toDouble()
            if ((actualSlope == null) != (row.expectedSlope == null)) {
                slopePresenceMismatches += 1
            } else if (actualSlope != null && row.expectedSlope != null) {
                maximumSlopeError = maxOf(
                    maximumSlopeError,
                    abs(actualSlope - row.expectedSlope)
                )
            }
        }
        return ParityResult(
            decisionMismatches,
            slopePresenceMismatches,
            maximumSlopeError,
            latencies
        )
    }

    private fun verifyNonInterference(): Boolean {
        val baselineGateway = PlannerGateway()
        val shadowGateway = PlannerGateway()
        val baseline = AssistDecisionKernel()
        val shadow = AssistDecisionKernel()
        baseline.startSession(900L)
        shadow.startSession(900L)
        var lastBaseline = baseline.processFrame(
            emptyList(),
            NON_INTERFERENCE_FRAME_SIZE,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            baselineGateway,
            nowMs = 900L
        )
        var lastShadow = shadow.processFrame(
            emptyList(),
            NON_INTERFERENCE_FRAME_SIZE,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            shadowGateway,
            nowMs = 900L
        )
        repeat(7) { index ->
            val height = 500f * exp(0.30f * index * 0.1f)
            val detection = Detection(
                classId = 0,
                label = "person",
                confidence = 0.95f,
                boundingBox = BoundingBox(
                    350f,
                    200f,
                    650f,
                    200f + height
                ),
                frameSize = NON_INTERFERENCE_FRAME_SIZE
            )
            val frame = stamp(
                "d35:non-interference",
                index.toLong(),
                1_000_000_000L + index * 100_000_000L
            )
            lastBaseline = baseline.processFrame(
                listOf(detection),
                NON_INTERFERENCE_FRAME_SIZE,
                AlertProfile.STANDARD,
                AssistScenario.GENERAL,
                metrics(),
                baselineGateway,
                nowMs = 1_000L + index * 100L,
                sourceFrame = frame,
                decisionAtNs = frame.capturedAtNs + 10_000_000L
            )
            lastShadow = shadow.processFrame(
                listOf(detection),
                NON_INTERFERENCE_FRAME_SIZE,
                AlertProfile.STANDARD,
                AssistScenario.GENERAL,
                metrics(),
                shadowGateway,
                nowMs = 1_000L + index * 100L,
                sourceFrame = frame,
                decisionAtNs = frame.capturedAtNs + 10_000_000L,
                dualLoopMode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY
            )
            if (
                lastBaseline.evaluation.rawRisk != lastShadow.evaluation.rawRisk ||
                lastBaseline.evaluation.stableRisk != lastShadow.evaluation.stableRisk ||
                lastBaseline.evaluation.riskEvent != lastShadow.evaluation.riskEvent ||
                lastBaseline.feedbackDecision != lastShadow.feedbackDecision ||
                lastBaseline.sessionSummary != lastShadow.sessionSummary ||
                baselineGateway.notifyCalls != shadowGateway.notifyCalls
            ) {
                return false
            }
        }
        return lastShadow.evaluation.dualLoopShadow.disposition ==
            DualLoopShadowDisposition.ADMITTED_SHADOW &&
            lastShadow.evaluation.dualLoopShadow.correctionDecision ==
            DualLoopCorrectionDecision.CONFIRM_APPROACH &&
            !lastShadow.evaluation.dualLoopShadow.eventMutationAllowed &&
            !lastShadow.evaluation.dualLoopShadow.feedbackMutationAllowed &&
            lastBaseline.sessionSummary == lastShadow.sessionSummary &&
            baselineGateway.notifyCalls == shadowGateway.notifyCalls
    }

    private fun stamp(
        sourceId: String,
        frameId: Long,
        capturedAtNs: Long
    ) = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs + 1_000_000L,
        sourceId = sourceId,
        coordinateFrame = "jrdb:stitched-rgb",
        clockDomain = FrameClockDomain.REPLAY_TIMELINE
    )

    private fun metrics() = DetectorMetrics(
        totalMs = 3L,
        preprocessMs = 1L,
        inferenceMs = 1L,
        postprocessMs = 1L,
        fps = 30f,
        modelStatus = "ready"
    )

    private fun percentile(values: List<Long>, quantile: Double): Long {
        require(values.isNotEmpty())
        val ordered = values.sorted()
        val index = ((ordered.size - 1) * quantile).toInt()
        return ordered[index]
    }

    private fun jsonEscape(value: String): String =
        value.replace("\\", "\\\\").replace("\"", "\\\"")

    private data class Corpus(val sha256: String, val rows: List<Row>)

    private data class Row(
        val sequence: String,
        val trackId: Int,
        val frameIndex: Int,
        val timestampNs: Long,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val expectedDecision: String,
        val expectedSlope: Double?
    )

    private data class ParityResult(
        val decisionMismatches: Int,
        val slopePresenceMismatches: Int,
        val maximumSlopeError: Double,
        val latenciesNs: List<Long>
    )

    private class PlannerGateway : FeedbackGateway {
        var notifyCalls = 0
            private set

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifyCalls += 1
            val plan = FeedbackPlanner.planFor(
                risk,
                profile = profile,
                scenario = scenario
            )
            return if (plan == null) {
                FeedbackDecision(
                    null,
                    triggered = false,
                    reason = FeedbackReason.NO_FEEDBACK_RISK
                )
            } else {
                FeedbackDecision(
                    plan,
                    triggered = true,
                    reason = FeedbackReason.TRIGGERED
                )
            }
        }
    }

    private companion object {
        const val ASSET_NAME = "hftf_d35_parity_input.tsv.gzbin"
        const val EXPECTED_ROWS = 5_366
        const val EXPECTED_INPUT_SHA256 =
            "d1f24dc7c61890e912d2a4a1cbca23e4b729dfceb1ef76b435cd573c97e6021e"
        const val MAXIMUM_SLOPE_ERROR = 1e-5
        const val MAXIMUM_P95_MS = 0.10
        val JRDB_FRAME_SIZE = FrameSize(3_760, 480)
        val NON_INTERFERENCE_FRAME_SIZE = FrameSize(1_000, 1_000)
    }
}
