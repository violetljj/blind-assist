package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DualLoopShadowAdmitterTest {
    private val frameSize = FrameSize(1000, 1000)
    private val box = BoundingBox(400f, 200f, 600f, 700f)
    private val detection = Detection(
        classId = 0,
        label = "person",
        confidence = 0.9f,
        boundingBox = box,
        frameSize = frameSize
    )
    private val previousFrame = stamp(10L, 1_000_000_000L)
    private val currentFrame = stamp(11L, 1_100_000_000L)
    private val baseline = RiskAnalyzer().analyze(listOf(detection), frameSize)

    @Test
    fun productionAllowlistIsEmptyAndFailsClosed() {
        val result = DualLoopShadowAdmitter().evaluate(
            mode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY,
            evidence = evidence(),
            sourceFrame = currentFrame,
            detections = listOf(detection),
            baselineRisk = baseline,
            decisionAtNs = 1_120_000_000L,
            decisionClockDomain = FrameClockDomain.REPLAY_TIMELINE
        )

        assertEquals(DualLoopShadowDisposition.SOURCE_NOT_ADMITTED, result.disposition)
        assertFalse(result.admitted)
        assertTrue(result.productionRiskUnchanged)
        assertFalse(result.eventMutationAllowed)
        assertFalse(result.feedbackMutationAllowed)
    }

    @Test
    fun offAndMissingEvidenceAreExplicitAndNonActuating() {
        val admitter = admitted()
        val off = evaluate(admitter, mode = DualLoopRuntimeMode.OFF)
        val missing = evaluate(admitter, candidate = null)

        assertEquals(DualLoopShadowDisposition.OFF, off.disposition)
        assertEquals(DualLoopShadowDisposition.EVIDENCE_ABSENT, missing.disposition)
        assertFalse(off.eventMutationAllowed)
        assertFalse(missing.feedbackMutationAllowed)
    }

    @Test
    fun blankSourceIdentityFailsClosed() {
        val result = evaluate(
            admitted(),
            candidate = evidence().copy(sourceId = " ")
        )

        assertEquals(DualLoopShadowDisposition.SOURCE_ID_INVALID, result.disposition)
        assertFalse(result.admitted)
    }

    @Test
    fun allowlistBindsContractAndSourceIdentityAndIsDefensivelyCopied() {
        val identities = mutableSetOf(sourceIdentity())
        val admitter = DualLoopShadowAdmitter(admittedSourceIdentities = identities)
        identities.clear()

        val admitted = evaluate(admitter)
        val forgedSource = evaluate(
            admitter,
            candidate = evidence().copy(sourceId = "forged-source")
        )

        assertEquals(DualLoopShadowDisposition.ADMITTED_SHADOW, admitted.disposition)
        assertEquals(DualLoopShadowDisposition.SOURCE_NOT_ADMITTED, forgedSource.disposition)
    }

    @Test
    fun frameAndTimeFailuresAllAbstain() {
        val admitter = admitted()
        val missingFrame = evaluate(admitter, sourceFrame = null)
        val frameMismatch = evaluate(
            admitter,
            candidate = evidence().copy(currentFrame = stamp(12L, 1_200_000_000L))
        )
        val invalidPrevious = evaluate(
            admitter,
            candidate = evidence().copy(previousFrame = currentFrame)
        )
        val future = evaluate(
            admitter,
            candidate = evidence().copy(availableAtNs = 1_130_000_000L)
        )
        val stale = evaluate(
            admitter,
            candidate = evidence().copy(validUntilNs = 1_110_000_000L)
        )

        assertEquals(DualLoopShadowDisposition.SOURCE_FRAME_MISSING, missingFrame.disposition)
        assertEquals(DualLoopShadowDisposition.CURRENT_FRAME_MISMATCH, frameMismatch.disposition)
        assertEquals(DualLoopShadowDisposition.PREVIOUS_FRAME_INVALID, invalidPrevious.disposition)
        assertEquals(DualLoopShadowDisposition.EVIDENCE_NOT_AVAILABLE, future.disposition)
        assertEquals(DualLoopShadowDisposition.EVIDENCE_STALE, stale.disposition)
    }

    @Test
    fun clockDomainMustBindFrameAvailabilityAndDecision() {
        val mismatch = evaluate(
            admitted(),
            candidate = evidence().copy(
                availabilityClockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
            )
        )

        assertEquals(DualLoopShadowDisposition.CLOCK_DOMAIN_MISMATCH, mismatch.disposition)
    }

    @Test
    fun targetMismatchAndAmbiguityAbstain() {
        val admitter = admitted()
        val mismatch = evaluate(
            admitter,
            candidate = evidence().copy(targetLabel = "chair")
        )
        val ambiguous = evaluate(
            admitter,
            detections = listOf(detection, detection.copy(confidence = 0.8f))
        )

        assertEquals(DualLoopShadowDisposition.TARGET_NOT_SELECTED, mismatch.disposition)
        assertEquals(DualLoopShadowDisposition.TARGET_AMBIGUOUS, ambiguous.disposition)
    }

    @Test
    fun sourceAbstentionNonfiniteAndLowQualityNeverReachAdmission() {
        val admitter = admitted()
        val sourceAbstained = evaluate(
            admitter,
            candidate = evidence().copy(
                signedApproachRatePerS = null,
                quality = null,
                sourceAbstentionReason = "INSUFFICIENT_HISTORY"
            )
        )
        val nonfinite = evaluate(
            admitter,
            candidate = evidence().copy(signedApproachRatePerS = Float.NaN)
        )
        val lowQuality = evaluate(
            admitter,
            candidate = evidence().copy(quality = 0.49f)
        )
        val excessiveQuality = evaluate(
            admitter,
            candidate = evidence().copy(quality = 1.01f)
        )
        val infiniteQuality = evaluate(
            admitter,
            candidate = evidence().copy(quality = Float.POSITIVE_INFINITY)
        )

        assertEquals(DualLoopShadowDisposition.SOURCE_ABSTAINED, sourceAbstained.disposition)
        assertEquals(DualLoopShadowDisposition.NONFINITE_RATE, nonfinite.disposition)
        assertEquals(DualLoopShadowDisposition.LOW_QUALITY, lowQuality.disposition)
        assertEquals(DualLoopShadowDisposition.LOW_QUALITY, excessiveQuality.disposition)
        assertEquals(DualLoopShadowDisposition.LOW_QUALITY, infiniteQuality.disposition)
    }

    @Test
    fun triStateDecisionMustAgreeWithSignedDiagnosticRate() {
        val contradictWithPositiveRate = evaluate(
            admitted(),
            candidate = evidence().copy(
                correctionDecision = DualLoopCorrectionDecision.CONTRADICT_APPROACH
            )
        )
        val explicitAbstainWithRate = evaluate(
            admitted(),
            candidate = evidence().copy(
                correctionDecision = DualLoopCorrectionDecision.ABSTAIN
            )
        )

        assertEquals(
            DualLoopShadowDisposition.DECISION_RATE_MISMATCH,
            contradictWithPositiveRate.disposition
        )
        assertEquals(
            DualLoopShadowDisposition.DECISION_RATE_MISMATCH,
            explicitAbstainWithRate.disposition
        )
    }

    @Test
    fun explicitSourceAbstentionPrecedesDownstreamFrameFailures() {
        val result = evaluate(
            admitted(),
            candidate = evidence().copy(
                currentFrame = stamp(99L, 9_900_000_000L),
                sourceAbstentionReason = "INSUFFICIENT_HISTORY",
                signedApproachRatePerS = null,
                quality = null
            )
        )

        assertEquals(DualLoopShadowDisposition.SOURCE_ABSTAINED, result.disposition)
    }

    @Test
    fun admittedEvidenceRemainsShadowOnly() {
        val result = evaluate(admitted())

        assertEquals(DualLoopShadowDisposition.ADMITTED_SHADOW, result.disposition)
        assertTrue(result.admitted)
        assertEquals(0.25f, result.signedApproachRatePerS)
        assertEquals(0.75f, result.quality)
        assertTrue(result.productionRiskUnchanged)
        assertFalse(result.eventMutationAllowed)
        assertFalse(result.feedbackMutationAllowed)
    }

    private fun admitted() = DualLoopShadowAdmitter(
        admittedSourceIdentities = setOf(sourceIdentity())
    )

    private fun evaluate(
        admitter: DualLoopShadowAdmitter,
        mode: DualLoopRuntimeMode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY,
        candidate: DualLoopGeometryEvidence? = evidence(),
        sourceFrame: FrameStamp? = currentFrame,
        detections: List<Detection> = listOf(detection)
    ) = admitter.evaluate(
        mode = mode,
        evidence = candidate,
        sourceFrame = sourceFrame,
        detections = detections,
        baselineRisk = baseline,
        decisionAtNs = 1_120_000_000L,
        decisionClockDomain = FrameClockDomain.REPLAY_TIMELINE
    )

    private fun evidence() = DualLoopGeometryEvidence(
        sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
        sourceId = "synthetic-shadow-fixture",
        previousFrame = previousFrame,
        currentFrame = currentFrame,
        availableAtNs = 1_110_000_000L,
        validUntilNs = 1_200_000_000L,
        availabilityClockDomain = FrameClockDomain.REPLAY_TIMELINE,
        trackEpoch = "fixture-epoch-1",
        targetClassId = detection.classId,
        targetLabel = detection.label,
        targetBoundingBox = detection.boundingBox,
        targetFrameSize = detection.frameSize,
        targetSource = DetectionSource.OBJECT_DETECTOR,
        correctionDecision = DualLoopCorrectionDecision.CONFIRM_APPROACH,
        signedApproachRatePerS = 0.25f,
        quality = 0.75f
    )

    private fun sourceIdentity() = DualLoopSourceIdentity(
        sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
        sourceId = "synthetic-shadow-fixture"
    )

    private fun stamp(frameId: Long, capturedAtNs: Long) = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs + 1_000_000L,
        sourceId = "synthetic-camera",
        coordinateFrame = "camera-optical",
        clockDomain = FrameClockDomain.REPLAY_TIMELINE
    )
}
